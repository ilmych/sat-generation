"""
Module for fixing SAT questions that fail validation.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from anthropic import Anthropic
from tqdm import tqdm
import re
import time
import importlib
import concurrent.futures

class QuestionFixer:
    """
    Class for fixing SAT questions that fail validation.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the QuestionFixer.
        
        Args:
            api_key: Anthropic API key. If None, it's loaded from environment variables.
            model_name: Name of the Claude model to use. If None, it's loaded from environment variables.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set it either directly or via ANTHROPIC_API_KEY environment variable.")
        
        # Use the default model for fixing
        self.model_name = model_name or os.getenv("MODEL_NAME", "claude-3-sonnet-20240229")
        
        # Use Claude 3.5 Sonnet for explanations (consistently with the validator)
        self.explanation_model = os.getenv("EXPLANATION_MODEL", "claude-3-5-sonnet-20240307")
        
        # Initialize the Anthropic client
        self.client = Anthropic(api_key=self.api_key)
        
        # Maximum number of retries for fixing a question
        self.max_retries = int(os.getenv("MAX_VALIDATION_RETRIES", "3"))
    
    def fix_question(self, question: Dict, validation_results: Dict) -> Dict:
        """
        Fix a question that failed validation.
        
        Args:
            question: The question dictionary to fix.
            validation_results: Validation results for the question.
            
        Returns:
            A dictionary with the fixed question.
        """
        # Validate input
        if not isinstance(validation_results, dict):
            print(f"Warning: validation_results is not a dictionary: {type(validation_results).__name__}")
            return question
            
        # Check if there's a failed image validation for COEQ questions
        question_type = question.get("type", "")
        skill = question.get("skill", "")
        
        # Special handling for COEQ image failures
        if (question_type == "general" and skill == "Command of Evidence: Quantitative") or "coeq" in question_type:
            if "image-readability" in validation_results and isinstance(validation_results["image-readability"], dict) and validation_results["image-readability"].get("score", 1) == 0:
                print("Detected failed image validation for COEQ question - attempting to fix the image.")
                return self.fix_coeq_image(question, validation_results["image-readability"])
                
        # Standard handling for other validation failures
        # Find the failed validations
        failed_validations = {}
        for name, result in validation_results.items():
            # Skip if name is "error" or result is not a dictionary
            if name == "error" or not isinstance(result, dict):
                continue
                
            score = result.get("score", 0)
            has_error = result.get("error", False)
            
            if score == 0 and not has_error and name != "image-readability":
                failed_validations[name] = result
        
        if not failed_validations:
            # No failed validations or only errors
            return question
        
        # Construct a prompt with the question and the failed validations
        fix_prompt = self._construct_fix_prompt(question, failed_validations)
        
        # Call the API to fix the question
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                temperature=0.2,  # Use low temperature for consistent fixes
                system="You are a professional SAT question writer who fixes issues with SAT questions to ensure they meet quality standards.",
                messages=[
                    {
                        "role": "user",
                        "content": fix_prompt
                    }
                ]
            )
            
            # Extract the fixed question from the response
            fixed_question_json_str = self._extract_json_from_response(response.content[0].text)
            if not fixed_question_json_str:
                raise ValueError("No valid JSON found in the response.")
            
            fixed_question = json.loads(fixed_question_json_str)
            
            # Ensure all original fields are preserved
            for key in question:
                if key not in fixed_question:
                    fixed_question[key] = question[key]
            
            print("Question has been fixed. It will now be validated.")
            return fixed_question
            
        except Exception as e:
            print(f"Error fixing question: {str(e)}")
            return question

    def fix_coeq_image(self, question: Dict, image_validation_result: Dict) -> Dict:
        """
        Fix a COEQ question with a failed image validation.
        
        Args:
            question: The question dictionary.
            image_validation_result: The validation result for the image.
            
        Returns:
            The question with an updated image.
        """
        # Try to find the original code for this image
        passage = question.get("passage", "")
        img_pattern = re.compile(r'<img[^>]*src=["\'](.*?)["\'][^>]*>')
        img_match = img_pattern.search(passage)
        
        if not img_match:
            print("Warning: No image URL found in the question passage.")
            return question
            
        image_url = img_match.group(1)
        print(f"Fixing image: {image_url}")
        
        # Extract image ID from Google Drive URL
        img_id_match = re.search(r'id=([^&]+)', image_url)
        if not img_id_match:
            print("Warning: Could not extract image ID from URL. Cannot fix image.")
            return question
            
        img_id = img_id_match.group(1)
        
        # Find the corresponding code file based on timestamp pattern in image filename
        timestamp_pattern = r'visualization_(\d{8}_\d{6})'
        
        # Try to find the original code file
        code_files = []
        for filename in os.listdir("generated_code"):
            if filename.endswith(".py"):
                code_files.append(filename)
        
        # Sort files by modification time (newest first)
        code_files.sort(key=lambda f: os.path.getmtime(os.path.join("generated_code", f)), reverse=True)
        
        # Try to find the matching code file
        original_code = None
        code_filename = None
        
        for filename in code_files:
            # First check if we can match by timestamp in the image URL
            timestamp_match = re.search(timestamp_pattern, filename)
            if timestamp_match:
                # Read the code file
                with open(os.path.join("generated_code", filename), 'r', encoding='utf-8') as f:
                    original_code = f.read()
                    code_filename = filename
                    break
                    
        if not original_code:
            print("Warning: Could not find original code file for the image. Using default fix.")
            return self._fix_coeq_question_without_code(question, image_validation_result)
        
        print(f"Found original code file: {code_filename}")
        
        try:
            # Import the validator module to use its fix method
            from src.validator import QuestionValidator
            validator = QuestionValidator(api_key=self.api_key, model_name=self.model_name)
            
            # Get the improvement suggestions
            improvement_suggestions = image_validation_result.get("improvement_suggestions", [])
            if not improvement_suggestions:
                print("No improvement suggestions found in validation result. Cannot fix image.")
                return question
                
            # Generate improved code
            improved_code, error = validator.fix_coeq_image(
                question, 
                original_code, 
                improvement_suggestions
            )
            
            if error:
                print(f"Error fixing image code: {error}")
                return question
                
            # Save the improved code
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            improved_code_filename = f"generated_code/visualization_fixed_{timestamp}.py"
            
            with open(improved_code_filename, 'w', encoding='utf-8') as f:
                f.write(improved_code)
                
            # Execute the improved code to generate a new image
            improved_image_filename = f"generated_code/visualization_fixed_{timestamp}.png"
            
            # Add the image filename to the code
            improved_code = improved_code.replace(
                "plt.savefig('visualization.png')", 
                f"plt.savefig('{improved_image_filename}')"
            )
            
            # Execute the code
            exec_globals = {'__name__': '__main__'}
            exec(improved_code, exec_globals)
            print(f"Fixed visualization saved to {improved_image_filename}")
            
            # Upload the new image to Google Drive
            # Import the generator to use its upload method
            from src.generator import QuestionGenerator
            generator = QuestionGenerator(api_key=self.api_key, model_name=self.model_name)
            
            new_image_url = generator.upload_to_gdrive(improved_image_filename)
            
            if not new_image_url:
                print("Warning: Failed to upload fixed image to Google Drive.")
                return question
                
            # Update the question's img tag with the new URL
            img_pattern = re.compile(r'<img[^>]*src=["\'](.*?)["\'][^>]*>')
            updated_passage = img_pattern.sub(
                lambda m: m.group(0).replace(f'src="{image_url}"', f'src="{new_image_url}"'),
                passage
            )
            
            # Update the question
            question["passage"] = updated_passage
            
            print("Successfully fixed and replaced the image in the question.")
            return question
            
        except Exception as e:
            print(f"Error during image fixing process: {str(e)}")
            return question
    
    def _fix_coeq_question_without_code(self, question: Dict, image_validation_result: Dict) -> Dict:
        """
        Fallback method to fix a COEQ question when the original code can't be found.
        
        Args:
            question: The question dictionary.
            image_validation_result: The validation result for the image.
            
        Returns:
            The question with adjustments that don't rely on fixing the image.
        """
        # If we can't fix the image, we'll try to adjust the question text instead
        try:
            # Create a prompt to adjust the question to work without relying on the problematic image
            adjustment_prompt = f"""
            # SAT Question Adjustment for Image Issues
            
            I have a Command of Evidence: Quantitative (COEQ) SAT question where the data visualization has readability issues.
            I cannot fix the image directly, so I need to adjust the question to work with the existing image.
            
            ## Original Question
            
            ```json
            {json.dumps(question, indent=2)}
            ```
            
            ## Image Readability Issues
            
            {image_validation_result.get('reasoning', 'The image has readability issues.')}
            
            ## Instructions
            
            1. Adjust the question to accommodate the existing image's limitations.
            2. Simplify the question if necessary to focus on clearly visible elements in the image.
            3. Add clarifying text in the passage to help students understand what they should be looking at.
            4. Do not change the fundamental concept or skill being tested.
            5. Maintain the same difficulty level.
            6. Return the modified question in the same JSON format as the original.
            
            Please provide the adjusted question in valid JSON format.
            """
            
            # Call the API to adjust the question
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                temperature=0.2,
                system="You are a professional SAT question writer who adjusts questions to accommodate limitations in their visual elements.",
                messages=[
                    {
                        "role": "user",
                        "content": adjustment_prompt
                    }
                ]
            )
            
            # Extract the adjusted question
            adjusted_question_json_str = self._extract_json_from_response(response.content[0].text)
            if not adjusted_question_json_str:
                print("Failed to generate adjusted question.")
                return question
                
            adjusted_question = json.loads(adjusted_question_json_str)
            
            # Ensure all original fields are preserved
            for key in question:
                if key not in adjusted_question:
                    adjusted_question[key] = question[key]
            
            print("Successfully adjusted the question to accommodate image limitations.")
            return adjusted_question
            
        except Exception as e:
            print(f"Error adjusting question: {str(e)}")
            return question
    
    def fix_questions(self, validated_questions: List[Dict]) -> List[Dict]:
        """
        Fix multiple questions that failed validation.
        Uses parallel processing to fix multiple questions simultaneously.
        After fixing, validates the fixed questions to ensure they pass all checks.
        Questions that still fail after max_retries attempts will be excluded from the final output.
        
        Args:
            validated_questions: A list of dictionaries with questions and their validation results.
            
        Returns:
            A list of dictionaries with fixed questions and their validation results.
        """
        # Import validator here to avoid circular imports
        from src.validator import QuestionValidator
        
        # Initialize validator with the same API key and model
        validator = QuestionValidator(api_key=self.api_key, model_name=self.model_name)
        
        # Define a worker function to fix a single question
        def fix_worker(i, validated):
            question = validated["question"]
            validation_results = validated["validation_results"]
            passed_all = validated["passed_all"]
            
            if passed_all:
                # Question passed all validations, no need to fix
                return {
                    "index": i,
                    "question": question,
                    "validation_results": validation_results,
                    "passed_all": passed_all,
                    "fixed": False,
                    "retries": 0,
                    "include_in_output": True  # Already passing, include in final output
                }
            
            # Try to fix the question up to max_retries times
            retries = 0
            current_question = question
            current_validation_results = validation_results
            current_passed_all = passed_all
            question_type = current_question.get("type", "")
            
            while not current_passed_all and retries < self.max_retries:
                try:
                    print(f"Fixing question {i+1}, attempt {retries+1}/{self.max_retries}...")
                    
                    # Fix the question
                    fixed_question = self.fix_question(current_question, current_validation_results)
                    
                    # Increment retry counter
                    retries += 1
                    
                    # Now validate the fixed question to see if the fixes worked
                    print(f"Validating fixed question {i+1} (attempt {retries})...")
                    
                    # Get base question type for validation
                    base_type = question_type
                    if "-" in question_type:
                        base_type = question_type.split("-")[0]
                    
                    # Run validation on the fixed question
                    new_validation_results = validator.validate_question(fixed_question)
                    new_passed_all = new_validation_results.get("passed", False)
                    
                    # Update current state
                    current_question = fixed_question
                    current_validation_results = new_validation_results
                    current_passed_all = new_passed_all
                    
                    if new_passed_all:
                        print(f"Fixed question {i+1} now passes all validations!")
                        
                        # Generate explanations for the fixed question
                        print(f"Generating explanations for fixed question {i+1}...")
                        explanations = validator.generate_explanation(fixed_question)
                        
                        # Add explanations to the fixed question
                        for key, explanation in explanations.items():
                            if not key.startswith("error"):  # Don't add error flag to question
                                fixed_question[key] = explanation
                        
                        return {
                            "index": i,
                            "question": fixed_question,
                            "validation_results": new_validation_results,
                            "passed_all": True,
                            "fixed": True,
                            "retries": retries,
                            "include_in_output": True  # Successfully fixed, include in final output
                        }
                    else:
                        # Continue to next retry if we haven't reached max_retries
                        if retries < self.max_retries:
                            print(f"Fixed question {i+1} still fails validation. Retrying...")
                            # Continue to next iteration of while loop
                        else:
                            print(f"Failed to fix question {i+1} after {retries} attempts.")
                            return {
                                "index": i,
                                "question": current_question,
                                "validation_results": current_validation_results,
                                "passed_all": False,
                                "fixed": True,
                                "retries": retries,
                                "include_in_output": False  # Failed after all retries, exclude from final output
                            }
                    
                except Exception as e:
                    print(f"Error fixing/validating question {i+1}: {str(e)}")
                    retries += 1
            
            # Return the best result we have after all retries
            return {
                "index": i,
                "question": current_question,
                "validation_results": current_validation_results,
                "passed_all": current_passed_all,
                "fixed": True,
                "retries": retries,
                "include_in_output": False  # Failed after all retries, exclude from final output
            }
        
        # Get the maximum number of workers from environment or default to CPU count
        max_workers = int(os.getenv("MAX_FIX_WORKERS", os.cpu_count() or 4))
        print(f"\nUsing up to {max_workers} parallel fix workers")
        
        fixed_questions = []
        
        # Use thread pool executor for API calls (better for I/O bound tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_index = {
                executor.submit(fix_worker, i, validated): i 
                for i, validated in enumerate(validated_questions)
            }
            
            # Collect results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                             desc="Fixing failed questions", total=len(validated_questions)):
                result = future.result()
                fixed_questions.append(result)
        
        # Sort results by original index to maintain order
        fixed_questions.sort(key=lambda x: x["index"])
        
        # Calculate validation stats after fixing
        total_questions = len(fixed_questions)
        passed_count = sum(1 for q in fixed_questions if q["passed_all"])
        fixed_count = sum(1 for q in fixed_questions if q.get("fixed", False) and q.get("passed_all", False))
        still_failing = total_questions - passed_count
        
        print(f"\nFix Results Summary:")
        print(f"Total questions processed: {total_questions}")
        print(f"Questions that now pass validation: {passed_count} ({100 * passed_count / total_questions:.1f}%)")
        print(f"Questions that were successfully fixed: {fixed_count}")
        print(f"Questions that still have validation issues: {still_failing} ({100 * still_failing / total_questions:.1f}%)")
        
        if still_failing > 0:
            print(f"\nWARNING: {still_failing} questions could not be fixed after {self.max_retries} attempts.")
            print("These questions will be excluded from the final output but retained in validation_details.")
        
        # Remove the temporary index field from results
        for result in fixed_questions:
            result.pop("index", None)
        
        return fixed_questions
    
    def _construct_fix_prompt(self, question: Dict, failed_validations: Dict[str, Dict]) -> str:
        """
        Construct a prompt for fixing a question that failed validation.
        
        Args:
            question: The question dictionary to fix.
            failed_validations: Validation results for the failed validations.
            
        Returns:
            A string with the fix prompt.
        """
        prompt = """
        # SAT Question Fix Request

        I have an SAT question that failed some validation checks. Please fix the question to address all the failed validations.

        ## Original Question
        
        ```json
        {question_json}
        ```

        ## Failed Validations

        {failed_validations}

        ## Instructions

        1. Fix the question to address ALL of the failed validations.
        2. Do not change aspects of the question that weren't identified as problematic.
        3. Return the fixed question in the same JSON format as the original question.
        4. Maintain the same question type and difficulty level.
        5. Make the minimal changes necessary to fix the issues.

        ## Fixed Question

        Please provide the fixed question in valid JSON format:
        
        """
        
        # Format the failed validations as a list
        failed_validations_text = ""
        for name, result in failed_validations.items():
            failed_validations_text += f"### Validation: {name}\n"
            failed_validations_text += f"* Score: {result.get('score', 0)}\n"
            failed_validations_text += f"* Reasoning: {result.get('reasoning', 'No reasoning provided.')}\n\n"
        
        # Fill in the template
        return prompt.format(
            question_json=json.dumps(question, indent=2, ensure_ascii=False),
            failed_validations=failed_validations_text
        )
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        Extract a JSON string from the response text.
        
        Args:
            response_text: The text response from Claude.
            
        Returns:
            The extracted JSON string, or None if no JSON found.
        """
        # Try to find JSON in the response
        try:
            # Look for JSON object start and end
            start_idx = response_text.find('{')
            if start_idx == -1:
                return None
            
            # Find the matching closing brace
            brace_count = 0
            for i in range(start_idx, len(response_text)):
                if response_text[i] == '{':
                    brace_count += 1
                elif response_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return response_text[start_idx:i+1]
            
            return None
        except Exception:
            return None
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """
        Extract Python code from the response text.
        
        Args:
            response_text: The text response from Claude.
            
        Returns:
            The extracted Python code as a string.
        """
        import re
        
        # Look for Python code blocks
        code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)
        code_match = code_pattern.search(response_text)
        
        if code_match:
            return code_match.group(1).strip()
        
        # If no Python-specific code block, try generic code blocks
        generic_code_pattern = re.compile(r'```\s*(.*?)\s*```', re.DOTALL)
        generic_match = generic_code_pattern.search(response_text)
        
        if generic_match:
            return generic_match.group(1).strip()
        
        # If no code blocks found, return the whole response
        return response_text 