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
            
        # Identify key pieces of information
        question_type = question.get("type", "")
        skill = question.get("skill", "")
        is_coeq = (question_type == "coeq" or "coeq" in question_type or 
                  skill == "Command of Evidence: Quantitative")
        
        # Separate COEQ image validation from other validations
        image_validation_failed = False
        other_validations_failed = False
        
        # Get results detail dictionary
        details = validation_results.get("details", {})
        
        # Check if the COEQ image validation failed
        if is_coeq and "coeq_image" in details:
            image_validation = details["coeq_image"]
            if isinstance(image_validation, dict) and image_validation.get("score", 1) == 0:
                image_validation_failed = True
                print("COEQ image validation failed")
        
        # Check if other validations failed
        for name, result in details.items():
            if name != "coeq_image" and isinstance(result, dict):
                score = result.get("score", 1)
                has_error = result.get("error", False)
                
                if score == 0 and not has_error:
                    other_validations_failed = True
                    print(f"Validation '{name}' failed")
        
        # Import validator for revalidation
        from src.validator import QuestionValidator
        validator = QuestionValidator(api_key=self.api_key, model_name=self.model_name)
        
        # Now handle each case based on the specified rules
        if is_coeq:
            print("Processing COEQ question with special handling")
            
            # Case 1: Only image validation failed
            if image_validation_failed and not other_validations_failed:
                print("CASE 1: Only image validation failed - fixing and revalidating only the image")
                
                # Fix the image
                fixed_question = self.fix_coeq_image(question, details["coeq_image"])
                
                # Revalidate only the image
                image_validation_result = validator.validate_coeq_image_only(fixed_question)
                
                # Check if the fix worked
                if image_validation_result.get("passed", False):
                    print("✅ Image fix successful")
                    return fixed_question
                else:
                    print("❌ Image fix failed")
                    # Return the original question if the fix didn't work
                    return question
            
            # Case 2: Image validation passed but other validations failed
            elif not image_validation_failed and other_validations_failed:
                print("CASE 2: Other validations failed but image validation passed - fixing other issues")
                
                # Filter out image validation to create failed validations dictionary
                failed_validations = {}
                for name, result in details.items():
                    if name != "coeq_image" and isinstance(result, dict):
                        score = result.get("score", 1)
                        has_error = result.get("error", False)
                        
                        if score == 0 and not has_error:
                            failed_validations[name] = result
                
                # Fix the other issues
                fix_prompt = self._construct_fix_prompt(question, failed_validations)
                
                try:
                    # Call the API to fix the question
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=4000,
                        temperature=0.2,
                        system="You are a professional SAT question writer who fixes issues with SAT questions to ensure they meet quality standards. DO NOT modify any <img> tags or image-related aspects of the question.",
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
                    
                    fixed_question = fixed_question_json_str
                    
                    # Ensure all original fields are preserved, especially the image-related parts
                    for key in question:
                        if key not in fixed_question:
                            fixed_question[key] = question[key]
                    
                    # Make sure the passage with the image is preserved
                    if "passage" in question and "<img" in question["passage"]:
                        original_passage = question["passage"]
                        fixed_passage = fixed_question.get("passage", "")
                        
                        # If the img tag is missing from the fixed passage, restore it
                        if "<img" not in fixed_passage:
                            print("WARNING: Fixed question is missing the image tag. Restoring from original.")
                            fixed_question["passage"] = original_passage
                        elif "<img" in original_passage and "<img" in fixed_passage:
                            # Extract the img tag from the original
                            import re
                            img_pattern = re.compile(r'<img[^>]*?>')
                            original_img_match = img_pattern.search(original_passage)
                            
                            if original_img_match:
                                original_img_tag = original_img_match.group(0)
                                
                                # Replace the img tag in the fixed passage
                                fixed_question["passage"] = img_pattern.sub(original_img_tag, fixed_passage)
                    
                    # Now validate the other aspects again, but skip image validation
                    print("Validating the fixed question (skipping image validation)")
                    new_validation_results = validator.validate_question(fixed_question)
                    
                    # Check if the only remaining failures are related to the image
                    new_details = new_validation_results.get("details", {})
                    non_image_failures = False
                    
                    for name, result in new_details.items():
                        if name != "coeq_image" and isinstance(result, dict):
                            score = result.get("score", 1)
                            has_error = result.get("error", False)
                            
                            if score == 0 and not has_error:
                                non_image_failures = True
                                print(f"Validation '{name}' still failing after fix")
                    
                    if not non_image_failures:
                        print("✅ Non-image fixes successful")
                    else:
                        print("❌ Some non-image fixes failed")
                        
                    return fixed_question
                    
                except Exception as e:
                    print(f"Error fixing question: {str(e)}")
                    return question
            
            # Case 3: Both image validation and other validations failed
            elif image_validation_failed and other_validations_failed:
                print("CASE 3: Both image validation and other validations failed - fixing both")
                
                # Step 1: Fix the other issues first
                failed_validations = {}
                for name, result in details.items():
                    if name != "coeq_image" and isinstance(result, dict):
                        score = result.get("score", 0)
                        has_error = result.get("error", False)
                        
                        if score == 0 and not has_error:
                            failed_validations[name] = result
                
                # Fix the non-image issues
                if failed_validations:
                    fix_prompt = self._construct_fix_prompt(question, failed_validations)
                    
                    try:
                        # Call the API to fix the non-image issues
                        response = self.client.messages.create(
                            model=self.model_name,
                            max_tokens=4000,
                            temperature=0.2,
                            system="You are a professional SAT question writer who fixes issues with SAT questions to ensure they meet quality standards. DO NOT modify any <img> tags or image-related aspects of the question.",
                            messages=[
                                {
                                    "role": "user",
                                    "content": fix_prompt
                                }
                            ]
                        )
                        
                        # Extract the fixed question
                        fixed_question_json_str = self._extract_json_from_response(response.content[0].text)
                        if not fixed_question_json_str:
                            raise ValueError("No valid JSON found in the response.")
                        
                        question_with_fixes = fixed_question_json_str
                        
                        # Ensure all original fields are preserved
                        for key in question:
                            if key not in question_with_fixes:
                                question_with_fixes[key] = question[key]
                        
                        # Make sure the passage with the image is preserved
                        if "passage" in question and "<img" in question["passage"]:
                            original_passage = question["passage"]
                            fixed_passage = question_with_fixes.get("passage", "")
                            
                            # If the img tag is missing from the fixed passage, restore it
                            if "<img" not in fixed_passage:
                                print("WARNING: Fixed question is missing the image tag. Restoring from original.")
                                question_with_fixes["passage"] = original_passage
                            elif "<img" in original_passage and "<img" in fixed_passage:
                                # Extract the img tag from the original
                                import re
                                img_pattern = re.compile(r'<img[^>]*?>')
                                original_img_match = img_pattern.search(original_passage)
                                
                                if original_img_match:
                                    original_img_tag = original_img_match.group(0)
                                    
                                    # Replace the img tag in the fixed passage
                                    question_with_fixes["passage"] = img_pattern.sub(original_img_tag, fixed_passage)
                        
                        # Update the question to the one with non-image fixes
                        question = question_with_fixes
                        print("Non-image fixes applied")
                        
                    except Exception as e:
                        print(f"Error fixing non-image issues: {str(e)}")
                
                # Step 2: Now fix the image
                fixed_question = self.fix_coeq_image(question, details["coeq_image"])
                
                # Step 3: Validate both aspects
                print("Validating the completely fixed question")
                new_validation_results = validator.validate_question(fixed_question)
                
                if new_validation_results.get("passed", False):
                    print("✅ All fixes successful - question now passes all validations")
                else:
                    print("❌ Some validations still failing after fixes")
                
                return fixed_question
            
            # Case 4: All validations passed, nothing to fix
            else:
                print("CASE 4: All validations passed - no fixes needed")
                return question
        
        # Handle non-COEQ questions with standard approach
        else:
            print("Standard handling for non-COEQ question")
            
            # Find the failed validations
            failed_validations = {}
            for name, result in details.items():
                if name == "error" or not isinstance(result, dict):
                    continue
                    
                score = result.get("score", 0)
                has_error = result.get("error", False)
                
                if score == 0 and not has_error:
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
                
                fixed_question = fixed_question_json_str
                
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
        print("Attempting to fix COEQ image...")
        
        # Ensure we have a valid validation result
        if not isinstance(image_validation_result, dict):
            print("Warning: Invalid image validation result format. Cannot fix image.")
            return question
        
        # Try to find the original code for this image
        passage = question.get("passage", "")
        img_pattern = re.compile(r'<img[^>]*src=["\'](.*?)["\'][^>]*>')
        img_match = img_pattern.search(passage)
        
        if not img_match:
            print("Warning: No image URL found in the question passage.")
            return question
            
        image_url = img_match.group(1)
        print(f"Fixing image: {image_url}")
        
        # Check if visualization code is available in the question
        original_code = question.get("visualization_code")
        if original_code:
            print("Using visualization code from the question")
        else:
            # If not available in the question, try to find it in the generated_code folder
            
            # First check if we have a local file path
            local_path = question.get("local_image_path")
            if local_path and os.path.exists(local_path):
                # Try to find the corresponding .py file
                code_filename = local_path.replace(".png", ".py")
                if os.path.exists(code_filename):
                    try:
                        with open(code_filename, 'r', encoding='utf-8') as f:
                            original_code = f.read()
                            print(f"Found corresponding code file: {code_filename}")
                    except Exception as e:
                        print(f"Error reading code file: {str(e)}")
            
            # If still no code, try to find it by looking for Google Drive IDs
            if not original_code and "drive.google.com" in image_url:
                # Extract image ID from Google Drive URL
                img_id_match = re.search(r'id=([^&]+)', image_url)
                if img_id_match:
                    img_id = img_id_match.group(1)
                    print(f"Extracted Google Drive image ID: {img_id}")
                    
                    # Find the corresponding code file based on timestamp pattern in image filename
                    timestamp_pattern = r'visualization_(\d{8}_\d{6})'
                    
                    # Try to find the original code file
                    code_files = []
                    try:
                        for filename in os.listdir("generated_code"):
                            if filename.endswith(".py"):
                                code_files.append(filename)
                    except FileNotFoundError:
                        print("Warning: 'generated_code' directory not found")
                        code_files = []
                    
                    # Sort files by modification time (newest first)
                    code_files.sort(key=lambda f: os.path.getmtime(os.path.join("generated_code", f)), reverse=True)
                    
                    # Try to find the matching code file
                    for filename in code_files:
                        # First check if we can match by timestamp in the image URL
                        timestamp_match = re.search(timestamp_pattern, filename)
                        if timestamp_match:
                            # Read the code file
                            try:
                                with open(os.path.join("generated_code", filename), 'r', encoding='utf-8') as f:
                                    original_code = f.read()
                                    print(f"Found original code file: {filename}")
                                    break
                            except Exception as e:
                                print(f"Error reading code file: {str(e)}")
        
        # If we couldn't find the original code, use a fallback approach
        if not original_code:
            print("Warning: Could not find original code for the image. Using default fix.")
            return self._fix_coeq_question_without_code(question, image_validation_result)
        
        try:
            # Import the validator module to use its fix method
            from src.validator import QuestionValidator
            validator = QuestionValidator(api_key=self.api_key, model_name=self.model_name)
            
            # Get the improvement suggestions
            # Different validation structures may have the suggestions at different levels
            improvement_suggestions = []
            
            # Try to find improvement suggestions in various formats
            if "improvement_suggestions" in image_validation_result:
                improvement_suggestions = image_validation_result["improvement_suggestions"]
            elif "details" in image_validation_result and "improvement_suggestions" in image_validation_result.get("details", {}):
                improvement_suggestions = image_validation_result["details"]["improvement_suggestions"]
            elif "reasoning" in image_validation_result:
                # Try to parse suggestions from reasoning
                reasoning = image_validation_result["reasoning"]
                # Extract suggestions from the reasoning text, if any
                suggestion_lines = [line.strip() for line in reasoning.split('\n') if line.strip().startswith('-')]
                improvement_suggestions = suggestion_lines
            
            if not improvement_suggestions:
                print("No improvement suggestions found in validation result. Generating default suggestions.")
                improvement_suggestions = [
                    "Increase font size for all text elements to improve readability",
                    "Adjust color contrast to make data points more distinct",
                    "Add more padding around elements to prevent overlap",
                    "Make axis labels and titles more prominent",
                    "Ensure consistent formatting throughout the visualization"
                ]
                
            print(f"Found {len(improvement_suggestions)} improvement suggestions")
                
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
            
            # Create the directory if it doesn't exist
            os.makedirs("generated_code", exist_ok=True)
            
            with open(improved_code_filename, 'w', encoding='utf-8') as f:
                f.write(improved_code)
                
            # Execute the improved code to generate a new image
            improved_image_filename = f"generated_code/visualization_fixed_{timestamp}.png"
            
            # Add the image filename to the code
            if "plt.savefig" in improved_code:
                modified_code = improved_code.replace(
                "plt.savefig('visualization.png')", 
                f"plt.savefig('{improved_image_filename}')"
                ).replace(
                    'plt.savefig("visualization.png")', 
                    f'plt.savefig("{improved_image_filename}")'
            )
            else:
                # If no savefig call is found, add one at the end
                modified_code = improved_code + f"\n\n# Save the figure\nplt.savefig('{improved_image_filename}')\n"
            
            # Execute the code
            exec_globals = {'__name__': '__main__'}
            try:
                print(f"Executing improved visualization code to generate: {improved_image_filename}")
                # Make sure we have matplotlib imported in the global namespace
                exec('import matplotlib.pyplot as plt', exec_globals)
                exec('import numpy as np', exec_globals)
                exec('import pandas as pd', exec_globals)
                
                # Execute the visualization code
                exec(modified_code, exec_globals)
                
                # Check if the file was created
                if os.path.exists(improved_image_filename):
                    print(f"Successfully created image file: {improved_image_filename}")
                else:
                    print(f"Warning: Image file not created by code execution. Trying direct save method.")
                    # Try to directly access the figure and save it
                    if 'plt' in exec_globals:
                        plt = exec_globals['plt']
                        plt.savefig(improved_image_filename, dpi=300, bbox_inches='tight')
                        print(f"Direct save method: Saved figure to {improved_image_filename}")
                        
                        # Verify the file was created
                        if os.path.exists(improved_image_filename):
                            print(f"Confirmed image file exists after direct save: {improved_image_filename}")
                        else:
                            print(f"Error: Image file still not created after direct save attempt")
                            # Try one more approach - create a simple figure and save it
                            plt.figure(figsize=(8, 6))
                            plt.title("SAT Question Visualization")
                            plt.text(0.5, 0.5, "Placeholder visualization - original could not be generated", 
                                    ha='center', va='center', fontsize=12)
                            plt.axis('off')
                            plt.savefig(improved_image_filename, dpi=300, bbox_inches='tight')
                            print(f"Created placeholder image as last resort: {improved_image_filename}")
                    else:
                        print("Error: plt not found in execution namespace")
                        # Create a simple matplotlib figure as a fallback
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(8, 6))
                        plt.title("SAT Question Visualization")
                        plt.text(0.5, 0.5, "Placeholder visualization - original could not be generated", 
                                ha='center', va='center', fontsize=12)
                        plt.axis('off')
                        plt.savefig(improved_image_filename, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Created fallback placeholder image: {improved_image_filename}")
            except Exception as e:
                print(f"Error executing visualization code: {str(e)}")
                # Create a simple matplotlib figure as a fallback
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8, 6))
                    plt.title("SAT Question Visualization")
                    plt.text(0.5, 0.5, f"Error generating visualization: {str(e)}", 
                            ha='center', va='center', fontsize=12)
                    plt.axis('off')
                    plt.savefig(improved_image_filename, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Created error message placeholder image: {improved_image_filename}")
                except Exception as e2:
                    print(f"Failed to create fallback image: {str(e2)}")
                    # Last resort - create a basic image file using PIL
                    try:
                        from PIL import Image, ImageDraw, ImageFont
                        img = Image.new('RGB', (800, 600), color=(255, 255, 255))
                        d = ImageDraw.Draw(img)
                        d.text((400, 300), "Visualization Error", fill=(0, 0, 0))
                        img.save(improved_image_filename)
                        print(f"Created basic PIL image as absolute last resort: {improved_image_filename}")
                    except Exception as e3:
                        print(f"Failed to create PIL fallback image: {str(e3)}")
                        return question
            
            print(f"Fixed visualization saved to {improved_image_filename}")
            
            # Save the improved visualization code to the question
            question["visualization_code"] = improved_code
            
            # Upload the new image to Google Drive
            # Import the generator to use its upload method
            from src.generator import QuestionGenerator
            generator = QuestionGenerator(api_key=self.api_key, model_name=self.model_name)
            
            new_image_url = generator.upload_to_gdrive(improved_image_filename)
            
            if not new_image_url:
                print("Warning: Failed to upload fixed image to Google Drive. Using local file URL instead.")
                # Create a local file URL as fallback
                abs_path = os.path.abspath(improved_image_filename)
                new_image_url = f"file://{abs_path.replace(os.sep, '/')}"
                
            # Update the question's img tag with the new URL
            img_pattern = re.compile(r'<img[^>]*src=["\'](.*?)["\'][^>]*>')
            updated_passage = img_pattern.sub(
                lambda m: m.group(0).replace(f'src="{image_url}"', f'src="{new_image_url}"'),
                passage
            )
            
            # If the pattern didn't match, try a simpler approach
            if updated_passage == passage:
                print("First replacement pattern didn't match, trying simpler approach...")
                updated_passage = passage.replace(f'src="{image_url}"', f'src="{new_image_url}"')
                updated_passage = updated_passage.replace(f"src='{image_url}'", f"src='{new_image_url}'")
            
            # Update the question
            question["passage"] = updated_passage
            
            # Add information about the local image file path
            question["local_image_path"] = improved_image_filename
            
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
                
            adjusted_question = adjusted_question_json_str
            
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
    
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """
        Extract a JSON object from the response text with improved handling of 
        escape sequences and control characters.
        
        Args:
            response_text: The text response from Claude.
            
        Returns:
            The extracted JSON as a dictionary, or None if no JSON found or parsing failed.
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
                        # Found the full JSON object
                        json_text = response_text[start_idx:i+1]
                        
                        # Clean potential issues
                        try:
                            # First try direct parsing
                            return json.loads(json_text)
                        except json.JSONDecodeError as e:
                            print(f"Initial JSON parsing failed: {e}")
                            
                            try:
                                # Normalize newlines - JSON doesn't allow literal newlines in strings
                                normalized_text = json_text.replace('\n', '\\n')
                                # Clean control characters
                                import re
                                cleaned_text = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]', '', normalized_text)
                                return json.loads(cleaned_text)
                            except json.JSONDecodeError as e2:
                                print(f"Second JSON parsing attempt failed: {e2}")
                                
                                # If all else fails, return None
                                return None
            
            return None
        except Exception as e:
            print(f"Error extracting JSON: {str(e)}")
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