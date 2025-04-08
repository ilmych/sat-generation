"""
Question validation module for ensuring the quality of generated questions.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from anthropic import Anthropic
from tqdm import tqdm

from src.utils import get_prompt_by_name

class QuestionValidator:
    """
    Class for validating generated SAT questions using quality control prompts.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the QuestionValidator.
        
        Args:
            api_key: Anthropic API key. If None, it's loaded from environment variables.
            model_name: Name of the Claude model to use. If None, it's loaded from environment variables.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set it either directly or via ANTHROPIC_API_KEY environment variable.")
        
        self.model_name = model_name or os.getenv("MODEL_NAME", "claude-3-sonnet-20240229")
        self.client = Anthropic(api_key=self.api_key)
        
        # Load prompts
        self.prompts = []
        self.load_prompts()
        
        # Define standard quality control prompts that all questions should go through
        self.standard_validators = [
            "standout",
            "grammar",
            "relation",
            "similarity",
            "question-quality",
            "correct-distractor",
            "no-correct-answer",
            "multiple-correct",
            "plausibility",
            "length"  # Added length validator
        ]
        
        # Define specific validators for boundaries questions
        self.boundaries_validators = [
            "grammar",
            "correct-distractor",
            "no-correct-answer",
            "multiple-correct",
            "grammar-validator-boundaries",
            "no-punctuation-boundaries",
            "length"  # Added length validator
        ]
        
        # Define specific validators for transitions questions
        self.transitions_validators = [
            "grammar",
            "similarity",
            "correct-distractor",
            "no-correct-answer",
            "multiple-correct",
            "grammar-validator-transitions",
            "no-starting-blank-transitions",
            "length"  # Added length validator
        ]
    
    def load_prompts(self) -> None:
        """
        Load prompts from JSON file.
        """
        try:
            with open("sat-prompts.json", "r", encoding="utf-8") as f:
                self.prompts = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading prompts: {str(e)}")
    
    def validate_question(self, question: Dict, validator_name: Optional[str] = None) -> Dict:
        """
        Validate a single question using specified validators or the appropriate ones for its type.
        
        Args:
            question: The question dictionary to validate.
            validator_name: (Optional) Specific validator to use. If None, use all appropriate validators.
            
        Returns:
            A dictionary with validation scores and reasoning.
        """
        question_type = question.get("type", "")
        
        # Log the question being validated
        print(f"\nValidating {question_type} question: {question.get('question', '')[:50]}...")
        
        # Choose appropriate validators based on question type
        if validator_name:
            validators_to_use = [validator_name]
        else:
            if question_type in ["boundaries", "bound"]:
                validators_to_use = self.boundaries_validators
            elif question_type in ["transitions", "trans"]:
                validators_to_use = self.transitions_validators
            else:
                validators_to_use = self.standard_validators
        
        # Log the validators being used
        print(f"Using validators: {validators_to_use}")
        
        results = {}
        
        # Separate special validators (those that don't need prompts)
        special_validators = ["plausibility", "length"]
        regular_validators = [v for v in validators_to_use if v not in special_validators]
        
        # First, run all regular validators that need prompts
        for validator in regular_validators:
            # Find the validator prompt object with the matching name
            prompt_obj = get_prompt_by_name(self.prompts, validator)
            
            if not prompt_obj:
                print(f"Warning: Validator '{validator}' not found in prompts. Skipping.")
                continue
                
            try:
                result = self._run_validation_prompt(question, prompt_obj)
                
                # Ensure result is a dictionary with 'score' and 'reasoning' keys
                if isinstance(result, bool):
                    result = {
                        "score": 1 if result else 0,
                        "reasoning": f"Validation {'passed' if result else 'failed'}"
                    }
                elif not isinstance(result, dict):
                    result = {
                        "score": 0,
                        "reasoning": f"Unexpected validation result type: {type(result)}"
                    }
                
                # Ensure score is either 0 or 1
                if "score" in result and not isinstance(result["score"], (int, float)):
                    result["score"] = 1 if result["score"] else 0
                
                results[validator] = result
            except Exception as e:
                print(f"Error running validator '{validator}': {str(e)}")
                results[validator] = {
                    "score": 0,
                    "reasoning": f"Error during validation: {str(e)}"
                }
        
        # Special handling for plausibility validator
        if "plausibility" in validators_to_use:
            try:
                result = self._run_plausibility_validation(question)
                
                # Ensure result is a dictionary
                if isinstance(result, bool):
                    result = {
                        "score": 1 if result else 0,
                        "reasoning": f"Plausibility validation {'passed' if result else 'failed'}"
                    }
                elif not isinstance(result, dict):
                    result = {
                        "score": 0,
                        "reasoning": f"Unexpected plausibility validation result type: {type(result)}"
                    }
                
                results["plausibility"] = result
            except Exception as e:
                print(f"Error running plausibility validation: {str(e)}")
                results["plausibility"] = {
                    "score": 0,
                    "reasoning": f"Error during plausibility validation: {str(e)}"
                }
        
        # Special handling for length validator
        if "length" in validators_to_use:
            try:
                result = self.perform_length_check(question)
                
                # Ensure result is a dictionary
                if isinstance(result, bool):
                    result = {
                        "score": 1 if result else 0,
                        "reasoning": f"Length check {'passed' if result else 'failed'}"
                    }
                elif not isinstance(result, dict):
                    result = {
                        "score": 0,
                        "reasoning": f"Unexpected length validation result type: {type(result)}"
                    }
                
                results["length"] = result
            except Exception as e:
                print(f"Error running length validation: {str(e)}")
                results["length"] = {
                    "score": 0,
                    "reasoning": f"Error during length validation: {str(e)}"
                }
        
        # Special handling for COEQ images
        if question_type in ["coeq", "command-of-evidence-quantitative"]:
            # If the question contains visualization code, validate it
            if "visualization_code" in question:
                try:
                    image_result = self.validate_coeq_image(question)
                    
                    # Ensure result is a dictionary
                    if isinstance(image_result, bool):
                        image_result = {
                            "score": 1 if image_result else 0,
                            "reasoning": f"Image validation {'passed' if image_result else 'failed'}"
                        }
                    elif not isinstance(image_result, dict):
                        image_result = {
                            "score": 0,
                            "reasoning": f"Unexpected image validation result type: {type(image_result)}"
                        }
                    
                    results["coeq_image"] = image_result
                except Exception as e:
                    print(f"Error validating COEQ image: {str(e)}")
                    results["coeq_image"] = {
                        "score": 0,
                        "reasoning": f"Error during COEQ image validation: {str(e)}"
                    }
        
        # Check if all validators have a score of at least 1
        # Make sure to safely extract scores from results
        all_passed = all(result.get("score", 0) >= 1 if isinstance(result, dict) else result 
                       for result in results.values())
        
        # Calculate the overall score as the minimum of all scores
        if results:
            overall_score = min(result.get("score", 0) if isinstance(result, dict) else (1 if result else 0) 
                              for result in results.values())
        else:
            overall_score = 0
        
        # Compile the detailed results
        validation_result = {
            "passed": all_passed,
            "overall_score": overall_score,
            "details": results
        }
        
        return validation_result
    
    def validate_questions(self, questions: List[Dict], question_type: str) -> List[Dict]:
        """
        Validate multiple questions using all appropriate quality control prompts.
        Uses parallel processing to validate multiple questions simultaneously.
        
        Args:
            questions: The list of question dictionaries to validate.
            question_type: The type of questions being validated.
            
        Returns:
            A list of dictionaries with questions and their validation results.
        """
        import concurrent.futures
        from functools import partial
        
        results = []
        
        # Extract the base question type (without -lit/-info suffix)
        base_type = question_type
        if "-" in question_type:
            base_type = question_type.split("-")[0]
            
        print(f"\nStarting validation for {len(questions)} questions of type '{question_type}' (base type: '{base_type}')")
        
        # Initialize counters for summary statistics
        validation_stats = {
            "total": len(questions),
            "passed_all": 0,
            "failed": 0,
            "validator_failures": {}  # Count failures by validator
        }
        
        # Define a worker function to validate a single question
        def validate_worker(i, question):
            try:
                print(f"Validating question {i+1}...")
                validation_results = self.validate_question(question, None)
                
                # Determine if the question passed all validations
                # Safely check score attribute for each validation result
                details = validation_results.get("details", {})
                passed_all = validation_results.get("passed", False)
                
                if passed_all:
                    # Generate explanations for questions that pass validation
                    print(f"Question {i+1} passed validation - generating explanations...")
                    explanations = self.generate_explanation(question)
                    
                    # Add explanations to the question
                    for key, explanation in explanations.items():
                        if not key.startswith("error"):  # Don't add error flag to question
                            question[key] = explanation
                
                return {
                    "index": i,
                    "question": question,
                    "validation_results": validation_results,
                    "passed_all": passed_all
                }
            except Exception as e:
                print(f"Error validating question {i+1}: {str(e)}")
                return {
                    "index": i,
                    "question": question,
                    "validation_results": {"error": {"score": 0, "reasoning": str(e), "error": True}},
                    "passed_all": False
                }
        
        # Get the maximum number of workers from environment or default to CPU count
        max_workers = int(os.getenv("MAX_VALIDATION_WORKERS", os.cpu_count() or 4))
        print(f"Using up to {max_workers} parallel validation workers")
        
        # Use thread pool executor for API calls (better for I/O bound tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_index = {
                executor.submit(validate_worker, i, question): i 
                for i, question in enumerate(questions)
            }
            
            # Collect results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                             desc="Validating questions", total=len(questions)):
                result = future.result()
                results.append(result)
                
                # Update statistics
                passed_all = result["passed_all"]
                
                # Get validation details safely
                validation_results = result.get("validation_results", {})
                details = validation_results.get("details", {})
                
                # Count validator failures
                if isinstance(details, dict):
                    for validator_name, validator_result in details.items():
                        # Safely extract score
                        score = 0
                        if isinstance(validator_result, dict):
                            score = validator_result.get("score", 0)
                        elif isinstance(validator_result, bool):
                            score = 1 if validator_result else 0
                            
                        if score == 0:
                            if validator_name not in validation_stats["validator_failures"]:
                                validation_stats["validator_failures"][validator_name] = 0
                            validation_stats["validator_failures"][validator_name] += 1
                
                if passed_all:
                    validation_stats["passed_all"] += 1
                else:
                    validation_stats["failed"] += 1
        
        # Sort results by original index to maintain order
        results.sort(key=lambda x: x["index"])
        
        # Remove the temporary index field from results
        for result in results:
            result.pop("index", None)
        
        # Print summary statistics
        print("\nValidation Summary:")
        print(f"- Total questions: {validation_stats['total']}")
        print(f"- Passed all validators: {validation_stats['passed_all']} ({100 * validation_stats['passed_all'] / validation_stats['total']:.1f}%)")
        print(f"- Failed one or more validators: {validation_stats['failed']} ({100 * validation_stats['failed'] / validation_stats['total']:.1f}%)")
        
        if validation_stats["validator_failures"]:
            print("\nFailures by validator:")
            for validator, count in sorted(validation_stats["validator_failures"].items(), key=lambda x: x[1], reverse=True):
                failure_rate = 100 * count / validation_stats['total']
                print(f"- {validator}: {count} failures ({failure_rate:.1f}%)")
        
        return results
    
    def _run_validation_prompt(self, question: Dict, prompt_obj: Dict) -> Dict:
        """
        Run a validation prompt against a question with retry logic for API errors.
        
        Args:
            question: The question dictionary to validate.
            prompt_obj: The validation prompt object.
            
        Returns:
            A dictionary with the validation score and reasoning.
        """
        prompt_template = prompt_obj["prompt"]
        prompt_name = prompt_obj["name"]
        
        # Replace placeholders in the prompt with the question data
        validation_prompt = prompt_template.replace("{passage}", question.get("passage", ""))
        validation_prompt = validation_prompt.replace("{question}", question.get("question", ""))
        validation_prompt = validation_prompt.replace("{correct_answer}", question.get("correct_answer", ""))
        
        # Handle distractors
        distractor1 = question.get("distractor1", question.get("distractors_0", ""))
        distractor2 = question.get("distractor2", question.get("distractors_1", ""))
        distractor3 = question.get("distractor3", question.get("distractors_2", ""))
        
        validation_prompt = validation_prompt.replace("{distractor1}", distractor1)
        validation_prompt = validation_prompt.replace("{distractor2}", distractor2)
        validation_prompt = validation_prompt.replace("{distractor3}", distractor3)
        
        # Some prompts use "options" instead of individual distractors
        options = "\n".join([question.get("correct_answer", ""), distractor1, distractor2, distractor3])
        validation_prompt = validation_prompt.replace("{options}", options)
        
        # Set up retry parameters
        max_retries = int(os.getenv("MAX_API_RETRIES", "3"))
        retry_delay = float(os.getenv("API_RETRY_DELAY_SECONDS", "2.0"))
        
        # Call the API to validate the question with retry logic
        for retry_attempt in range(max_retries):
            try:
                print(f"Validation API call attempt {retry_attempt + 1}/{max_retries} for {prompt_name}...")
                
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    temperature=0.0,  # Use zero temperature for consistent validation
                    system="You are a professional SAT question validator who ensures the quality of SAT questions.",
                    messages=[
                        {
                            "role": "user",
                            "content": validation_prompt
                        }
                    ]
                )
                
                # Extract the JSON result from the response
                validation_json_str = self._extract_json_from_response(response.content[0].text)
                
                # Retry if JSON extraction failed
                if not validation_json_str:
                    if retry_attempt < max_retries - 1:
                        print(f"No valid JSON found in validation response. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return {
                            "score": 0,
                            "reasoning": "Failed to extract validation result from response after all retry attempts."
                        }
                
                try:
                    validation_result = json.loads(validation_json_str)
                    
                    # Ensure the validation result has the required fields
                    if "score" not in validation_result:
                        validation_result["score"] = 0
                    if "reasoning" not in validation_result:
                        validation_result["reasoning"] = "No reasoning provided by validator."
                    
                    return validation_result
                    
                except json.JSONDecodeError as e:
                    if retry_attempt < max_retries - 1:
                        print(f"JSON decode error: {str(e)}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return {
                            "score": 0,
                            "reasoning": f"Failed to decode validation result JSON: {str(e)}"
                        }
                
            except Exception as e:
                if retry_attempt < max_retries - 1:
                    # Determine if it's a rate limit error
                    is_rate_limit = "rate limit" in str(e).lower() or "rate_limit" in str(e).lower() or "too many requests" in str(e).lower()
                    
                    # Add exponential backoff for rate limit errors
                    if is_rate_limit:
                        backoff_time = retry_delay * (2 ** retry_attempt)
                        print(f"Rate limit hit during validation. Backing off for {backoff_time} seconds before retry {retry_attempt + 2}...")
                        time.sleep(backoff_time)
                    else:
                        print(f"API call error: {str(e)}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    continue
                else:
                    return {
                        "score": 0,
                        "reasoning": f"Failed to validate question after {max_retries} attempts: {str(e)}"
                    }
        
        # This should never be reached due to the loop structure, but just in case
        return {
            "score": 0,
            "reasoning": "Failed to validate question due to unexpected error."
        }
    
    def _run_plausibility_validation(self, question: Dict) -> Dict:
        """
        Special validation for plausibility - checks if each distractor is plausible.
        For easy questions, at least 1/3 distractors must pass.
        For medium and hard questions, at least 2/3 distractors must pass.
        
        Args:
            question: The question dictionary to validate.
            
        Returns:
            A dictionary with the validation score and reasoning.
        """
        # Find the plausibility prompt
        plausibility_prompts = [p for p in self.prompts if p.get("function") == "quality control" and p.get("name") == "plausibility"]
        if not plausibility_prompts:
            return {
                "score": 0,
                "reasoning": "Plausibility prompt not found in sat-prompts.json",
                "error": True
            }
            
        prompt_obj = plausibility_prompts[0]
        prompt_template = prompt_obj["prompt"]
        
        # Extract the necessary information
        passage = question.get("passage", "")
        question_text = question.get("question", "")
        correct_answer = question.get("correct_answer", "")
        difficulty = question.get("difficulty", "")
        
        # Get all distractors
        distractors = [
            question.get("distractor1", question.get("distractors_0", "")),
            question.get("distractor2", question.get("distractors_1", "")),
            question.get("distractor3", question.get("distractors_2", ""))
        ]
        
        # Track results for each distractor
        distractor_results = []
        detailed_reasoning = []
        
        print(f"Running plausibility check for each distractor ({len(distractors)} total)")
        
        # Run the validation for each distractor
        for i, distractor in enumerate(distractors):
            if not distractor:  # Skip empty distractors
                print(f"  - Skipping empty distractor {i+1}")
                continue
                
            print(f"  - Checking plausibility for distractor {i+1}: {distractor[:30]}...")
            
            # Replace placeholders in the prompt with the question data and current distractor
            validation_prompt = prompt_template.replace("{passage}", passage)
            validation_prompt = validation_prompt.replace("{question}", question_text)
            validation_prompt = validation_prompt.replace("{correct_answer}", correct_answer)
            validation_prompt = validation_prompt.replace("{distractor}", distractor)
            
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    temperature=0.0,
                    system="You are a professional SAT question validator specializing in assessing the plausibility of distractors.",
                    messages=[
                        {
                            "role": "user",
                            "content": validation_prompt
                        }
                    ]
                )
                
                # Extract the result
                validation_json_str = self._extract_json_from_response(response.content[0].text)
                if not validation_json_str:
                    distractor_results.append(0)  # Count as fail if we can't extract result
                    detailed_reasoning.append(f"Distractor {i+1}: Failed to extract result")
                    print(f"    - Failed to extract result")
                    continue
                    
                validation_result = json.loads(validation_json_str)
                score = validation_result.get("score", 0)
                reasoning = validation_result.get("reasoning", "No reasoning provided")
                
                distractor_results.append(score)
                detailed_reasoning.append(f"Distractor {i+1}: {'PASS' if score == 1 else 'FAIL'} - {reasoning}")
                print(f"    - Result: {'PASS' if score == 1 else 'FAIL'}")
                
            except Exception as e:
                distractor_results.append(0)  # Count as fail on error
                detailed_reasoning.append(f"Distractor {i+1}: Error - {str(e)}")
                print(f"    - Error: {str(e)}")
        
        # Determine overall score based on difficulty
        passed_count = sum(distractor_results)
        total_count = len(distractor_results)
        
        # Set passing threshold based on difficulty
        passing_threshold = 1
        if difficulty in ["2", "3", "medium", "hard", "Medium", "Hard"]:  # Medium or hard
            passing_threshold = 2
            
        # Overall pass/fail
        passed = passed_count >= passing_threshold
        score = 1 if passed else 0
        
        print(f"  - Plausibility validation: {passed_count}/{total_count} distractors passed")
        print(f"  - Required threshold for difficulty '{difficulty}': {passing_threshold}")
        print(f"  - Overall result: {'PASS' if passed else 'FAIL'}")
        
        # Compile the results
        return {
            "score": score,
            "reasoning": f"Plausibility: {passed_count}/{total_count} distractors passed. Needed {passing_threshold} for difficulty level '{difficulty}'.\n\n" + "\n\n".join(detailed_reasoning),
            "distractor_results": distractor_results
        }
    
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
            # First, look for <answer> tags (some prompts use these)
            start_tag = "<answer>"
            end_tag = "</answer>"
            if start_tag in response_text and end_tag in response_text:
                start_idx = response_text.find(start_tag) + len(start_tag)
                end_idx = response_text.find(end_tag)
                if start_idx < end_idx:
                    return response_text[start_idx:end_idx].strip()
            
            # Otherwise, look for JSON object start and end
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

    def validate_coeq_image(self, question: Dict) -> Dict:
        """
        Validate the readability and clarity of a COEQ image.
        
        Args:
            question: The question dictionary.
            
        Returns:
            Dictionary with validation results.
        """
        # Extract image URL from the question
        passage = question.get("passage", "")
        import re
        
        # Extract image URL from img tag
        img_pattern = re.compile(r'<img[^>]*src=["\'](.*?)["\'][^>]*>')
        img_match = img_pattern.search(passage)
        
        if not img_match:
            return {
                "score": 0,
                "reasoning": "No image URL found in the question passage.",
                "error": True
            }
            
        image_url = img_match.group(1)
        print(f"Validating image: {image_url}")
        
        # Download the image to get its binary content
        try:
            import requests
            response = requests.get(image_url)
            
            if response.status_code != 200:
                return {
                    "score": 0,
                    "reasoning": f"Failed to download image: HTTP status {response.status_code}",
                    "error": True
                }
                
            image_data = response.content
                
            # Define the validation prompt for Claude Vision
            vision_prompt = """
            You are evaluating a data visualization for an SAT quantitative evidence question. 
            
            Focus ONLY on the readability and clarity of the visualization:
            1. Can all text be clearly read? (labels, titles, legends, axis values)
            2. Is there any overlap of text or visual elements?
            3. Are the data points, bars, lines, or other visualization elements clearly visible?
            4. Is the contrast sufficient for all elements to be distinguished?
            5. Would students be able to extract the information needed from this visualization?
            
            Provide your assessment in the following JSON format:
            
            ```json
            {
                "score": 0 or 1 (0 = fails readability check, 1 = passes readability check),
                "reasoning": "Detailed explanation of your assessment, listing all readability issues found or confirming clear readability",
                "improvement_suggestions": [
                    "Specific technical suggestion 1",
                    "Specific technical suggestion 2",
                    ...
                ]
            }
            ```
            
            The improvement_suggestions should be specific, actionable changes to the matplotlib code that would fix the readability issues.
            Only include improvement_suggestions if the score is 0 (fails).
            If the score is 1 (passes), leave improvement_suggestions as an empty list.
            """
            
            # Call the Anthropic API with vision capability
            import base64
            
            # Convert image data to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Create the message content with the image
            message_content = [
                {
                    "type": "text",
                    "text": vision_prompt
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_base64
                    }
                }
            ]
            
            # Call the API with image for validation
            response = self.client.messages.create(
                model="claude-3-opus-20240229",  # Use a model with vision capabilities
                max_tokens=1500,
                temperature=0.0,
                system="You are a professional data visualization expert evaluating readability of visualizations for educational testing.",
                messages=[
                    {
                        "role": "user",
                        "content": message_content
                    }
                ]
            )
            
            # Extract the JSON result from the response
            validation_json_str = self._extract_json_from_response(response.content[0].text)
            if not validation_json_str:
                return {
                    "score": 0,
                    "reasoning": "Failed to extract image validation result from Claude Vision.",
                    "error": True
                }
                
            validation_result = json.loads(validation_json_str)
            
            # Ensure we have the required fields
            if "score" not in validation_result:
                validation_result["score"] = 0
            if "reasoning" not in validation_result:
                validation_result["reasoning"] = "No reasoning provided for image validation."
            if "improvement_suggestions" not in validation_result:
                validation_result["improvement_suggestions"] = []
                
            # Log the result
            print(f"  - Image validation result: {'PASS' if validation_result['score'] == 1 else 'FAIL'}")
            if validation_result["score"] == 0 and validation_result["improvement_suggestions"]:
                print(f"  - Improvement suggestions: {len(validation_result['improvement_suggestions'])} suggestion(s) provided")
                
            return validation_result
            
        except Exception as e:
            return {
                "score": 0,
                "reasoning": f"Error during image validation: {str(e)}",
                "error": True
            }
            
    def fix_coeq_image(self, question: Dict, original_code: str, improvement_suggestions: List[str]) -> Tuple[str, Optional[str]]:
        """
        Generate improved visualization code based on improvement suggestions.
        
        Args:
            question: The question dictionary.
            original_code: The original Python code that generated the image.
            improvement_suggestions: List of improvement suggestions from validate_coeq_image.
            
        Returns:
            A tuple of (improved_code, error_message). If successful, error_message will be None.
        """
        try:
            # Format the improvement suggestions
            suggestions_text = "\n".join([f"- {suggestion}" for suggestion in improvement_suggestions])
            
            # Create a prompt for Claude to improve the code
            improvement_prompt = f"""
            # Fix Data Visualization for SAT Question
            
            I need to improve a data visualization for an SAT question. The current visualization has readability issues.
            
            ## Original Question
            ```json
            {json.dumps(question, indent=2)}
            ```
            
            ## Original Python Visualization Code
            ```python
            {original_code}
            ```
            
            ## Readability Issues and Improvement Suggestions
            {suggestions_text}
            
            ## Instructions
            1. Fix the visualization code to address ALL of the readability issues.
            2. Maintain the same core data and visualization type (e.g., bar chart, scatter plot, etc.)
            3. Improve clarity, readability, and accessibility
            4. Make sure font sizes, colors, and contrast are appropriate for an SAT test
            5. Keep the same filename in the savefig command: 'visualization.png'
            6. Make sure all labels are clear and properly positioned 
            7. Ensure the visualization accurately represents the data in the question
            8. Return ONLY the complete, improved Python code with no additional explanations
            
            Your improved code should be a complete Python script that can be run directly to generate the improved visualization.
            """
            
            # Call the API to get improved code
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                temperature=0.2,
                system="You are an expert data visualization specialist with deep knowledge of matplotlib and Python. You create clear, readable, and accessible visualizations.",
                messages=[
                    {
                        "role": "user",
                        "content": improvement_prompt
                    }
                ]
            )
            
            # Extract the improved code
            improved_code = self._extract_code_from_response(response.content[0].text)
            
            if not improved_code:
                return original_code, "Failed to extract improved code from the API response."
                
            return improved_code, None
            
        except Exception as e:
            return original_code, str(e)
            
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
        
    def generate_explanation(self, question: Dict) -> Dict[str, str]:
        """
        Generate explanations for why each answer option is correct or incorrect.
        Includes retry logic for handling API errors and JSON extraction failures.
        
        Args:
            question: The question dictionary.
            
        Returns:
            A dictionary with explanations for the correct answer and each distractor.
        """
        # Set up retry parameters
        max_retries = int(os.getenv("MAX_API_RETRIES", "3"))
        retry_delay = float(os.getenv("API_RETRY_DELAY_SECONDS", "2.0"))
        
        for retry_attempt in range(max_retries):
            try:
                print(f"Explanation API call attempt {retry_attempt + 1}/{max_retries}...")
                
                # Create the prompt for explanation generation
                explanation_prompt = f"""You are an expert SAT tutor tasked with explaining answer choices for SAT-style questions. Given a passage, question, and answer options, provide a concise, specific explanation for why each option is correct or incorrect. Your explanations should:

1. Be concise (2-4 sentences) yet informative.
2. Directly address why the option is correct or incorrect.
3. Reference specific parts of the passage or question when relevant.
4. Provide context by restating relevant information.
5. Use comparative analysis for incorrect options.
6. Outline logical reasoning steps.
7. Maintain a consistent structure across different question types.
8. Stick to information provided in the passage and question.
9. Highlight the specific skill or convention being tested when applicable.
10. Use phrases like "This option is incorrect because..." for wrong answers.
11. Tailor explanations slightly to different question types (e.g., evidence-based, grammar, rhetorical).
12. Remain objective and avoid personal opinions.
13. Use clear, straightforward language.

For each option, start your explanation with either "This option is the best answer." or "This option is incorrect."

Format your response as a JSON object matching this structure:
{{
  "explanation_correct": [your explanation here],
  "explanation_distractor1": [your explanation here],
  "explanation_distractor2": [your explanation here],
  "explanation_distractor3": [your explanation here]
}}

Here is the question to explain:
{json.dumps(question, indent=2)}"""
                
                # Call the API to generate the explanation
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",  # Use Claude 3.5 Sonnet for explanations
                    max_tokens=2000,
                    temperature=0.0,  # Use temperature 0 for consistent explanations
                    system="You are an expert SAT tutor who explains answer choices in SAT questions with clarity and precision.",
                    messages=[
                        {
                            "role": "user",
                            "content": explanation_prompt
                        }
                    ]
                )
                
                # Extract the JSON result from the response
                explanation_json_str = self._extract_json_from_response(response.content[0].text)
                
                # Retry if JSON extraction failed
                if not explanation_json_str:
                    if retry_attempt < max_retries - 1:
                        print(f"No valid JSON found in explanation response. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return {
                            "explanation_correct": "Failed to extract explanation from response after all retry attempts.",
                            "explanation_distractor1": "Failed to extract explanation from response after all retry attempts.",
                            "explanation_distractor2": "Failed to extract explanation from response after all retry attempts.",
                            "explanation_distractor3": "Failed to extract explanation from response after all retry attempts.",
                            "error": True
                        }
                
                try:
                    explanations = json.loads(explanation_json_str)
                except json.JSONDecodeError as e:
                    if retry_attempt < max_retries - 1:
                        print(f"JSON decode error in explanation: {str(e)}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return {
                            "explanation_correct": f"Failed to decode JSON after all retry attempts: {str(e)}",
                            "explanation_distractor1": f"Failed to decode JSON after all retry attempts: {str(e)}",
                            "explanation_distractor2": f"Failed to decode JSON after all retry attempts: {str(e)}",
                            "explanation_distractor3": f"Failed to decode JSON after all retry attempts: {str(e)}",
                            "error": True
                        }
                
                # Ensure we have all required fields
                required_fields = ["explanation_correct", "explanation_distractor1", "explanation_distractor2", "explanation_distractor3"]
                for field in required_fields:
                    if field not in explanations:
                        explanations[field] = f"No explanation provided for {field}."
                
                return explanations
                
            except Exception as e:
                if retry_attempt < max_retries - 1:
                    # Determine if it's a rate limit error
                    is_rate_limit = "rate limit" in str(e).lower() or "rate_limit" in str(e).lower() or "too many requests" in str(e).lower()
                    
                    # Add exponential backoff for rate limit errors
                    if is_rate_limit:
                        backoff_time = retry_delay * (2 ** retry_attempt)
                        print(f"Rate limit hit during explanation. Backing off for {backoff_time} seconds before retry {retry_attempt + 2}...")
                        time.sleep(backoff_time)
                    else:
                        print(f"API call error during explanation: {str(e)}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    continue
                else:
                    return {
                        "explanation_correct": f"Error generating explanation after {max_retries} attempts: {str(e)}",
                        "explanation_distractor1": f"Error generating explanation after {max_retries} attempts: {str(e)}",
                        "explanation_distractor2": f"Error generating explanation after {max_retries} attempts: {str(e)}",
                        "explanation_distractor3": f"Error generating explanation after {max_retries} attempts: {str(e)}",
                        "error": True
                    }

    def perform_length_check(self, question: Dict) -> Dict:
        """
        Check if the correct answer length is proportional to the distractors.
        This helps prevent students from selecting the correct answer just based on its length.
        
        Args:
            question: The question dictionary to validate.
            
        Returns:
            A dictionary with the validation score and reasoning.
        """
        # Extract the correct answer and distractors
        correct_answer = question.get("correct_answer", "")
        
        # Handle different formats of distractors
        distractors = []
        if "distractor1" in question:
            distractors = [
                question.get("distractor1", ""),
                question.get("distractor2", ""),
                question.get("distractor3", "")
            ]
        elif "distractors_0" in question:
            distractors = [
                question.get("distractors_0", ""),
                question.get("distractors_1", ""),
                question.get("distractors_2", "")
            ]
        elif "distractors" in question and isinstance(question["distractors"], list):
            distractors = question["distractors"][:3]  # Take up to 3 distractors
            
        # Remove any empty distractors
        distractors = [d for d in distractors if d]
        
        if not distractors:
            return {
                "score": 0,
                "reasoning": "No distractors found to compare length with correct answer",
                "error": True
            }
            
        # Calculate lengths
        # Strip HTML tags if present
        def strip_html(text):
            import re
            return re.sub(r'<[^>]*>', '', text)
            
        correct_text = strip_html(correct_answer)
        distractor_texts = [strip_html(d) for d in distractors]
        
        correct_length = len(correct_text)
        distractor_lengths = [len(d) for d in distractor_texts]
        longest_distractor = max(distractor_lengths)
        shortest_distractor = min(distractor_lengths)

        # Check if all answers are very short (<=3 words)
        if len(correct_text.split()) <= 3 and all(len(d.split()) <= 3 for d in distractor_texts):
            return {
                "score": 1,
                "reasoning": "All answers are very short (<=3 words), length differences are acceptable"
            }
        
        # Check if correct answer is too long compared to longest distractor
        if correct_length > 1.1 * longest_distractor:
            return {
                "score": 0,
                "reasoning": f"The correct answer ({correct_length} chars) is more than 10% longer than the longest distractor ({longest_distractor} chars)"
            }
            
        # Check if correct answer is too short compared to shortest distractor
        if correct_length < shortest_distractor / 1.5:
            return {
                "score": 0,
                "reasoning": f"The correct answer ({correct_length} chars) is less than 2/3 the length of the shortest distractor ({shortest_distractor} chars)"
            }
            
        # All checks passed
        return {
            "score": 1,
            "reasoning": f"The correct answer length ({correct_length} chars) is within acceptable range compared to distractors (shortest: {shortest_distractor}, longest: {longest_distractor})"
        } 