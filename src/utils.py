"""
Utility functions for the SAT Question Generator.
"""

import os
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

def load_json_file(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        The parsed JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: The data to save.
        file_path: Path to the JSON file.
        indent: Indentation level for the JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

def get_prompt_by_name(prompts: List[Dict], name: str) -> Optional[Dict]:
    """
    Get a prompt by its name.
    
    Args:
        prompts: List of prompt dictionaries.
        name: Name of the prompt to find.
        
    Returns:
        The prompt dictionary if found, None otherwise.
    """
    for prompt in prompts:
        if prompt.get("name") == name:
            return prompt
    return None

def get_random_examples_by_type(examples: List[Dict], question_type: str, count: int = 1, 
                               difficulty: Optional[str] = None) -> List[Dict]:
    """
    Get a random sample of examples of a specific type and optionally difficulty.
    
    Args:
        examples: List of example dictionaries.
        question_type: Type of question to filter by.
        count: Number of examples to return.
        difficulty: Optional difficulty to filter by.
        
    Returns:
        A list of randomly selected example dictionaries.
    """
    filtered_examples = [ex for ex in examples if ex.get("type") == question_type]
    
    if difficulty:
        filtered_examples = [ex for ex in filtered_examples if ex.get("difficulty") == difficulty]
    
    if not filtered_examples:
        return []
    
    # Return at most count examples
    return random.sample(filtered_examples, min(count, len(filtered_examples)))

def get_timestamp() -> str:
    """
    Get a formatted timestamp for the current time.
    
    Returns:
        A string representation of the current timestamp.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_output_filename(question_type: str, difficulty: str) -> str:
    """
    Generate a filename for saving generated questions.
    
    Args:
        question_type: Type of questions.
        difficulty: Difficulty of questions.
        
    Returns:
        A string with the generated filename.
    """
    timestamp = get_timestamp()
    return f"{question_type}_{difficulty}_{timestamp}.json"

def ensure_output_directory() -> str:
    """
    Ensure the output directory exists.
    
    Returns:
        The path to the output directory.
    """
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir

def map_difficulty_level(difficulty: str) -> str:
    """
    Map difficulty level from user input to the format used in examples.
    
    Args:
        difficulty: User-friendly difficulty (easy, medium, hard).
        
    Returns:
        The mapped difficulty level (1, 2, 3).
    """
    mapping = {
        "easy": "1",
        "medium": "2",
        "hard": "3"
    }
    return mapping.get(difficulty.lower(), "1")  # Default to easy if unknown

def normalize_question(question: Dict) -> Dict:
    """
    Normalize a question dictionary to ensure it has a consistent format.
    
    Args:
        question: Question dictionary to normalize.
        
    Returns:
        The normalized question dictionary.
    """
    # Ensure the question has all required fields
    required_fields = ["passage", "question", "correct_answer", "skill", "domain", "difficulty"]
    
    for field in required_fields:
        if field not in question:
            question[field] = ""
    
    # Ensure distractors are correctly named
    if "distractor1" not in question and "distractors_0" in question:
        question["distractor1"] = question.pop("distractors_0")
    if "distractor2" not in question and "distractors_1" in question:
        question["distractor2"] = question.pop("distractors_1")
    if "distractor3" not in question and "distractors_2" in question:
        question["distractor3"] = question.pop("distractors_2")
    
    return question

def setup_gdrive_credentials(credentials_data: Optional[str] = None, credentials_path: Optional[str] = None) -> bool:
    """
    Set up Google Drive credentials either from data or file path.
    
    Args:
        credentials_data: JSON string containing credentials data.
        credentials_path: Path to save credentials file.
        
    Returns:
        True if credentials were set up successfully, False otherwise.
    """
    import os
    import json
    
    # Default path if not provided
    if not credentials_path:
        credentials_path = os.getenv("GDRIVE_CREDENTIALS_PATH", "gdrive_credentials.json")
    
    try:
        # If credentials data is provided as a string, save it to the file
        if credentials_data:
            # Ensure the data is valid JSON
            try:
                json.loads(credentials_data)
            except json.JSONDecodeError:
                print("Error: Provided credentials data is not valid JSON")
                return False
                
            # Write credentials to file
            with open(credentials_path, 'w', encoding='utf-8') as f:
                f.write(credentials_data)
            
            print(f"Google Drive credentials saved to {credentials_path}")
            return True
        
        # If no credentials data is provided, check if the file already exists
        elif os.path.exists(credentials_path):
            # Validate the existing file
            try:
                with open(credentials_path, 'r', encoding='utf-8') as f:
                    json.load(f)
                print(f"Using existing Google Drive credentials at {credentials_path}")
                return True
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading existing credentials file: {str(e)}")
                return False
        else:
            print(f"No Google Drive credentials provided and no file found at {credentials_path}")
            print("Please provide credentials data or set up the credentials file")
            return False
            
    except Exception as e:
        print(f"Error setting up Google Drive credentials: {str(e)}")
        return False 