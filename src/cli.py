"""
Command-line interface for the SAT Question Generator.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import time
from tqdm import tqdm

from src.generator import QuestionGenerator, QUESTION_TYPE_DISPLAY, DUAL_VARIANT_TYPES
from src.validator import QuestionValidator
from src.fix import QuestionFixer
from src.utils import (
    generate_output_filename,
    ensure_output_directory,
    map_difficulty_level,
    save_json_file
)

def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """
    Get input from the user with a default value.
    
    Args:
        prompt: The prompt to display to the user.
        default: The default value to use if the user enters nothing.
        
    Returns:
        The user's input, or the default value if nothing was entered.
    """
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()

def choose_from_list(options: List[str], prompt: str, default: Optional[int] = None) -> str:
    """
    Let the user choose an option from a list.
    
    Args:
        options: The list of options to choose from.
        prompt: The prompt to display to the user.
        default: The default option index to use if the user enters nothing.
        
    Returns:
        The chosen option.
    """
    print(f"\n{prompt}:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    # Default display
    default_display = f" [{default}]" if default is not None else ""
    
    while True:
        choice = input(f"Enter your choice (1-{len(options)}){default_display}: ").strip()
        
        # Use default if input is empty and default is provided
        if not choice and default is not None:
            return options[default - 1]
        
        try:
            choice_index = int(choice)
            if 1 <= choice_index <= len(options):
                return options[choice_index - 1]
            else:
                print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Please enter a valid number.")

def run_cli() -> None:
    """
    Run the command-line interface.
    """
    print("\n" + "=" * 80)
    print("SAT Question Generator".center(80))
    print("=" * 80 + "\n")
    
    try:
        # Initialize the generator
        generator = QuestionGenerator()
        
        # Get available question types (prompts)
        question_types = generator.get_question_generation_prompts()
        
        if not question_types:
            print("No question generation prompts found. Please check your sat-prompts.json file.")
            sys.exit(1)
        
        # Let the user choose a question type
        chosen_question_type = choose_from_list(
            question_types,
            "Choose a question type to generate",
            default=1
        )
        
        # Get the base question type 
        base_type = None
        for qtype, display in QUESTION_TYPE_DISPLAY.items():
            if display == chosen_question_type:
                base_type = qtype
                break
        
        if not base_type:
            print(f"Error: Couldn't find the base type for {chosen_question_type}")
            sys.exit(1)
                
        # Let the user choose a difficulty level
        difficulty_options = ["easy", "medium", "hard"]
        chosen_difficulty = choose_from_list(
            difficulty_options,
            "Choose a difficulty level",
            default=1
        )
        
        # Map difficulty to the format used in examples
        mapped_difficulty = map_difficulty_level(chosen_difficulty)
        
        # Get the number of questions to generate
        default_count = os.getenv("DEFAULT_QUESTION_COUNT", "5")
        count_input = get_user_input("How many questions do you want to generate?", default_count)
        
        try:
            question_count = int(count_input)
            if question_count <= 0:
                raise ValueError("Count must be positive")
        except ValueError:
            print(f"Invalid count, using default value of {default_count}.")
            question_count = int(default_count)
        
        # Get question type information
        question_type_info = generator.get_question_type_info(base_type)
        
        print("\n" + "-" * 80)
        print(f"Generating {question_count} {chosen_difficulty} {chosen_question_type} questions...")
        
        # If dual variant type, display the distribution
        if base_type in DUAL_VARIANT_TYPES:
            distribution = generator.calculate_variant_distribution(question_count)
            print(f"Distribution: {distribution['info']} informational, {distribution['lit']} literary")
            print("Using random works and topics from works.json and topics.json")
        else:
            if question_type_info["needs_work"]:
                print("Using random works from works.json")
            elif question_type_info["needs_topic"]:
                print("Using random topics from topics.json")
                
        print("-" * 80 + "\n")
        
        # Generate the questions with no explicit work or topic (will use random ones)
        start_time = time.time()
        questions = generator.generate_multiple_questions(
            base_type, mapped_difficulty, question_count, None
        )
        generation_time = time.time() - start_time
        
        if not questions:
            print("No questions were generated. Please try again.")
            sys.exit(1)
        
        print(f"\nGenerated {len(questions)} questions in {generation_time:.2f} seconds.")
        print("\n" + "-" * 80)
        print("Validating questions...")
        print("-" * 80 + "\n")
        
        # Validate the questions
        validator = QuestionValidator()
        start_time = time.time()
        validation_results = validator.validate_questions(questions, base_type)
        validation_time = time.time() - start_time
        
        passed_count = sum(1 for result in validation_results if result["passed_all"])
        failed_count = len(validation_results) - passed_count
        
        print(f"\nValidation completed in {validation_time:.2f} seconds.")
        print(f"Passed: {passed_count}/{len(validation_results)} questions.")
        print(f"Failed: {failed_count}/{len(validation_results)} questions.")
        
        # Fix failed questions if any
        if failed_count > 0:
            print("\n" + "-" * 80)
            print("Fixing failed questions...")
            print("-" * 80 + "\n")
            
            fixer = QuestionFixer()
            start_time = time.time()
            fixed_results = fixer.fix_questions(validation_results)
            fix_time = time.time() - start_time
            
            print(f"\nFixes completed in {fix_time:.2f} seconds.")
            
            # Use fixed questions for final results
            validation_results = fixed_results
        else:
            fix_time = 0
        
        # Save the results
        output_dir = ensure_output_directory()
        output_filename = generate_output_filename(base_type, chosen_difficulty)
        output_path = os.path.join(output_dir, output_filename)
        
        # Filter out questions that failed fixing and shouldn't be included in the output
        included_questions = []
        for result in validation_results:
            if result.get("include_in_output", True):  # Default to True for backward compatibility
                included_questions.append(result["question"])
        
        # Count questions that were excluded due to failed fixes
        excluded_count = len(validation_results) - len(included_questions)
        if excluded_count > 0:
            print(f"\nExcluded {excluded_count} questions that failed validation after fixing.")
        
        # Get distribution of variants in the final questions
        variant_counts = {}
        for question in included_questions:
            variant = question.get("variant", "general")
            variant_counts[variant] = variant_counts.get(variant, 0) + 1
        
        # Prepare the final output
        final_output = {
            "metadata": {
                "question_type": base_type,
                "display_name": chosen_question_type,
                "difficulty": chosen_difficulty,
                "count": question_count,
                "variant_distribution": variant_counts,
                "generation_time": generation_time,
                "validation_time": validation_time,
                "fix_time": fix_time,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "excluded_count": excluded_count,
                "included_count": len(included_questions)
            },
            "questions": included_questions,
            "validation_details": validation_results
        }
        
        save_json_file(final_output, output_path)
        
        print(f"\nResults saved to {output_path}")
        print("\n" + "=" * 80)
        print("Done!".center(80))
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_cli() 