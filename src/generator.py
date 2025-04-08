"""
Question generation module using Anthropic's Claude API.
"""

import os
import json
import random
import math
import time
import concurrent.futures
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from anthropic import Anthropic
from tqdm import tqdm

from src.utils import (
    get_prompt_by_name,
    get_random_examples_by_type,
    normalize_question,
    load_json_file
)

# Mapping of question types to prompt names
QUESTION_TYPE_TO_PROMPT = {
    "bound": "bound",               # Boundaries
    "cid-lit": "cid-lit",           # Central Ideas and Details (literary)
    "cid-info": "cid-info",         # Central Ideas and Details (informational)
    "coeq": "coeq",                 # Command of Evidence: Quantitative
    "coet-lit": "coet-lit",         # Command of Evidence: Textual (literary)
    "coet-info": "coet-info",       # Command of Evidence: Textual (informational)
    "ctc": "ctc",                   # Cross-Text Connections
    "fss": "fss",                   # Form, Structure, and Sense
    "inf": "inf",                   # Inferences
    "rhet": "rhet",                 # Rhetorical Synthesis
    "tsp-lit": "tsp-lit",           # Text Structure and Purpose (literary)
    "tsp-info": "tsp-info",         # Text Structure and Purpose (informational)
    "trans": "trans",               # Transitions
    "wic-lit": "wic-lit",           # Words in Context (literary)
    "wic-info": "wic-info",         # Words in Context (informational)
}

# Mapping of question types to skill names
QUESTION_TYPE_TO_SKILL = {
    "bound": "Boundaries",
    "cid-lit": "Central Ideas and Details",
    "cid-info": "Central Ideas and Details",
    "coeq": "Command of Evidence: Quantitative",
    "coet-lit": "Command of Evidence: Textual",
    "coet-info": "Command of Evidence: Textual",
    "ctc": "Cross-Text Connections",
    "fss": "Form, Structure, and Sense",
    "inf": "Inferences",
    "rhet": "Rhetorical Synthesis",
    "tsp-lit": "Text Structure and Purpose",
    "tsp-info": "Text Structure and Purpose",
    "trans": "Transitions",
    "wic-lit": "Words in Context",
    "wic-info": "Words in Context",
}

# Simplified mapping for CLI display
QUESTION_TYPE_DISPLAY = {
    "bound": "Boundaries",
    "cid": "Central Ideas and Details",
    "coeq": "Command of Evidence: Quantitative",
    "coet": "Command of Evidence: Textual",
    "ctc": "Cross-Text Connections",
    "fss": "Form, Structure, and Sense",
    "inf": "Inferences",
    "rhet": "Rhetorical Synthesis",
    "tsp": "Text Structure and Purpose",
    "trans": "Transitions",
    "wic": "Words in Context",
}

# Question types that support both literary and informational variants
DUAL_VARIANT_TYPES = ["cid", "coet", "tsp", "wic"]

# Question types that require literary works
LITERARY_QUESTION_TYPES = ["cid-lit", "coet-lit", "tsp-lit", "wic-lit"]

# Question types that require topics
# Note: All question types that aren't LITERARY_QUESTION_TYPES require topics
TOPIC_QUESTION_TYPES = ["cid-info", "coet-info", "coeq", "ctc", "inf", "rhet", "tsp-info", "wic-info", "bound", "fss", "trans"]

# Distribution ratio for literary vs informational questions
LIT_INFO_RATIO = {
    "lit": 0.3,  # 30% literary
    "info": 0.7  # 70% informational
}

class QuestionGenerator:
    """
    Class for generating SAT questions using Anthropic's Claude API.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the QuestionGenerator.
        
        Args:
            api_key: Anthropic API key. If None, it's loaded from environment variables.
            model_name: Name of the Claude model to use. If None, it's loaded from environment variables.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set it either directly or via ANTHROPIC_API_KEY environment variable.")
        
        self.model_name = model_name or os.getenv("MODEL_NAME", "claude-3-sonnet-20240229")
        self.client = Anthropic(api_key=self.api_key)
        
        # Load prompts and examples
        self.prompts = []
        self.examples = []
        self.load_prompts_and_examples()
        
        # Load literary works and topics
        self.works = self.load_works()
        self.topics = self.load_topics()
        
        # Track used examples with timestamps to enable better cycling strategy
        # Format: {example_key: last_used_timestamp}
        self.used_examples = {}
        
        # Track used works and topics to avoid repetition
        # Format: {work_title: last_used_timestamp}
        self.used_works = {}
        # Format: {topic: last_used_timestamp}
        self.used_topics = {}
    
    def load_prompts_and_examples(self) -> None:
        """
        Load prompts and examples from JSON files.
        """
        try:
            with open("sat-prompts.json", "r", encoding="utf-8") as f:
                self.prompts = json.load(f)
            
            with open("sat-examples.json", "r", encoding="utf-8") as f:
                self.examples = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading prompts or examples: {str(e)}")
    
    def load_works(self) -> List[Dict[str, Any]]:
        """
        Load literary works from works.json file.
        
        Returns:
            A list of literary work dictionaries.
        """
        try:
            return load_json_file("works.json")
        except Exception as e:
            print(f"Warning: Could not load works.json: {str(e)}. Using default works.")
            return [
                {"title": "Pride and Prejudice", "author": "Jane Austen", "year": 1813, "type": "novel"},
                {"title": "To Kill a Mockingbird", "author": "Harper Lee", "year": 1960, "type": "novel"},
                {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "year": 1925, "type": "novel"},
                {"title": "Hamlet", "author": "William Shakespeare", "year": 1603, "type": "play"},
                {"title": "Jane Eyre", "author": "Charlotte Brontë", "year": 1847, "type": "novel"},
                {"title": "Frankenstein", "author": "Mary Shelley", "year": 1818, "type": "novel"},
                {"title": "The Odyssey", "author": "Homer", "year": -800, "type": "epic poem"},
                {"title": "Their Eyes Were Watching God", "author": "Zora Neale Hurston", "year": 1937, "type": "novel"}
            ]
    
    def load_topics(self) -> List[str]:
        """
        Load topics from topics.json file.
        
        Returns:
            A list of topic strings.
        """
        try:
            data = load_json_file("topics.json")
            return data.get("topics", [])
        except Exception as e:
            print(f"Warning: Could not load topics.json: {str(e)}. Using default topics.")
            return [
                "climate change", 
                "artificial intelligence", 
                "democracy", 
                "renewable energy",
                "social media", 
                "public health",
                "space exploration",
                "economic inequality",
                "sustainable agriculture",
                "digital privacy",
                "urbanization",
                "genetic engineering"
            ]
    
    def get_random_work(self) -> str:
        """
        Get a random literary work formatted as a string.
        
        Returns:
            A string describing the literary work.
        """
        work = random.choice(self.works)
        return f"{work['title']} by {work['author']} ({work['year']}, {work['type']})"
    
    def get_random_topic(self) -> str:
        """
        Get a random topic.
        
        Returns:
            A string representing a topic.
        """
        return random.choice(self.topics)
    
    def get_matching_examples(self, skill: str, difficulty: str, question_type: str) -> List[Dict]:
        """
        Get examples that match the skill, difficulty, and type.
        
        Args:
            skill: The skill to match (e.g., "Words in Context")
            difficulty: The difficulty level to match ("1", "2", "3")
            question_type: The question type to match ("lit" or "info")
            
        Returns:
            A list of matching example dictionaries.
        """
        matching_examples = []
        
        for example in self.examples:
            if (example.get("skill") == skill and 
                example.get("difficulty") == difficulty and 
                example.get("type") == question_type):
                matching_examples.append(example)
        
        return matching_examples
    
    def get_random_example(self, skill: str, difficulty: str, question_type: str) -> Optional[Dict]:
        """
        Get a random example that matches the skill, difficulty, and type.
        Tries to avoid repeating examples if possible, using a weighted selection
        approach to favor less recently used examples.
        
        Args:
            skill: The skill to match (e.g., "Words in Context")
            difficulty: The difficulty level to match ("1", "2", "3")
            question_type: The question type to match ("lit" or "info")
            
        Returns:
            A random matching example dictionary, or None if no matching examples found.
        """
        matching_examples = self.get_matching_examples(skill, difficulty, question_type)
        
        if not matching_examples:
            print(f"Warning: No examples found for skill '{skill}', difficulty '{difficulty}', type '{question_type}'.")
            # If no examples match the exact difficulty, try to find examples with any difficulty
            matching_examples = [e for e in self.examples 
                              if e.get("skill") == skill and 
                                 e.get("type") == question_type]
            
            if not matching_examples:
                # If still no examples, try matching just on skill
                print(f"Warning: Falling back to any example with skill '{skill}'.")
                matching_examples = [e for e in self.examples if e.get("skill") == skill]
                
                if not matching_examples:
                    return None
        
        # Create a unique identifier for each example using content hash
        example_keys = []
        for e in matching_examples:
            # Create a stable hash identifier for the example based on its content
            content_str = f"{e.get('passage', '')[:100]}_{e.get('question', '')}_{e.get('correct_answer', '')}"
            # Create MD5 hash for a stable, unique identifier
            hash_obj = hashlib.md5(content_str.encode('utf-8'))
            example_key = hash_obj.hexdigest()
            example_keys.append(example_key)
        
        # Check which examples we haven't used yet
        unused_examples_indices = [i for i, key in enumerate(example_keys) if key not in self.used_examples]
        
        if unused_examples_indices:
            # If we have unused examples, choose randomly from them
            selected_index = random.choice(unused_examples_indices)
        else:
            # All examples have been used before, select based on least recently used
            # Sort the indices by when they were last used (oldest first)
            used_indices = [(i, self.used_examples.get(key, 0)) for i, key in enumerate(example_keys)]
            used_indices.sort(key=lambda x: x[1])  # Sort by timestamp
            # Select the least recently used example
            selected_index = used_indices[0][0]
            print(f"All {len(matching_examples)} examples for this criteria have been used. Using least recently used example.")
            
        # Get the selected example and its key
        selected_example = matching_examples[selected_index]
        selected_key = example_keys[selected_index]
        
        # Mark as used with current timestamp
        self.used_examples[selected_key] = time.time()
        
        return selected_example
    
    def get_question_generation_prompts(self) -> List[str]:
        """
        Get a list of available question generation prompt names for CLI display.
        
        Returns:
            A list of question type display names.
        """
        available_prompts = [p["name"] for p in self.prompts if p.get("function") == "generate question"]
        
        # Get unique base question types (without -lit/-info suffix)
        unique_types = set()
        for prompt_name in available_prompts:
            # Extract base type (e.g., "cid" from "cid-lit")
            base_type = prompt_name.split("-")[0]
            unique_types.add(base_type)
        
        # Return the display names for the available question types
        return [QUESTION_TYPE_DISPLAY[qtype] for qtype in unique_types if qtype in QUESTION_TYPE_DISPLAY]
    
    def get_prompt_name_from_display(self, display_name: str, variant: Optional[str] = None) -> str:
        """
        Convert a display name back to a prompt name.
        
        Args:
            display_name: The display name of the question type.
            variant: Optional variant ("lit" or "info").
            
        Returns:
            The corresponding prompt name.
        """
        # Find the base type key by display name
        base_type = None
        for qtype, display in QUESTION_TYPE_DISPLAY.items():
            if display == display_name:
                base_type = qtype
                break
        
        if not base_type:
            raise ValueError(f"Unknown question type: {display_name}")
        
        # If the type has literary/informational variants, use the specified variant
        if base_type in DUAL_VARIANT_TYPES:
            if variant and variant in ["lit", "info"]:
                return f"{base_type}-{variant}"
            else:
                # Default to informational if not specified
                return f"{base_type}-info"
        else:
            # Otherwise, return the base type
            return base_type
    
    def calculate_variant_distribution(self, count: int) -> Dict[str, int]:
        """
        Calculate the distribution of literary and informational questions.
        
        Args:
            count: Total number of questions to generate.
            
        Returns:
            A dictionary with the count of each variant.
        """
        lit_count = math.floor(count * LIT_INFO_RATIO["lit"])
        info_count = count - lit_count
        
        return {
            "lit": lit_count,
            "info": info_count
        }
    
    def get_question_type_info(self, prompt_name: str) -> Dict[str, Any]:
        """
        Get information about a question type.
        
        Args:
            prompt_name: The name of the prompt.
            
        Returns:
            A dictionary with information about the question type.
        """
        # Determine if we need a literary work or topic
        needs_work = prompt_name in LITERARY_QUESTION_TYPES
        needs_topic = prompt_name in TOPIC_QUESTION_TYPES
        
        # Determine the question subtype (lit or info)
        if prompt_name.endswith("-lit"):
            subtype = "lit"
        elif prompt_name.endswith("-info"):
            subtype = "info"
        else:
            subtype = "general"
        
        # Get the base type (without -lit/-info suffix)
        if "-" in prompt_name:
            base_type = prompt_name.split("-")[0]
        else:
            base_type = prompt_name
        
        return {
            "prompt_name": prompt_name,
            "base_type": base_type,
            "needs_work": needs_work,
            "needs_topic": needs_topic,
            "subtype": subtype,
            "has_variants": base_type in DUAL_VARIANT_TYPES
        }
    
    def get_question_validation_prompts(self) -> List[str]:
        """
        Get a list of available question validation prompt names.
        
        Returns:
            A list of prompt names for question validation.
        """
        return [p["name"] for p in self.prompts if p.get("function") == "quality control"]
    
    def generate_question(self, prompt_name: str, difficulty: str, 
                      work_or_topic: Optional[str] = None) -> Dict:
        """
        Generate a question using the specified prompt and parameters.
        Includes retry logic for handling API errors and JSON extraction failures.
        
        Args:
            prompt_name: Name of the prompt to use.
            difficulty: Difficulty level of the question.
            work_or_topic: Work or topic to use for the question, if applicable.
            
        Returns:
            A dictionary containing the generated question.
        """
        # Get the question type information
        type_info = self.get_question_type_info(prompt_name)
        
        # Get prompt object by name
        prompt_obj = get_prompt_by_name(self.prompts, prompt_name)
        
        if not prompt_obj:
            raise ValueError(f"Prompt '{prompt_name}' not found.")
        
        # Get the skill for this question type
        skill = type_info.get("skill", QUESTION_TYPE_TO_SKILL.get(prompt_name, "Reading Comprehension"))
        
        # Get the prompt type (base or variant)
        subtype = prompt_name
        base_type = type_info.get("base_type", prompt_name)
        
        # Determine the example type to use
        # Examples use "lit" for literary or "info" for informational questions
        if prompt_name.endswith("-lit"):
            example_type = "lit"
        elif prompt_name.endswith("-info"):
            example_type = "info"
        else:
            # For non-variant question types, default to "info" as they're typically informational
            example_type = "info"
            
        # Print debug information to help diagnose example selection issues
        print(f"Searching for examples with: skill='{skill}', difficulty='{difficulty}', type='{example_type}'")
        
        # Get a random example that matches the skill and difficulty
        example = self.get_random_example(skill, difficulty, example_type)
        
        # Get the prompt template
        prompt_template = prompt_obj["prompt"]
        
        # Replace placeholders in the prompt template
        if "{difficulty}" in prompt_template:
            prompt_template = prompt_template.replace("{difficulty}", difficulty)
        
        if example:
            # Replace the example placeholders
            prompt_template = prompt_template.replace("{example_passage}", example.get("passage", ""))
            prompt_template = prompt_template.replace("{example_question}", example.get("question", ""))
            prompt_template = prompt_template.replace("{example_correct_answer}", example.get("correct_answer", ""))
            
            # Handle distractors with different naming conventions
            example_distractor1 = example.get("distractor1", example.get("distractors", ["", "", ""])[0])
            example_distractor2 = example.get("distractor2", example.get("distractors", ["", "", ""])[1])
            example_distractor3 = example.get("distractor3", example.get("distractors", ["", "", ""])[2])
            
            prompt_template = prompt_template.replace("{example_distractor1}", example_distractor1)
            prompt_template = prompt_template.replace("{example_distractor2}", example_distractor2)
            prompt_template = prompt_template.replace("{example_distractor3}", example_distractor3)
        
        # Construct the completion prompt based on whether it needs a work or topic
        completion_prompt = ""
        
        if type_info["needs_work"]:
            # Fill in work for literary passages
            if not work_or_topic:
                work_or_topic = self.get_random_work()
            completion_prompt = f"""
            Please generate an SAT question following this prompt:
            
            {prompt_template.replace("{work}", work_or_topic)}
            """
        elif type_info["needs_topic"]:
            # Fill in topic for informational passages
            if not work_or_topic:
                work_or_topic = self.get_random_topic()
            completion_prompt = f"""
            Please generate an SAT question following this prompt:
            
            {prompt_template.replace("{topic}", work_or_topic)}
            """
        else:
            completion_prompt = f"""
            Please generate an SAT question following this prompt:
            
            {prompt_template}
            """
        
        # Call the Anthropic API to generate the question, with retry logic
        max_retries = int(os.getenv("MAX_API_RETRIES", "3"))
        retry_delay = float(os.getenv("API_RETRY_DELAY_SECONDS", "2.0"))
        
        for retry_attempt in range(max_retries):
            try:
                print(f"API call attempt {retry_attempt + 1}/{max_retries}...")
                
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4000,
                    temperature=0.2,
                    system="You are a professional SAT question writer who will generate high-quality SAT questions based on the provided prompt.",
                    messages=[
                        {
                            "role": "user",
                            "content": completion_prompt
                        }
                    ]
                )
                
                # Extract the generated question from the response
                question_json_str = self._extract_json_from_response(response.content[0].text)
                
                # Retry if JSON extraction failed
                if not question_json_str:
                    if retry_attempt < max_retries - 1:
                        print(f"No valid JSON found in the response. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise ValueError("No valid JSON found in the response after all retry attempts.")
                
                try:
                    question = json.loads(question_json_str)
                except json.JSONDecodeError as e:
                    if retry_attempt < max_retries - 1:
                        print(f"JSON decode error: {str(e)}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise ValueError(f"Failed to decode JSON after all retry attempts: {str(e)}")
                
                # Add metadata about the skill, type, and content used
                normalized_question = normalize_question(question)
                normalized_question["skill"] = skill
                normalized_question["type"] = subtype
                
                # For dual-variant types, add variant information
                if "-lit" in prompt_name:
                    normalized_question["variant"] = "literary"
                elif "-info" in prompt_name:
                    normalized_question["variant"] = "informational"
                
                # Special handling for COEQ questions with images
                if prompt_name == "coeq" or skill == "Command of Evidence: Quantitative":
                    normalized_question = self.process_coeq_question(normalized_question)
                
                # Return the generated question
                return normalized_question
                
            except Exception as e:
                if retry_attempt < max_retries - 1:
                    # Determine if it's a rate limit error (customize this based on Anthropic's API error format)
                    is_rate_limit = "rate limit" in str(e).lower() or "rate_limit" in str(e).lower() or "too many requests" in str(e).lower()
                    
                    # Add exponential backoff for rate limit errors
                    if is_rate_limit:
                        backoff_time = retry_delay * (2 ** retry_attempt)
                        print(f"Rate limit hit. Backing off for {backoff_time} seconds before retry {retry_attempt + 2}...")
                        time.sleep(backoff_time)
                    else:
                        print(f"API call error: {str(e)}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    continue
                else:
                    raise ValueError(f"Failed to generate question after {max_retries} attempts: {str(e)}")
        
        # This should never be reached due to the loop structure, but just in case
        raise ValueError("Failed to generate question due to unexpected error.")
    
    def generate_multiple_questions(self, question_type: str, difficulty: str, 
                               count: int, work_or_topic: Optional[str] = None) -> List[Dict]:
        """
        Generate multiple questions of a specified type and difficulty.
        Uses parallel processing to generate multiple questions simultaneously.
        
        Args:
            question_type: Type of questions to generate.
            difficulty: Difficulty level of questions.
            count: Number of questions to generate.
            work_or_topic: Optional work or topic to use for all questions. If None, random works/topics are used.
            
        Returns:
            A list of generated question dictionaries.
        """
        import concurrent.futures
        
        # Get display name and handle variant types (literary/informational)
        prompt_display = QUESTION_TYPE_DISPLAY.get(question_type, question_type)
        
        # Print message about generation
        print(f"Generating {count} {difficulty} {prompt_display} questions...")
        
        # Special handling for dual-variant question types (e.g., rc-lit, rc-info)
        if question_type in DUAL_VARIANT_TYPES:
            # Calculate the distribution of literary and informational questions
            distribution = self.calculate_variant_distribution(count)
            
            print(f"Distribution: {distribution['info']} informational, {distribution['lit']} literary")
            
            # Create a list to track the works and topics we've already used
            # This helps us avoid using the same work for multiple questions
            used_works = set()
            used_topics = set()
            
            # Initialize the question lists for both variants
            questions_lit = []
            questions_info = []
            
            # Define a worker function to generate a single question
            def generate_worker(i, variant, specific_work_or_topic=None):
                try:
                    # For informational passages, use a random topic from topics.json
                    if variant == "lit":
                        if specific_work_or_topic is None:
                            # Get a work we haven't used yet
                            unused_works = [w for w in self.works if w["title"] not in self.used_works]
                            
                            if unused_works:
                                # If we have unused works, select randomly
                                selected_work = random.choice(unused_works)
                            else:
                                # All works have been used, select least recently used one
                                work_usage = [(w, self.used_works.get(w["title"], 0)) for w in self.works]
                                work_usage.sort(key=lambda x: x[1])  # Sort by timestamp (oldest first)
                                selected_work = work_usage[0][0]
                                print("Warning: Exhausted all unique works! Using least recently used work.")
                            
                            specific_work = selected_work["title"]
                        else:
                            specific_work = specific_work_or_topic
                            
                        # Generate a literary question
                        variant_type = f"{question_type}-lit"
                        print(f"Generating {variant_type} question {i+1}/{count} with work: {specific_work}")
                        question = self.generate_question(variant_type, difficulty, specific_work)
                        
                        # Add the work to used works with current timestamp
                        self.used_works[specific_work] = time.time()
                        
                        # Add variant information to the question
                        question["variant"] = "literary"
                        return {"index": i, "question": question}
                    else:
                        if specific_work_or_topic is None:
                            # Get a topic we haven't used yet
                            unused_topics = [t for t in self.topics if t not in self.used_topics]
                            
                            if unused_topics:
                                # If we have unused topics, select randomly
                                specific_topic = random.choice(unused_topics)
                            else:
                                # All topics have been used, select least recently used one
                                topic_usage = [(t, self.used_topics.get(t, 0)) for t in self.topics]
                                topic_usage.sort(key=lambda x: x[1])  # Sort by timestamp (oldest first)
                                specific_topic = topic_usage[0][0]
                                print("Warning: Exhausted all unique topics! Using least recently used topic.")
                        else:
                            specific_topic = specific_work_or_topic
                            
                        # Generate an informational question
                        variant_type = f"{question_type}-info"
                        print(f"Generating {variant_type} question {i+1}/{count} with topic: {specific_topic}")
                        question = self.generate_question(variant_type, difficulty, specific_topic)
                        
                        # Add the topic to used topics with current timestamp
                        self.used_topics[specific_topic] = time.time()
                        
                        # Add variant information to the question
                        question["variant"] = "informational"
                        return {"index": i, "question": question}
                except Exception as e:
                    print(f"Error generating question {i+1}: {str(e)}")
                    return {"index": i, "error": str(e)}
            
            # Get the maximum number of workers from environment or default to half of CPU count
            max_workers = int(os.getenv("MAX_GENERATION_WORKERS", (os.cpu_count() or 4) // 2))
            print(f"Using up to {max_workers} parallel generation workers")
            
            # Generate questions in parallel
            all_results = []
            
            # Create tasks for literary variant
            lit_tasks = [(i, "lit", work_or_topic) for i in range(distribution["lit"])]
            
            # Create tasks for informational variant
            info_tasks = [(i + distribution["lit"], "info", work_or_topic) for i in range(distribution["info"])]
            
            # Combine all tasks
            all_tasks = lit_tasks + info_tasks
            
            # Use thread pool executor for API calls
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks to the executor
                future_to_index = {
                    executor.submit(generate_worker, i, variant, specific): i 
                    for i, variant, specific in all_tasks
                }
                
                # Collect results as they complete
                for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                                 desc="Generating questions", total=len(all_tasks)):
                    try:
                        result = future.result()
                        if "error" not in result:
                            all_results.append(result)
                    except Exception as e:
                        print(f"Task raised an exception: {str(e)}")
            
            # Sort results by original index to maintain order
            all_results.sort(key=lambda x: x["index"])
            
            # Extract just the questions
            all_questions = [result["question"] for result in all_results]
            
            return all_questions
            
        else:
            # Regular question types (non-dual-variant)
            
            # For works/topics that should change with each question
            if work_or_topic is None:
                # Define a worker function to generate a single question
                def generate_worker(i):
                    try:
                        # Get the question type info
                        question_info = self.get_question_type_info(question_type)
                        
                        # Determine if this question type needs a work or topic
                        if question_info["needs_work"]:
                            # Check if we have any unused works
                            unused_works = [w for w in self.works if w["title"] not in self.used_works]
                            
                            if unused_works:
                                # Select a random unused work
                                selected_work = random.choice(unused_works)
                            else:
                                # All works have been used, select least recently used
                                work_usage = [(w, self.used_works.get(w["title"], 0)) for w in self.works]
                                work_usage.sort(key=lambda x: x[1])  # Sort by timestamp (oldest first)
                                selected_work = work_usage[0][0]
                                print("Warning: All works have been used. Using least recently used work.")
                                
                            specific_work = f"{selected_work['title']} by {selected_work['author']} ({selected_work['year']}, {selected_work['type']})"
                            print(f"Generating question {i+1}/{count} with work: {selected_work['title']}")
                            question = self.generate_question(question_type, difficulty, specific_work)
                            
                            # Mark this work as used
                            self.used_works[selected_work["title"]] = time.time()
                            
                        elif question_info["needs_topic"]:
                            # Check if we have any unused topics
                            unused_topics = [t for t in self.topics if t not in self.used_topics]
                            
                            if unused_topics:
                                # Select a random unused topic
                                specific_topic = random.choice(unused_topics)
                            else:
                                # All topics have been used, select least recently used
                                topic_usage = [(t, self.used_topics.get(t, 0)) for t in self.topics]
                                topic_usage.sort(key=lambda x: x[1])  # Sort by timestamp (oldest first)
                                specific_topic = topic_usage[0][0]
                                print("Warning: All topics have been used. Using least recently used topic.")
                                
                            print(f"Generating question {i+1}/{count} with topic: {specific_topic}")
                            question = self.generate_question(question_type, difficulty, specific_topic)
                            
                            # Mark this topic as used
                            self.used_topics[specific_topic] = time.time()
                            
                        else:
                            # No work or topic needed
                            print(f"Generating question {i+1}/{count}")
                            question = self.generate_question(question_type, difficulty)
                        
                        return {"index": i, "question": question}
                    except Exception as e:
                        print(f"Error generating question {i+1}: {str(e)}")
                        return {"index": i, "error": str(e)}
                
                # Get the maximum number of workers from environment or default to half of CPU count
                max_workers = int(os.getenv("MAX_GENERATION_WORKERS", (os.cpu_count() or 4) // 2))
                print(f"Using up to {max_workers} parallel generation workers")
                
                all_results = []
                
                # Use thread pool executor for API calls
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks to the executor
                    future_to_index = {
                        executor.submit(generate_worker, i): i 
                        for i in range(count)
                    }
                    
                    # Collect results as they complete
                    for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                                     desc="Generating questions", total=count):
                        try:
                            result = future.result()
                            if "error" not in result:
                                all_results.append(result)
                        except Exception as e:
                            print(f"Task raised an exception: {str(e)}")
                
                # If we couldn't generate enough questions due to errors, try to generate more
                if len(all_results) < count:
                    print(f"Generated only {len(all_results)}/{count} questions due to errors. Trying to generate more...")
                    additional_needed = count - len(all_results)
                    
                    # Generate the remaining questions sequentially to avoid more errors
                    for i in range(additional_needed):
                        try:
                            result = generate_worker(len(all_results) + i)
                            if "error" not in result:
                                all_results.append(result)
                        except Exception as e:
                            print(f"Error generating additional question: {str(e)}")
                
                # Sort results by original index to maintain order
                all_results.sort(key=lambda x: x["index"])
                
                # Extract just the questions
                all_questions = [result["question"] for result in all_results]
                
                return all_questions
            else:
                # For fixed work/topic
                questions = []
                
                # Define a worker function to generate a single question
                def generate_worker(i):
                    try:
                        print(f"Generating question {i+1}/{count} with fixed work/topic: {work_or_topic}")
                        question = self.generate_question(question_type, difficulty, work_or_topic)
                        return {"index": i, "question": question}
                    except Exception as e:
                        print(f"Error generating question {i+1}: {str(e)}")
                        return {"index": i, "error": str(e)}
                
                # Get the maximum number of workers from environment or default to half of CPU count
                max_workers = int(os.getenv("MAX_GENERATION_WORKERS", (os.cpu_count() or 4) // 2))
                print(f"Using up to {max_workers} parallel generation workers")
                
                all_results = []
                
                # Use thread pool executor for API calls
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks to the executor
                    future_to_index = {
                        executor.submit(generate_worker, i): i 
                        for i in range(count)
                    }
                    
                    # Collect results as they complete
                    for future in tqdm(concurrent.futures.as_completed(future_to_index), 
                                     desc="Generating questions", total=count):
                        try:
                            result = future.result()
                            if "error" not in result:
                                all_results.append(result)
                        except Exception as e:
                            print(f"Task raised an exception: {str(e)}")
                
                # If we couldn't generate enough questions due to errors, try to generate more
                if len(all_results) < count:
                    print(f"Generated only {len(all_results)}/{count} questions due to errors. Trying to generate more...")
                    additional_needed = count - len(all_results)
                    
                    # Generate the remaining questions sequentially to avoid more errors
                    for i in range(additional_needed):
                        try:
                            result = generate_worker(len(all_results) + i)
                            if "error" not in result:
                                all_results.append(result)
                        except Exception as e:
                            print(f"Error generating additional question: {str(e)}")
                
                # Sort results by original index to maintain order
                all_results.sort(key=lambda x: x["index"])
                
                # Extract just the questions
                all_questions = [result["question"] for result in all_results]
                
                return all_questions
    
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
            
    def process_coeq_question(self, question: Dict) -> Dict:
        """
        Process a Command of Evidence: Quantitative question by:
        1. Extracting the data visualization description from the alt attribute
        2. Generating Python code to create the visualization
        3. Executing the code to create the image
        4. Uploading the image to Google Drive
        5. Updating the question's img tag with the actual URL
        
        Args:
            question: The question dictionary with the img tag containing alt description
            
        Returns:
            The updated question dictionary with the img URL pointing to the hosted image
        """
        # Extract the alt description from the img tag in the passage
        passage = question.get("passage", "")
        import re
        img_pattern = re.compile(r'<img[^>]*alt=["\'](.*?)["\'][^>]*>')
        img_match = img_pattern.search(passage)
        
        if not img_match:
            print("Warning: No img tag with alt description found in the passage.")
            return question
            
        alt_description = img_match.group(1)
        
        # Generate Python code for the visualization
        visualization_code = self.generate_visualization_code(alt_description)
        
        # Save the generated code for reference
        import os
        import time
        os.makedirs("generated_code", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        code_filename = f"generated_code/visualization_{timestamp}.py"
        
        with open(code_filename, "w", encoding="utf-8") as f:
            f.write(visualization_code)
            
        # Execute the code to generate the image
        image_filename = f"generated_code/visualization_{timestamp}.png"
        
        # Add the image filename to the code environment
        visualization_code = visualization_code.replace(
            "plt.savefig('visualization.png')", 
            f"plt.savefig('{image_filename}')"
        )
        
        # Execute the code
        exec_globals = {'__name__': '__main__'}
        try:
            exec(visualization_code, exec_globals)
            print(f"Visualization saved to {image_filename}")
        except Exception as e:
            print(f"Error executing visualization code: {str(e)}")
            return question
            
        # Upload the image to Google Drive
        image_url = self.upload_to_gdrive(image_filename)
        
        if not image_url:
            print("Warning: Failed to upload image to Google Drive.")
            return question
            
        # Update the question's img tag with the actual URL
        updated_passage = img_pattern.sub(
            lambda m: m.group(0).replace('alt="' + alt_description + '"', 
                                      f'alt="{alt_description}" src="{image_url}"'),
            passage
        )
        
        question["passage"] = updated_passage
        
        return question
        
    def generate_visualization_code(self, description: str) -> str:
        """
        Generate Python code to create a visualization based on the description.
        
        Args:
            description: Detailed description of the data visualization
            
        Returns:
            Python code as a string that will generate the visualization
        """
        prompt = f"""
        # Data Visualization Code Generation
        
        Please write Python code that generates a visualization based on the following description:
        
        "{description}"
        
        Requirements:
        1. Use matplotlib for generating the visualization
        2. Make sure all text is clearly visible and does not overlap
        3. Use appropriate font sizes and spacing
        4. Include a title and labels as specified in the description
        5. Use a clean, professional style suitable for an academic test
        6. Save the figure as 'visualization.png'
        7. Don't use plt.show() as this will be run in a non-interactive environment
        8. Use only the necessary imports (matplotlib, numpy, pandas if needed)
        9. The code should be self-contained and runnable without additional data files
        10. Use readable code with comments explaining key steps
        
        The code should be production-ready, error-free, and generate a high-quality visualization.
        """
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=4000,
                temperature=0.2,  # Use low temperature for consistent code
                system="You are a professional Python programmer and data visualization expert who creates clean, accurate, and readable matplotlib code.",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract the Python code from the response
            return self._extract_code_from_response(response.content[0].text)
        except Exception as e:
            print(f"Error generating visualization code: {str(e)}")
            return ""
            
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
        # (this is a fallback and likely not to be used)
        return response_text
        
    def upload_to_gdrive(self, image_path: str) -> Optional[str]:
        """
        Upload an image to Google Drive and return the public URL.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Public URL of the uploaded image, or None if upload failed
        """
        try:
            from pydrive.auth import GoogleAuth
            from pydrive.drive import GoogleDrive
            from oauth2client.service_account import ServiceAccountCredentials
            import os
            
            # Check if credentials file exists
            credentials_path = os.getenv("GDRIVE_CREDENTIALS_PATH", "gdrive_credentials.json")
            if not os.path.exists(credentials_path):
                print(f"Google Drive credentials not found at {credentials_path}")
                print("Please set GDRIVE_CREDENTIALS_PATH environment variable to your credentials file")
                return None
            
            # Authenticate with Google Drive
            gauth = GoogleAuth()
            
            # Try to use service account first
            try:
                scope = ['https://www.googleapis.com/auth/drive']
                gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
                    credentials_path, scope)
            except Exception as e:
                print(f"Service account authentication failed: {str(e)}")
                print("Falling back to saved credentials...")
                
                # Try using saved credentials
                gauth.LoadCredentialsFile("credentials.txt")
                if gauth.credentials is None:
                    print("No saved credentials found. Authentication required.")
                    
                    # Create a local webserver for authentication
                    gauth.LocalWebserverAuth()
                    
                    # Save credentials for future use
                    gauth.SaveCredentialsFile("credentials.txt")
            
            drive = GoogleDrive(gauth)
            
            # Create a file object
            file_drive = drive.CreateFile({'title': os.path.basename(image_path)})
            
            # Upload the file
            file_drive.SetContentFile(image_path)
            file_drive.Upload()
            
            # Make the file publicly viewable
            file_drive.InsertPermission({
                'type': 'anyone',
                'value': 'anyone',
                'role': 'reader'
            })
            
            # Get the public URL
            file_id = file_drive['id']
            public_url = f"https://drive.google.com/thumbnail?id={file_id}&sz=w800"
            
            print(f"Image uploaded to Google Drive: {public_url}")
            return public_url
            
        except ImportError as e:
            print(f"Required packages not installed: {str(e)}")
            print("Please install pydrive with: pip install pydrive")
            return None
        except Exception as e:
            print(f"Error uploading to Google Drive: {str(e)}")
            return None 