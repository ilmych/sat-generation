# SAT Question Generator

A sophisticated CLI tool for generating and quality-controlling SAT questions using Claude 3.7 Sonnet.

## Features

- Generate SAT questions of various types and difficulty levels
- Quality control generated questions using comprehensive validation criteria
- Automatically fix questions that fail validation
- Select appropriate examples based on skill, difficulty, and question type
- Save generated questions with detailed explanations to JSON files
- Configure the system via environment variables
- Generate visualizations for quantitative evidence questions
- Upload and host images on Google Drive
- Parallel processing for efficient generation, validation, and fixing
- Robust error handling and retry mechanisms

## Setup

1. Clone this repository
2. Create a `.env` file with your configuration:

```
# Required
ANTHROPIC_API_KEY=your_api_key_here

# Optional with defaults
MODEL_NAME=claude-3-7-sonnet-20250219
OUTPUT_DIR=./output
MAX_VALIDATION_RETRIES=3
DEFAULT_QUESTION_COUNT=5
MAX_VALIDATION_WORKERS=4
MAX_FIX_WORKERS=4
MAX_GENERATION_WORKERS=2
MAX_API_RETRIES=3
API_RETRY_DELAY_SECONDS=2.0
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Google Drive Setup (for Quantitative Evidence Questions)

Command of Evidence: Quantitative questions require generating data visualizations and uploading them to Google Drive. To set up Google Drive integration:

1. Run the setup script:

```bash
python setup_gdrive.py --new
```

2. This will create a template `client_secrets.json` file.
3. Visit the [Google Developer Console](https://console.developers.google.com/) to create API credentials.
4. Enable the Google Drive API for your project.
5. Create OAuth 2.0 credentials and add them to your `client_secrets.json` file.
6. Run the authentication process:

```bash
python setup_gdrive.py
```

7. Follow the browser prompts to authenticate and grant access.
8. Alternatively, if you have existing credentials, you can provide them:

```bash
python setup_gdrive.py --file path/to/credentials.json
```

## Usage

### Basic Usage

Run the CLI tool with:

```bash
python main.py
```

Follow the interactive prompts to:
1. Select question type
2. Choose difficulty level (easy, medium, hard)
3. Specify number of questions to generate

The system will:
1. Generate the requested questions
2. Validate each question for quality
3. Fix any questions that fail validation
4. Save the results to a timestamped JSON file in the output directory

### Special Question Types

#### Command of Evidence: Quantitative (coeq)

This question type involves a multi-step generation process:

1. The question is generated with a detailed description of a data visualization.
2. Python code is automatically generated to create the visualization.
3. The visualization is uploaded to Google Drive.
4. The question is updated with the image URL.

These questions test students' ability to interpret data from graphs, charts, or tables.

## Validation System

The system uses a comprehensive validation pipeline to ensure question quality:

### Validation Types

- **Plausibility**: Ensures distractors are plausible but incorrect
- **Question Quality**: Verifies questions are clear, well-structured, and answerable
- **Similarity**: Checks for redundancy among answer options
- **Correct/Distractor**: Confirms the marked correct answer is actually correct
- **Multiple Correct**: Ensures there's only one correct answer
- **Missing Correct Answer**: Verifies at least one option answers the question
- **Grammar Validation**: Specialized checks for boundaries and transitions questions 
- **Length Check**: Verifies answer options have comparable lengths
- **COEQ Image**: Validates data visualizations for clarity and readability

### Validation Process

1. Each question undergoes all applicable validations
2. Results are compiled with detailed reasoning for each validation
3. Questions that pass all validations have explanations generated
4. Failed validations are sent to the fix module

## Fix System

The system includes a robust question fixing pipeline:

### Fix Capabilities

- **General Fixes**: Corrects issues detected by validators
- **COEQ Image Fixes**: Improves data visualizations that fail validation
- **Content Fixes**: Adjusts wording, phrasing, and structure while preserving intent
- **Explanation Generation**: Creates detailed explanations for fixed questions

### Fix Process

1. Failed validations are analyzed to determine required fixes
2. Claude receives a targeted prompt to fix specific issues
3. Fixed questions are validated again if necessary
4. Fixes are applied with minimal changes to preserve question intent

## Error Handling

The system includes sophisticated error handling:

- **API Retry Logic**: Automatically retries failed API calls with exponential backoff
- **Robust Validation**: Handles unexpected response formats and failures gracefully
- **Fallback Mechanisms**: Provides alternative approaches when primary methods fail
- **Type Validation**: Verifies input and output formats to prevent runtime errors
- **Detailed Logging**: Provides comprehensive information for troubleshooting

## Configuration

Configure the following settings in the `.env` file:

- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `OUTPUT_DIR`: Directory for saving output files (default: `./output`)
- `MODEL_NAME`: Claude model to use (default: `claude-3-sonnet-20240229`)
- `GDRIVE_CREDENTIALS_PATH`: Path to Google Drive credentials file
- `MAX_VALIDATION_RETRIES`: Maximum retry attempts for validation (default: 3)
- `DEFAULT_QUESTION_COUNT`: Default number of questions to generate (default: 5)
- `MAX_VALIDATION_WORKERS`: Maximum parallel validation workers (default: 4)
- `MAX_FIX_WORKERS`: Maximum parallel fix workers (default: 4)
- `MAX_GENERATION_WORKERS`: Maximum parallel generation workers (default: 2)
- `MAX_API_RETRIES`: Maximum API retry attempts (default: 3)
- `API_RETRY_DELAY_SECONDS`: Delay between API retries (default: 2.0)

## Output Format

Generated questions are saved as JSON files with detailed metadata:

```json
{
  "metadata": {
    "question_type": "rc",
    "display_name": "Reading Comprehension",
    "difficulty": "medium",
    "count": 5,
    "variant_distribution": {"lit": 2, "info": 3},
    "generation_time": 45.2,
    "validation_time": 30.5,
    "fix_time": 12.8,
    "passed_count": 4,
    "failed_count": 1
  },
  "questions": [
    {
      "passage": "...",
      "question": "...",
      "correct_answer": "...",
      "distractor1": "...",
      "distractor2": "...",
      "distractor3": "...",
      "skill": "...",
      "domain": "...",
      "difficulty": "2",
      "type": "rc-info",
      "explanation_correct": "...",
      "explanation_distractor1": "...",
      "explanation_distractor2": "...",
      "explanation_distractor3": "..."
    }
  ],
  "validation_details": [...]
}
```

## Project Structure

```
sat-generation/
├── main.py                 # Entry point for the CLI
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (not in git)
├── sat-prompts.json        # Prompts for generating questions
├── sat-examples.json       # Example questions
├── setup_gdrive.py         # Script for Google Drive setup
├── src/
│   ├── __init__.py         # Package initialization
│   ├── generator.py        # Question generation module
│   ├── validator.py        # Question validation module
│   ├── cli.py              # CLI interface
│   ├── utils.py            # Utility functions
│   └── fix.py              # Fix module for correcting failed questions
├── generated_code/         # Generated visualization code and images
└── output/                 # Generated questions (not in git)
```

## Example Generation Process

1. **Selection**: User selects question type, difficulty, and count
2. **Generation**: System generates questions using Claude with appropriate prompts
3. **Validation**: Each question undergoes comprehensive validation
4. **Fixing**: Failed questions are automatically fixed
5. **Output**: All questions and metadata are saved to a JSON file

## Troubleshooting

If you encounter issues:

- Verify your Anthropic API key is correctly set in `.env`
- Check you have the correct permissions for Google Drive API
- Ensure all dependencies are installed (`pip install -r requirements.txt`)
- Look for error messages in the console output
- Try running test scripts to isolate issues 