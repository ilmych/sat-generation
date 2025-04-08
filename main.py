#!/usr/bin/env python3
"""
SAT Question Generator - Main CLI Entry Point
"""

import os
import sys
import json
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Add the current directory to the path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from src.cli import run_cli

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run the CLI application
    run_cli() 