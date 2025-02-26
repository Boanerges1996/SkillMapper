#!/usr/bin/env python3
"""
Script to run the full comparative study between BERT and Generative AI models
for skill extraction.
"""

import os
import logging
import argparse
from skill_extraction_comparison import run_comparison
from comparison_results_analysis import run_analysis
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("comparative_study.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_api_keys():
    """Check if all required API keys are set"""
    required_keys = {
        "OPENAI_API_KEY": "GPT-4o",
        "GOOGLE_API_KEY": "Gemini",
        "ANTHROPIC_API_KEY": "Claude",
        # "META_API_KEY": "LLaMA" # Uncomment if using hosted Meta API
    }
    
    missing_keys = []
    for key, model in required_keys.items():
        if not os.environ.get(key):
            missing_keys.append(f"{key} for {model}")
    
    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        logger.warning("Some models will be skipped in the comparison.")
        return False
    
    return True

def main():
    """Run the full comparative study pipeline"""
    parser = argparse.ArgumentParser(description="Run comparative study of skill extraction methods")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip running the comparison and only analyze existing results")
    parser.add_argument("--models", nargs="+", choices=["bert", "gpt", "gemini", "claude", "llama", "all"], 
                        default=["all"], help="Specify which models to include in the comparison")
    parser.add_argument("--data-limit", type=int, default=100, help="Limit the number of examples per data type")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Check API keys
    api_keys_available = check_api_keys()
    if not api_keys_available:
        logger.info("Continuing with available API keys...")
    
    # Run the full pipeline or just the analysis
    if not args.skip_comparison:
        logger.info("Starting skill extraction comparison...")
        run_comparison(models=args.models, data_limit=args.data_limit)
        logger.info("Skill extraction comparison complete.")
    
    logger.info("Starting analysis of results...")
    run_analysis()
    logger.info("Analysis complete. Results available in analysis_output/ directory.")
    logger.info("See analysis_output/research_findings.md for a summary of the findings.")

if __name__ == "__main__":
    main()