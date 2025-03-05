# config.py
import os

# File paths
ESCO_CSV_PATH = (
    "skills_en.csv"  # CSV with fields: conceptUri, preferredLabel, description, etc.
)
COURSE_JSON_PATH = "./data/course.json"
JOB_JSON_PATH = "./data/job.json"
CV_JSON_PATH = "./data/cv.json"

# Maximum number of threads (based on available cores)
MAX_WORKERS = os.cpu_count() or 1

# LLM parameters
LLM_CHOICE = os.getenv("LLM_CHOICE", "gemini")  # Options: "gemini" or "llama"
MAX_TOKENS = 4000
TEMPERATURE = 0.7
TOP_P = 0.9
FREQUENCY_PENALTY = 0.5  # Used only for Gemini
TIMEOUT = 30
MAX_RETRIES = 6  # For exponential backoff/retry logic

# Ollama-specific model name (as recognized by your Ollama instance)
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3.2")
