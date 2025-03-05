import logging
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import Fore
from typing import List, Dict, TypedDict

from config import (
    ESCO_CSV_PATH,
    MAX_WORKERS,
    COURSE_JSON_PATH,
    JOB_JSON_PATH,
    CV_JSON_PATH,
)
from llm_utils import update_json_file, call_llm_for_skill

# Locks for thread-safe file updates
course_lock = threading.Lock()
job_lock = threading.Lock()
cv_lock = threading.Lock()


def process_skill_row(skill: dict, llm) -> None:
    """
    For a single ESCO skill (row), concurrently generate sentences for course, job, and CV.
    Save each output to its respective JSON file.
    """
    data_types = ["course", "job", "cv"]
    results = {}

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_type = {
            executor.submit(call_llm_for_skill, data_type, skill, llm): data_type
            for data_type in data_types
        }
        for future in as_completed(future_to_type):
            data_type = future_to_type[future]
            try:
                result = future.result()
                results[data_type] = result
                logging.info(
                    f"{Fore.GREEN}Generated {data_type} sentences for ESCOID: {skill['conceptUri']}"
                )
            except Exception as e:
                logging.error(
                    f"{Fore.RED}Error generating {data_type} sentences for ESCOID {skill['conceptUri']}: {e}"
                )

    # Update the JSON files (each update is thread-safe)
    if "course" in results:
        update_json_file(COURSE_JSON_PATH, results["course"], course_lock)
    if "job" in results:
        update_json_file(JOB_JSON_PATH, results["job"], job_lock)
    if "cv" in results:
        update_json_file(CV_JSON_PATH, results["cv"], cv_lock)


# Define a simple state schema to pass data between nodes.
class State(TypedDict):
    esco_skills: List[Dict[str, str]]
    processed_count: int


def load_esco_node(state: State) -> dict:
    """
    Load the ESCO CSV into state. Only keep fields: conceptUri, preferredLabel, description.
    """
    logging.info(f"{Fore.CYAN}Loading ESCO dataset from {ESCO_CSV_PATH} ...")
    df = pd.read_csv(ESCO_CSV_PATH)
    df = df[["conceptUri", "preferredLabel", "description"]]
    skills = df.to_dict(orient="records")
    logging.info(f"{Fore.GREEN}Loaded {len(skills)} skills from ESCO dataset.")
    return {"esco_skills": skills, "processed_count": 0}


def process_all_skills_node(state: State, llm) -> dict:
    """
    Iterate through all ESCO skills and process each to generate sentences.
    Uses a ThreadPoolExecutor to process multiple rows concurrently.
    """
    skills = state["esco_skills"]
    total_skills = len(skills)
    logging.info(f"{Fore.CYAN}Starting generation for all {total_skills} skills...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_skill_row, skill, llm): idx
            for idx, skill in enumerate(skills, start=1)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()
                state["processed_count"] += 1
                logging.info(
                    f"{Fore.YELLOW}Processed {state['processed_count']}/{total_skills} skills."
                )
            except Exception as e:
                logging.error(f"{Fore.RED}Error processing skill at index {idx}: {e}")

    return state


def finish_node(state: State) -> dict:
    logging.info(
        f"{Fore.GREEN}All skills processed. Total skills processed: {state['processed_count']}"
    )
    return {}
