import json
import os
import logging


def load_data(file_paths: dict):
    """Load data from JSON files and add a document type label."""
    samples = []
    logging.info("Loading data from JSON files...")
    for dtype, fpath in file_paths.items():
        if os.path.exists(fpath):
            logging.info(f"Loading '{dtype}' data from {fpath}")
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for sample in data:
                    sample["doc_type"] = dtype
                    samples.append(sample)
        else:
            logging.warning(f"File {fpath} does not exist.")
    logging.info(f"Loaded {len(samples)} total samples.")
    return samples


def extract_ground_truth(sample: dict):
    """
    Extract ground truth skills from a sample.
    Expected to be the list of skill names in sample["skills"].
    """
    gt_skills = []
    for skill_item in sample.get("skills", []):
        if "skill_name" in skill_item:
            gt_skills.append(skill_item["skill_name"].lower())
    return list(set(gt_skills))
