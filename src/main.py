import json
import os
import re
import getpass
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging

# Set up logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

load_dotenv()

# ------------------------------
# SET UP GOOGLE API KEY
# ------------------------------
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# ------------------------------
# MODEL IMPORTS & CONFIGURATION
# ------------------------------
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Define a dictionary of BERT model names to use.
bert_model_names = {
    "ModernBERT": "answerdotai/ModernBERT-base",
    "RoBERTa": "Jean-Baptiste/roberta-large-ner-english",
}

# Load each model into a dictionary of NER pipelines.
bert_extractors = {}
for model_name, model_id in bert_model_names.items():
    logging.info(f"Loading {model_name} model from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    # Use aggregation_strategy "simple" and force CPU (device=-1)
    bert_extractors[model_name] = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,
    )

# ------------------------------
# Google Gemini (GPT) Configuration via LangChain
# ------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI

logging.info("Initializing GPT (Gemini) model...")
gpt_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=60,
    max_retries=2,
)

# Limit concurrent GPT calls with a semaphore (e.g., max 2 at a time)
gpt_semaphore = threading.Semaphore(2)

# ------------------------------
# Pydantic Schema for Structured Output (for GPT)
# ------------------------------
from pydantic import BaseModel, Field


class SkillsOutput(BaseModel):
    skills: list[str] = Field(description="List of extracted skills as strings.")


# ------------------------------
# DATA FILE PATHS
# ------------------------------
DATA_FILES = {
    "course": "course_data.json",
    "cv": "cv_data.json",
    "job": "job_data.json",
}


# ------------------------------
# DATA LOADING & PREPROCESSING
# ------------------------------
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


# ------------------------------
# MODEL INFERENCE FUNCTIONS
# ------------------------------
def bert_extract_skills_from_pipeline(text: str, extractor):
    """
    Use the given BERT NER pipeline to extract skills from text.
    Performs basic cleaning (removes punctuation) and deduplication.
    """
    logging.info("Starting BERT extraction using extractor...")
    start_time = time.time()
    ner_results = extractor(text)
    extracted = [
        entity["word"].lower() for entity in ner_results if len(entity["word"]) > 2
    ]
    extracted = list(
        set([re.sub(r"[^a-zA-Z0-9\s]", "", skill).strip() for skill in extracted])
    )
    elapsed = time.time() - start_time
    logging.info(
        f"BERT extraction complete in {elapsed:.2f} seconds; found {len(extracted)} skills."
    )
    return extracted


def gpt_extract_skills(text: str, max_attempts=5, base_delay=5):
    """
    Use the Gemini model via ChatGoogleGenerativeAI to extract skills,
    leveraging LangChain's structured output functionality.

    The prompt instructs the model to return only a JSON object with key "skills".
    In case of failures, retries occur with exponential backoff.
    """
    logging.info("Starting GPT extraction using structured output...")
    prompt = (
        "Extract all explicitly mentioned skills from the following text and return a JSON object "
        "with a key 'skills' mapping to a list of strings. Do not include any additional text. "
        f"Text: '''{text}'''"
    )

    attempt = 1
    while attempt <= max_attempts:
        try:
            with gpt_semaphore:
                start_time = time.time()
                model_with_structure = gpt_model.with_structured_output(SkillsOutput)
                structured_output = model_with_structure.invoke(prompt)
                elapsed = time.time() - start_time
                logging.info(
                    f"GPT extraction complete in {elapsed:.2f} seconds on attempt {attempt}."
                )
                return [skill.lower() for skill in structured_output.skills]
        except Exception as e:
            logging.error(f"GPT extraction failed on attempt {attempt}: {e}")
            delay = base_delay * attempt
            logging.info(
                f"Waiting {delay} seconds before retrying (attempt {attempt+1}/{max_attempts})."
            )
            time.sleep(delay)
            attempt += 1

    logging.error("GPT extraction failed after maximum attempts.")
    return []


# ------------------------------
# EVALUATION METRICS FUNCTIONS
# ------------------------------
def compute_precision_recall_f1(gt: set, pred: list):
    gt = set(gt)
    pred = list(dict.fromkeys(pred))
    true_positives = len(gt.intersection(pred))
    precision = true_positives / len(pred) if pred else 0
    recall = true_positives / len(gt) if gt else 0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1


def compute_r_at_k(gt: set, pred: list, k=5):
    gt = set(gt)
    top_k = pred[:k]
    hit_count = sum([1 for skill in gt if skill in top_k])
    return hit_count / len(gt) if gt else 0


def compute_mrr(gt: set, pred: list):
    gt = set(gt)
    reciprocal_ranks = []
    for skill in gt:
        try:
            rank = pred.index(skill) + 1
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0


def compute_accuracy(gt: set, pred: list):
    return 1 if set(pred) == gt else 0


# ------------------------------
# PER-SAMPLE EVALUATION FUNCTION
# ------------------------------
def evaluate_sample(sample):
    """Evaluate a single sample with all BERT models and GPT; return metrics."""
    # Combine all text parts.
    text_parts = []
    if isinstance(sample.get("text"), dict):
        for key, value in sample["text"].items():
            if isinstance(value, str):
                text_parts.append(value)
    text = "\n".join(text_parts)
    logging.info(
        "Processing sample of type '%s' with text length %d",
        sample.get("doc_type", "unknown"),
        len(text),
    )

    gt_skills = extract_ground_truth(sample)
    logging.info("Ground truth skills: %s", gt_skills)

    result = {"doc_type": sample.get("doc_type", "unknown"), "ground_truth": gt_skills}

    # For each BERT model, extract skills and compute metrics.
    for model_name, extractor in bert_extractors.items():
        pred = bert_extract_skills_from_pipeline(text, extractor)
        result[f"{model_name}_pred"] = pred
        precision, recall, f1 = compute_precision_recall_f1(gt_skills, pred)
        result[f"{model_name}_precision"] = precision
        result[f"{model_name}_recall"] = recall
        result[f"{model_name}_f1"] = f1
        result[f"{model_name}_r@5"] = compute_r_at_k(gt_skills, pred, k=5)
        result[f"{model_name}_mrr"] = compute_mrr(gt_skills, pred)
        result[f"{model_name}_accuracy"] = compute_accuracy(gt_skills, pred)

    # GPT extraction.
    gpt_pred = gpt_extract_skills(text)
    result["gpt_pred"] = gpt_pred
    gpt_precision, gpt_recall, gpt_f1 = compute_precision_recall_f1(gt_skills, gpt_pred)
    result["gpt_precision"] = gpt_precision
    result["gpt_recall"] = gpt_recall
    result["gpt_f1"] = gpt_f1
    result["gpt_r@5"] = compute_r_at_k(gt_skills, gpt_pred, k=5)
    result["gpt_mrr"] = compute_mrr(gt_skills, gpt_pred)
    result["gpt_accuracy"] = compute_accuracy(gt_skills, gpt_pred)

    return result


# ------------------------------
# RUN EVALUATION ACROSS SAMPLES CONCURRENTLY
# ------------------------------
def evaluate_models(samples, max_workers=8):
    results = []
    logging.info("Starting concurrent model evaluation on %d samples...", len(samples))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(evaluate_sample, sample): sample for sample in samples
        }
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                logging.error("Sample evaluation generated an exception: %s", exc)
    logging.info("Concurrent model evaluation complete.")
    return results


# ------------------------------
# VISUALIZATION FUNCTIONS
# ------------------------------
def generate_results_table(results):
    """Generate a Pandas DataFrame summarizing predictions and metrics per sample."""
    df = pd.DataFrame(results)

    # Convert list columns to comma-separated strings.
    def join_list(x):
        return ", ".join(x) if isinstance(x, list) else x

    for col in df.columns:
        if "pred" in col or col == "ground_truth":
            df[col] = df[col].apply(join_list)
    return df


def plot_metrics(results):
    """
    Plot bar charts for each evaluation metric.
    For each metric (precision, recall, f1, r@5, mrr, accuracy),
    we plot the average score for each model (each BERT model and GPT).
    """
    metrics = ["precision", "recall", "f1", "r@5", "mrr", "accuracy"]
    model_names = list(bert_extractors.keys()) + ["gpt"]
    # Prepare a dictionary: metric -> {model_name: average_value}
    avg_scores = {metric: {} for metric in metrics}
    for model in model_names:
        for metric in metrics:
            key = f"{model}_{metric}" if model != "gpt" else f"gpt_{metric}"
            values = [res[key] for res in results if key in res]
            avg_scores[metric][model] = np.mean(values) if values else 0

    # Get distinct colors from the tab10 colormap.
    colors = plt.cm.get_cmap("tab10").colors

    # Plot each metric in a subplot.
    n_metrics = len(metrics)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        models = list(avg_scores[metric].keys())
        scores = [avg_scores[metric][m] for m in models]
        # Assign a distinct color to each model (cycling if necessary)
        model_colors = [colors[j % len(colors)] for j in range(len(models))]
        axs[i].bar(models, scores, color=model_colors)
        axs[i].set_title(f"Average {metric.capitalize()}")
        axs[i].set_ylim(0, 1)
        for j, score in enumerate(scores):
            axs[i].annotate(
                f"{score:.2f}",
                xy=(j, score),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
    plt.tight_layout()
    plt.show()


# ------------------------------
# MAIN EXECUTION
# ------------------------------
if __name__ == "__main__":
    logging.info("Starting main execution...")
    samples = load_data(DATA_FILES)
    if not samples:
        logging.error("No samples loaded. Please check your JSON file paths.")
        exit(1)

    eval_results = evaluate_models(samples, max_workers=8)
    results_df = generate_results_table(eval_results)
    logging.info("Evaluation Results per Sample:")
    logging.info("\n" + results_df.to_string(index=False))
    plot_metrics(eval_results)
    logging.info("Script execution complete.")
