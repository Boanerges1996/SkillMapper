import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import logging
import torch
from sentence_transformers import SentenceTransformer, util
import spacy
from sentence_transformers import util

# GPT models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOllama
from pydantic import BaseModel, Field
from difflib import SequenceMatcher
import nltk

nltk.download("punkt_tab")  # Ensure the tokenizer is available


def is_close_match(a: str, b: str, threshold=0.8) -> bool:
    """Check if two strings are sufficiently similar based on a threshold."""
    ratio = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return ratio >= threshold


def compute_rp_at_k_fuzzy(gt: set, pred: list, k=5, threshold=0.8) -> float:
    """
    Compute Ranked Precision at k (RP@k) using fuzzy matching:
    For the top k predictions, count how many are correct (i.e., have a fuzzy match with any gold skill),
    then divide by k.
    """
    top_k = pred[:k]
    correct_count = 0
    for p in top_k:
        if any(is_close_match(p, g, threshold) for g in gt):
            correct_count += 1
    return correct_count / k if k > 0 else 0


def compute_mrr_fuzzy_per_sample(gt: set, pred: list, threshold=0.7) -> float:
    reciprocal_ranks = []
    for g in gt:
        for idx, p in enumerate(pred):
            if is_close_match(p, g, threshold):
                reciprocal_ranks.append(1 / (idx + 1))
                break
    # Take the best rank per sample
    return max(reciprocal_ranks) if reciprocal_ranks else 0


def compute_mrr_fuzzy(gt: set, pred: list, threshold=0.8):
    """
    Compute Mean Reciprocal Rank using fuzzy matching:
    For each gold skill, find the rank of the first predicted skill that is a close match.
    """
    reciprocal_ranks = []
    for g in gt:
        found = False
        for idx, p in enumerate(pred):
            if is_close_match(p, g, threshold):
                reciprocal_ranks.append(1 / (idx + 1))
                found = True
                break
        if not found:
            reciprocal_ranks.append(0)
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0


def compute_accuracy_continuous(gt: set, pred: list, match_threshold=0.8) -> float:
    """
    Compute continuous accuracy as the fraction of gold skills (gt) that have at least one
    fuzzy match in the predictions (pred). No minimum threshold is used.
    """
    if not gt:
        return 0.0
    matched_count = 0
    for g in gt:
        if any(is_close_match(p, g, match_threshold) for p in pred):
            matched_count += 1
    return matched_count / len(gt)


def lexical_or_synonym_check(
    candidate: str, skill_name: str, synonyms: list[str]
) -> bool:
    """Check if the candidate phrase directly matches a skill, a synonym, or is a close match."""
    if skill_name.lower() in candidate.lower():
        return True
    for syn in synonyms:
        if syn.lower() in candidate.lower() or is_close_match(candidate, syn):
            return True
    if is_close_match(candidate, skill_name):
        return True
    return False


def prompt_based_skill_extraction(text: str, llm) -> list:
    """
    Use an LLM to extract skills from the text, including implicit or reworded mentions.
    """
    prompt = (
        "Read the following text and list any relevant ESCO skill labels. "
        "Include skills that might be mentioned implicitly or in reworded form. "
        "Return your answer as a JSON array of strings. "
        f"Text: '''{text}'''"
    )
    result = llm.invoke(prompt)
    try:
        extracted = json.loads(result.content)
    except Exception as e:
        logging.error("Error parsing LLM output: %s", e)
        extracted = []
    return [skill.lower() for skill in extracted]


# ------------------------------
# Setup & Environment
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
load_dotenv()

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# If GPU is available, tell spaCy to use it
if device == "cuda":
    spacy.require_gpu()

# Initialize SBERT model with the chosen device
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Load spaCy English model for noun phrase extraction
nlp = spacy.load("en_core_web_sm")

# ------------------------------
# Data File Paths
# ------------------------------
DATA_FILES = {
    "course": "./data/course_data.json",
    "cv": "./data/cv_data.json",
    "job": "./data/job_data.json",
}


# ------------------------------
# Data Loading & Ground Truth Extraction
# ------------------------------
def load_data(file_paths: dict):
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
    gt_skills = []
    for skill_item in sample.get("skills", []):
        if "skill_name" in skill_item:
            gt_skills.append(skill_item["skill_name"].lower())
    return list(set(gt_skills))


# ------------------------------
# Evaluation Metrics
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
# Load ESCO Skills with Precomputed Embeddings
# ------------------------------
def load_esco_skills(file_path: str) -> list:
    df = pd.read_csv(file_path)
    skills = []
    for _, row in df.iterrows():
        preferred = row.get("preferredLabel", "")
        alt_labels_raw = row.get("altLabels", "")
        alt_labels_list = []
        if isinstance(alt_labels_raw, str):
            alt_labels_list = [
                lbl.strip() for lbl in alt_labels_raw.split("\n") if lbl.strip()
            ]
        # Combine the preferred and alt labels for embedding
        combined_text = " ".join([preferred] + alt_labels_list)
        combined_embedding = embedding_model.encode(
            combined_text, convert_to_numpy=True, normalize_embeddings=True
        )
        skills.append(
            {
                "skill_name": preferred,
                "esco_id": row.get("conceptUri", ""),
                "alt_labels": alt_labels_list,
                "embedding": combined_embedding.tolist(),
            }
        )
    return skills


# ------------------------------
# Improved SBERT Token-Level Extraction
# ------------------------------
def extract_noun_phrases(text: str) -> list:
    """Extract noun phrases from text using spaCy."""
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]


def dynamic_cosine_threshold(similarities: np.ndarray) -> float:
    """
    Compute a dynamic threshold based on the candidate's similarity scores.
    New threshold: mean + 0.7 * std, with a minimum threshold of 0.7.
    """
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    return max(mean_sim + 1 * std_sim, 0.7)


def sbert_token_level_skill_extraction(text: str, esco_skills: list) -> list:
    noun_phrases = extract_noun_phrases(text)
    sentences = nltk.sent_tokenize(text)

    candidate_phrases = list(set(noun_phrases + sentences))
    candidate_phrases = [phrase for phrase in candidate_phrases if len(phrase) > 3]
    if not candidate_phrases:
        return []

    phrase_embeddings = embedding_model.encode(
        candidate_phrases, convert_to_numpy=True, normalize_embeddings=True
    )

    skill_texts = [skill["skill_name"] for skill in esco_skills]
    skill_embeddings = embedding_model.encode(
        skill_texts, convert_to_numpy=True, normalize_embeddings=True
    )

    similarity_matrix = np.dot(phrase_embeddings, skill_embeddings.T)

    # Use a dictionary to keep track of the highest similarity for each skill
    extracted_skills = {}
    for i, candidate in enumerate(candidate_phrases):
        sims = similarity_matrix[i]
        top_indices = sims.argsort()[::-1][:10]
        for idx in top_indices:
            if sims[idx] < 0.5:
                continue
            skill_candidate = skill_texts[idx].lower()
            alt_labels = esco_skills[idx].get("alt_labels", [])
            if lexical_or_synonym_check(candidate, skill_candidate, alt_labels):
                # Store the highest similarity score for the skill
                if skill_candidate in extracted_skills:
                    extracted_skills[skill_candidate] = max(
                        extracted_skills[skill_candidate], sims[idx]
                    )
                else:
                    extracted_skills[skill_candidate] = sims[idx]

    # Sort skills by similarity score in descending order
    sorted_skills = sorted(extracted_skills.items(), key=lambda x: x[1], reverse=True)
    return [skill for skill, score in sorted_skills]


# ------------------------------
# GPT Models Extraction (unchanged)
# ------------------------------
gpt_models = {
    "Gemini": ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=60,
        max_retries=2,
    ),
    "Llama-3.2": ChatOllama(
        model="llama3.2",
        max_tokens=4000,
        temperature=0.7,
        top_p=0.95,
    ),
    # Uncomment if you want to use Deepseek-R1
    # "Deepseek-R1": ChatOllama(
    #     model="deepseek-r1:14b",
    #     max_tokens=4000,
    #     temperature=0.7,
    #     top_p=0.95,
    # ),
}
gpt_semaphore = threading.Semaphore(2)


class SkillsOutput(BaseModel):
    skills: list[str] = Field(description="List of extracted skills as strings.")


def gpt_extract_skills_from_model(
    text: str, gpt_model_instance, model_name: str, max_attempts=5, base_delay=5
):
    logging.info(f"Starting GPT extraction using {model_name}...")
    if model_name.lower() == "deepseek-r1":
        prompt = (
            "Extract all explicitly mentioned skills from the following text, and those implicitly mentioned. "
            "Return ONLY a JSON object exactly in the following format: "
            '{"skills": ["skill1", "skill2", ...]}. '
            "Do not include any chain-of-thought, explanations, or extra text. list them in order of relevance."
            f"Text: '''{text}'''"
        )
    else:
        prompt = (
            "Extract all explicitly mentioned skills from the following text, and those implicitly mentioned, list them in order of relevance, and return a JSON object "
            "with a key 'skills' mapping to a list of strings. Do not include any additional text. "
            f"Text: '''{text}'''"
        )
    attempt = 1
    while attempt <= max_attempts:
        try:
            with gpt_semaphore:
                start_time = time.time()
                if model_name.lower() in ["llama-3.2", "deepseek-r1"]:
                    response = gpt_model_instance.invoke(prompt).content.strip()
                    if not response:
                        raise ValueError("Empty response received.")
                    logging.info(f"{model_name} raw response: {response}")
                    try:
                        parsed = json.loads(response)
                        skills = (
                            parsed["skills"]
                            if isinstance(parsed, dict) and "skills" in parsed
                            else []
                        )
                    except Exception as parse_error:
                        logging.warning(
                            f"Manual JSON parsing failed for {model_name} on attempt {attempt}: {parse_error}. Response: {response}"
                        )
                        raise ValueError("JSON parsing error")
                else:
                    model_with_structure = gpt_model_instance.with_structured_output(
                        SkillsOutput
                    )
                    structured_output = model_with_structure.invoke(prompt)
                    skills = structured_output.skills
                elapsed = time.time() - start_time
                logging.info(
                    f"{model_name} extraction complete in {elapsed:.2f} seconds on attempt {attempt}."
                )
                return [skill.lower() for skill in skills]
        except Exception as e:
            logging.error(f"{model_name} extraction failed on attempt {attempt}: {e}")
            delay = base_delay * attempt
            logging.info(
                f"Waiting {delay} seconds before retrying (attempt {attempt+1}/{max_attempts})."
            )
            time.sleep(delay)
            attempt += 1
    logging.error(f"{model_name} extraction failed after maximum attempts.")
    return []


def compute_accuracy_threshold(gt: set, pred: list, min_matches: int = 5) -> int:
    """
    Compute a binary accuracy score based on a threshold:
    If the number of fuzzy-matched gold skills is at least min_matches (or all if fewer than min_matches exist),
    return 1 (accurate); otherwise, return 0.
    """
    # Use the minimum of min_matches and the total gold count if there are fewer gold skills.
    threshold = min(min_matches, len(gt))
    matched_gold = set()
    for p in pred:
        for g in gt:
            if g in matched_gold:
                continue
            if is_close_match(p, g, threshold=0.8):
                matched_gold.add(g)
                break
    return 1 if len(matched_gold) >= threshold else 0


# ------------------------------
# Per-Sample Evaluation Function
# ------------------------------
def evaluate_sample(sample, esco_skills):
    # Combine text fields from the sample
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
    gt_set = set(gt_skills)

    # Use improved SBERT token-level extraction
    sbert_pred = sbert_token_level_skill_extraction(text, esco_skills)
    result["sbert_pred"] = sbert_pred

    # Evaluate GPT models predictions and store them in a dictionary
    gpt_preds = {}
    for model_name, gpt_instance in gpt_models.items():
        pred = gpt_extract_skills_from_model(text, gpt_instance, model_name)
        gpt_preds[model_name] = pred
        key_prefix = f"{model_name}"
        result[f"{key_prefix}_pred"] = pred

    # If any model (SBERT or any GPT) returns an empty prediction,
    # skip this sample from the evaluation.
    if not sbert_pred or any(len(gpt_preds[m]) == 0 for m in gpt_preds):
        logging.info(
            "Skipping sample due to empty predictions from one or more models."
        )
        return None

    # Compute continuous fuzzy metrics for SBERT predictions
    precision, recall, f1 = compute_precision_recall_f1_fuzzy(
        gt_set, sbert_pred, threshold=0.8
    )
    result["sbert_precision"] = precision
    result["sbert_recall"] = recall
    result["sbert_f1"] = f1
    # Instead of R@5, use RP@5:
    result["sbert_rp@5"] = compute_rp_at_k_fuzzy(gt_set, sbert_pred, k=5, threshold=0.8)
    result["sbert_mrr"] = compute_mrr_fuzzy_per_sample(
        gt_set, sbert_pred, threshold=0.7
    )
    result["sbert_accuracy"] = compute_accuracy_continuous(
        set(gt_skills), sbert_pred, match_threshold=0.8
    )

    # Compute metrics for each GPT model
    for model_name, pred in gpt_preds.items():
        key_prefix = f"{model_name}"
        precision, recall, f1 = compute_precision_recall_f1_fuzzy(
            gt_set, pred, threshold=0.8
        )
        result[f"{key_prefix}_precision"] = precision
        result[f"{key_prefix}_recall"] = recall
        result[f"{key_prefix}_f1"] = f1
        result[f"{key_prefix}_rp@5"] = compute_rp_at_k_fuzzy(
            gt_set, pred, k=5, threshold=0.8
        )
        result[f"{key_prefix}_mrr"] = compute_mrr_fuzzy_per_sample(
            gt_set, pred, threshold=0.7
        )
        result[f"{key_prefix}_accuracy"] = compute_accuracy_continuous(
            set(gt_skills), pred, match_threshold=0.8
        )

    return result


def compute_precision_recall_f1_fuzzy(gt: set, pred: list, threshold=0.8):
    """
    Compute precision, recall, and F1 using fuzzy matching.
    A predicted skill counts as correct if it closely matches a gold skill (per is_close_match).
    Each gold skill is matched only once.
    """
    matched_gold = set()
    for p in pred:
        for g in gt:
            # If this gold skill is already matched, skip it.
            if g in matched_gold:
                continue
            if is_close_match(p, g, threshold):
                matched_gold.add(g)
                break

    true_positives = len(matched_gold)
    precision = true_positives / len(pred) if pred else 0
    recall = true_positives / len(gt) if gt else 0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0
    )
    return precision, recall, f1


# ------------------------------
# Concurrent Evaluation of Models
# ------------------------------
def evaluate_models(samples, esco_skills, max_workers=8):
    results = []
    logging.info("Starting concurrent model evaluation on %d samples...", len(samples))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(evaluate_sample, sample, esco_skills): sample
            for sample in samples
        }
        for future in as_completed(future_to_sample):
            try:
                result = future.result()
                if (
                    result is not None
                ):  # Only keep samples with predictions from all models
                    results.append(result)
                else:
                    logging.info("A sample was skipped due to empty predictions.")
            except Exception as exc:
                logging.error("Sample evaluation generated an exception: %s", exc)
    logging.info("Concurrent model evaluation complete.")
    return results


# ------------------------------
# Results Table Generation & Plotting
# ------------------------------
def generate_results_table(results):
    df = pd.DataFrame(results)

    def join_list(x):
        return ", ".join(x) if isinstance(x, list) else x

    for col in df.columns:
        if "pred" in col or col == "ground_truth":
            df[col] = df[col].apply(join_list)
    return df


def plot_metrics(results, save_path="metrics.png"):
    # Remove 'accuracy' from the metrics to be plotted
    metrics = ["precision", "recall", "f1", "rp@5", "mrr", "accuracy"]
    model_names = ["sbert"] + [f"{name}" for name in gpt_models.keys()]
    avg_scores = {metric: {} for metric in metrics}
    for model in model_names:
        for metric in metrics:
            key = f"{model}_{metric}"
            values = [res[key] for res in results if key in res]
            avg_scores[metric][model] = np.mean(values) if values else 0
    colors = plt.cm.get_cmap("tab10").colors
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        model_names_metric = list(avg_scores[metric].keys())
        scores = [avg_scores[metric][m] for m in model_names_metric]
        model_colors = [colors[j % len(colors)] for j in range(len(model_names_metric))]
        bars = axs[i].bar(range(len(model_names_metric)), scores, color=model_colors)
        axs[i].set_title(f"Average {metric.capitalize()}")
        axs[i].set_ylim(0, 1)
        axs[i].set_xticks([])
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axs[i].text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{model_names_metric[j]}\n({height:.2f})",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"Metrics plot saved to {save_path}.")
    plt.show()


def generate_summary_table(results):
    """
    Generate a summary table that aggregates key metrics for each model.
    Each row represents a model (SBERT, Gemini, Llama-3.2, etc.) and each column is the
    average of a particular metric across all evaluated samples.
    """
    models = ["sbert"] + list(gpt_models.keys())
    metrics = ["precision", "recall", "f1", "r@5", "mrr", "accuracy"]
    summary = {}
    for model in models:
        summary[model] = {}
        for metric in metrics:
            key = f"{model}_{metric}"
            metric_values = [res[key] for res in results if key in res]
            summary[model][metric] = np.mean(metric_values) if metric_values else None
    return pd.DataFrame(summary).T


# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    logging.info("Starting main execution...")
    # Load ESCO skills (with SBERT embeddings)
    esco_skills = load_esco_skills("skills_en.csv")

    samples = load_data(DATA_FILES)
    if not samples:
        logging.error("No samples loaded. Please check your JSON file paths.")
        exit(1)

    eval_results = evaluate_models(samples, esco_skills, max_workers=8)
    eval_results = [r for r in eval_results if r is not None]
    results_df = generate_results_table(eval_results)
    results_csv_path = "evaluation_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    logging.info(f"Evaluation results saved to {results_csv_path}.")

    # Generate and log a summary table
    summary_df = generate_summary_table(eval_results)
    print("Summary Evaluation Table:")
    print(summary_df.to_string())
    logging.info("Evaluation Results per Sample:")
    logging.info("\n" + results_df.to_string(index=False))
    plot_metrics(eval_results, save_path="metrics.png")
    logging.info("Script execution complete.")
