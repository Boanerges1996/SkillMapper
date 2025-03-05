import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from data import extract_ground_truth
from inference import bert_extract_skills_from_pipeline, gpt_extract_skills_from_model
from metrics import (
    compute_precision_recall_f1,
    compute_r_at_k,
    compute_mrr,
    compute_accuracy,
)


def evaluate_sample(sample, bert_extractors, gpt_models):
    """
    Evaluate a single sample with all active BERT models and all active GPT models.
    Returns a dictionary with predictions and computed metrics.
    """

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

    for model_name, gpt_instance in gpt_models.items():
        pred = gpt_extract_skills_from_model(text, gpt_instance, model_name)
        key_prefix = f"gpt_{model_name}"
        result[f"{key_prefix}_pred"] = pred
        precision, recall, f1 = compute_precision_recall_f1(gt_skills, pred)
        result[f"{key_prefix}_precision"] = precision
        result[f"{key_prefix}_recall"] = recall
        result[f"{key_prefix}_f1"] = f1
        result[f"{key_prefix}_r@5"] = compute_r_at_k(gt_skills, pred, k=5)
        result[f"{key_prefix}_mrr"] = compute_mrr(gt_skills, pred)
        result[f"{key_prefix}_accuracy"] = compute_accuracy(gt_skills, pred)

    return result


def evaluate_models(samples, bert_extractors, gpt_models, max_workers=8):
    results = []
    logging.info("Starting concurrent model evaluation on %d samples...", len(samples))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {
            executor.submit(
                evaluate_sample, sample, bert_extractors, gpt_models
            ): sample
            for sample in samples
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
