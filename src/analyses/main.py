import logging
from data import load_data
from config import gpt_models, load_bert_extractors
from evaluation import evaluate_models
from visualisation import generate_results_table, plot_metrics
import pandas as pd
import os


DATA_FILES = {
    "course": "./data/course_data.json",
    "cv": "./data/cv_data.json",
    "job": "./data/job_data.json",
}

if __name__ == "__main__":
    logging.info("Starting main execution...")
    samples = load_data(DATA_FILES)
    if not samples:
        logging.error("No samples loaded. Please check your JSON file paths.")
        exit(1)

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    bert_extractors = load_bert_extractors()
    eval_results = evaluate_models(samples, bert_extractors, gpt_models, max_workers=8)
    results_df = generate_results_table(eval_results)

    results_csv_path = "./results/evaluation_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    logging.info(f"Evaluation results saved to {results_csv_path}.")

    plot_metrics(eval_results, save_path="./results/metrics.png")
    logging.info("Script execution complete.")
