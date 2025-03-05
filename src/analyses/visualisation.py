import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_metrics(results, save_path="metrics.png"):
    """
    Plot bar charts for each evaluation metric.
    For each metric (precision, recall, f1, r@5, mrr, accuracy), plot the average score
    for each model (all BERT models and all GPT models). Each model is assigned a distinct color.
    Model names are annotated as labels on the bars.
    The plot is saved to disk at the given save_path.
    """
    metrics = ["precision", "recall", "f1", "r@5", "mrr", "accuracy"]
    # We know our model names from config.
    bert_names = ["ModernBERT", "RoBERTa"]
    gpt_names = [f"gpt_{name}" for name in ["Gemini", "Llama-3.2", "Deepseek-R1"]]
    all_models = bert_names + gpt_names

    avg_scores = {metric: {} for metric in metrics}
    for model in all_models:
        for metric in metrics:
            key = f"{model}_{metric}"
            values = [res[key] for res in results if key in res]
            avg_scores[metric][model] = np.mean(values) if values else 0

    colors = plt.cm.get_cmap("tab10").colors

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        model_names = list(avg_scores[metric].keys())
        scores = [avg_scores[metric][m] for m in model_names]
        model_colors = [colors[j % len(colors)] for j in range(len(model_names))]
        bars = axs[i].bar(range(len(model_names)), scores, color=model_colors)
        axs[i].set_title(f"Average {metric.capitalize()}")
        axs[i].set_ylim(0, 1)
        axs[i].set_xticks([])  # remove x-axis ticks
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axs[i].text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{model_names[j]}\n({height:.2f})",
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
