import numpy as np


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
