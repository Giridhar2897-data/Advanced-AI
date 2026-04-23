import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)


def best_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0,
    )
    idx = np.argmax(f1[:-1])
    return thresholds[idx], f1[idx]


def compute_metrics(y_true, y_scores):
    thresh, _ = best_threshold(y_true, y_scores)
    y_pred = (y_scores >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "AUPRC": round(average_precision_score(y_true, y_scores), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_scores), 4),
        "F1": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Threshold": round(thresh, 4),
    }


def cost_analysis(y_true, y_scores, fraud_loss=500, review_cost=5, decline_cost=50):
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fn * fraud_loss + (tp + fp) * review_cost + fp * decline_cost
        costs.append(cost)
    return thresholds, np.array(costs)
