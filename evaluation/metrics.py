"""
Evaluation metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix (threshold 0.5).
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """
    From probabilities and true labels compute all metrics.
    y_true: (N,) 0/1. y_prob: (N,) in [0, 1].
    Returns dict with accuracy, precision, recall, f1, roc_auc, confusion_matrix (2x2).
    """
    y_pred = (y_prob >= threshold).astype(np.int64)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape != (2, 2):
        cm = np.zeros((2, 2), dtype=np.int64)
        cm[0, 0] = np.sum((y_true == 0) & (y_pred == 0))
        cm[0, 1] = np.sum((y_true == 0) & (y_pred == 1))
        cm[1, 0] = np.sum((y_true == 1) & (y_pred == 0))
        cm[1, 1] = np.sum((y_true == 1) & (y_pred == 1))
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }


def get_roc_curve(y_true: np.ndarray, y_prob: np.ndarray):
    """Return fpr, tpr, thresholds for ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds
