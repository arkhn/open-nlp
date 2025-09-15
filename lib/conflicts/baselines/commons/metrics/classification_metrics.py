from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names for reporting

    Returns:
        Dictionary with metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Precision, recall, F1 (macro and weighted averages)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "support": support.tolist(),
    }

    # Add per-class metrics if labels provided
    if labels is not None:
        per_class_metrics = {}
        for i, label in enumerate(labels):
            per_class_metrics[f"{label}_precision"] = precision_per_class[i]
            per_class_metrics[f"{label}_recall"] = recall_per_class[i]
            per_class_metrics[f"{label}_f1"] = f1_per_class[i]
            per_class_metrics[f"{label}_support"] = support[i]

        metrics.update(per_class_metrics)

    return metrics


def print_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None
) -> str:
    """
    Print detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names for reporting

    Returns:
        Classification report string
    """
    if labels is None:
        labels = [f"Class_{i}" for i in range(len(np.unique(y_true)))]

    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0, digits=4)

    return report


def compute_confusion_matrix_metrics(cm: np.ndarray) -> Dict[str, float]:
    """
    Compute additional metrics from confusion matrix.

    Args:
        cm: Confusion matrix

    Returns:
        Dictionary with additional metrics
    """
    # True positives, false positives, false negatives for each class
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    # Additional metrics
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)  # Same as recall

    return {
        "specificity": specificity.tolist(),
        "sensitivity": sensitivity.tolist(),
        "true_positives": tp.tolist(),
        "false_positives": fp.tolist(),
        "false_negatives": fn.tolist(),
        "true_negatives": tn.tolist(),
    }
