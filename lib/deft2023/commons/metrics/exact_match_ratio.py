import torch
import torchmetrics


def exact_match_ratio(y_pred, y_true):
    """Computes the exact match ratio of the predictions.

    Args:
        y: List of true labels.
        y_hat: List of predicted labels.

    Returns:
        Exact match ratio of the predictions.
    """
    return torchmetrics.functional.classification.multilabel_exact_match(
        torch.Tensor(y_pred),
        torch.tensor(y_true),
        num_labels=5,
    )
