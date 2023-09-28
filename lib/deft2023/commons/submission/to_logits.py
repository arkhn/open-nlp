import numpy as np
import torch


def to_logits(predictions, threshold=0.5):
    """Converts predictions from logits to letters.

    Args:
        predictions: List of predictions.
        threshold: Threshold for the sigmoid function.

    Returns:
        List of predictions in letters.
    """
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    # check if all predicted probabilities are 0
    zero_indices = np.all(y_pred == 0, axis=1)
    if np.any(zero_indices):
        # set the index of the highest probability to 1
        idx = np.argmax(probs[zero_indices], axis=1)
        y_pred[zero_indices, idx] = 1

    return y_pred
