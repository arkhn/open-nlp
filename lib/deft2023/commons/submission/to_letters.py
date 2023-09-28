def to_letters(predictions, id2label):
    """Converts predictions from logits to letters.

    Args:
        predictions: List of predictions.
        id2label: Dictionary of id to label.

    Returns:
        List of predictions in letters.
    """
    predictions_classes = []
    for p in predictions:
        predictions_classes.append([id2label[i] for i, p in enumerate(p) if p == 1])

    return predictions_classes
