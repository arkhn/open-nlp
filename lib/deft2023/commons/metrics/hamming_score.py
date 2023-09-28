def hamming_score(y_pred, y_true):
    """Computes the hamming score of the predictions.


    Args:
        y_pred: List of predicted labels.
        y_true: List of true labels.

    Returns:
        Hamming score of the predictions.
    """

    list_response = ["a", "b", "c", "d", "e"]
    assert y_true.shape == y_pred.shape, "Input arrays must have the same shape"
    hamming_scores = []
    for ref, pred in zip(y_true, y_pred):
        ref = [
            list_response[idx_response]
            for idx_response, response in enumerate(ref)
            if response == 1
        ]
        pred = [
            list_response[idx_response]
            for idx_response, response in enumerate(pred)
            if response == 1
        ]
        corrects = [True for p in pred if p in ref]
        corrects = sum(corrects)
        total_refs = len(list(set(pred + ref)))
        hamming_scores.append(corrects / total_refs)

    return sum(hamming_scores) / len(hamming_scores)
