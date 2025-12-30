from thefuzz import fuzz


def find_similar_text(document: str, target_text: str, similarity_threshold: float = 60) -> str:
    """
    Find text in the document that is similar to the target text using sliding window approach.

    Args:
        document: The document to search in
        target_text: The text to find similar matches for
        similarity_threshold: Minimum similarity score (0.0 to 100.0)

    Returns:
        The most similar text excerpt found, or empty string if no good match
    """
    target_lower = target_text.lower()
    target_len = len(target_text)
    best_match = ""
    best_score = 0.0

    # Use sliding window with target text length
    for i in range(len(document) - target_len + 1):
        window = document[i : i + target_len]
        window_lower = window.lower()

        similarity = max(
            fuzz.ratio(target_lower, window_lower), fuzz.partial_ratio(target_lower, window_lower)
        )

        if similarity > best_score:
            best_score = similarity
            best_match = window if similarity >= similarity_threshold else best_match

    return best_match
