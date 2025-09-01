from thefuzz import fuzz

MIN_SENTENCE_LENGTH = 10
MEDICAL_CATEGORIES = {
    "assessment": ["assessment", "diagnosis", "impression", "findings", "clinical picture"],
    "vital_signs": [
        "vital signs",
        "blood pressure",
        "heart rate",
        "pulse",
        "temperature",
        "respiratory rate",
        "oxygen saturation",
        "sp02",
        "bp",
        "hr",
        "temp",
    ],
    "laboratory": [
        "lab",
        "laboratory",
        "test",
        "result",
        "value",
        "level",
        "ng/ml",
        "mg/dl",
        "mmol/l",
        "units",
        "reference range",
    ],
    "medication": [
        "medication",
        "drug",
        "prescription",
        "dose",
        "mg",
        "tablet",
        "capsule",
        "injection",
        "iv",
        "oral",
        "subcutaneous",
    ],
    "procedures": [
        "procedure",
        "surgery",
        "operation",
        "intervention",
        "treatment",
        "catheterization",
        "intubation",
        "ventilation",
    ],
    "symptoms": [
        "symptoms",
        "complaints",
        "pain",
        "shortness of breath",
        "dyspnea",
        "nausea",
        "vomiting",
        "dizziness",
        "weakness",
    ],
}

EDIT_SUGGESTION_KEYWORDS = {
    "medical_condition": ["diagnosis", "condition", "disease", "syndrome", "assessment"],
    "medication": ["medication", "drug", "prescription", "dose", "mg", "tablet"],
    "vital_signs": ["blood pressure", "heart rate", "temperature", "pulse", "respiratory"],
    "lab_values": ["lab", "laboratory", "test", "result", "value", "level"],
    "procedures": ["procedure", "surgery", "operation", "intervention", "treatment"],
}

# Text processing constants
MAX_SUGGESTIONS_PER_CATEGORY = 3
MIN_TEXT_SEGMENT_LENGTH = 30
MIN_GENERAL_TEXT_LENGTH = 50


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
