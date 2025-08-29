import logging
import re
from typing import Any, Dict, List

from config import (MEDICAL_CATEGORIES, MIN_SENTENCE_LENGTH,
                    SENTENCE_SPLIT_PATTERN, SIMILARITY_THRESHOLD,
                    WORD_OVERLAP_THRESHOLD)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for better processing.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # Split on common sentence endings, but be careful with abbreviations
    sentences = re.split(SENTENCE_SPLIT_PATTERN, text)

    # Filter out very short or empty sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > MIN_SENTENCE_LENGTH]

    return sentences


def find_similar_text(document: str, target_text: str, similarity_threshold: float = None) -> str:
    """
    Find text in the document that is similar to the target text using fuzzy matching.

    Args:
        document: The document to search in
        target_text: The text to find similar matches for
        similarity_threshold: Minimum similarity score (0.0 to 1.0)

    Returns:
        The most similar text found, or empty string if no good match
    """
    if similarity_threshold is None:
        similarity_threshold = SIMILARITY_THRESHOLD

    try:
        from difflib import SequenceMatcher

        # Split document into sentences for better matching
        sentences = split_into_sentences(document)

        best_match = ""
        best_score = 0.0

        # First try exact substring matching with case insensitivity
        target_lower = target_text.lower()
        for sentence in sentences:
            if target_lower in sentence.lower():
                return sentence

        # If no exact substring match, try fuzzy matching
        for sentence in sentences:
            # Calculate similarity between target and this sentence
            similarity = SequenceMatcher(None, target_lower, sentence.lower()).ratio()

            if similarity > best_score and similarity >= similarity_threshold:
                best_score = similarity
                best_match = sentence

        # If still no good match, try word-based matching
        if not best_match:
            target_words = set(target_lower.split())
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                word_overlap = len(target_words.intersection(sentence_words))
                if word_overlap >= WORD_OVERLAP_THRESHOLD:
                    return sentence

        return best_match

    except ImportError:
        # Fallback if difflib is not available
        logging.warning("difflib not available, skipping fuzzy matching")
        return ""
    except Exception as e:
        logging.warning(f"Error in fuzzy matching: {e}")
        return ""


def categorize_text_by_medical_domain(document: str) -> Dict[str, List[str]]:
    """
    Extract text from document organized by medical categories.

    Args:
        document: The document to analyze

    Returns:
        Dictionary with text organized by medical categories
    """
    sentences = split_into_sentences(document)
    categorized_text = {category: [] for category in MEDICAL_CATEGORIES.keys()}

    for sentence in sentences:
        sentence_lower = sentence.lower()

        for category, keywords in MEDICAL_CATEGORIES.items():
            if any(keyword in sentence_lower for keyword in keywords):
                categorized_text[category].append(sentence)

    # Add general category for sentences that don't fit specific categories
    categorized_text["general"] = []
    for sentence in sentences:
        if not any(sentence in " ".join(texts) for texts in categorized_text.values() if texts):
            categorized_text["general"].append(sentence)

    return categorized_text


def find_suitable_text_for_editing(
    document: str, target_keywords: List[str] = None, min_length: int = None
) -> List[str]:
    """
    Find suitable text segments in a document that could be used for editing operations.

    Args:
        document: The document to search in
        target_keywords: Optional list of keywords to prioritize
        min_length: Minimum length of text segments to return

    Returns:
        List of suitable text segments
    """
    if min_length is None:
        from config import MIN_TEXT_SEGMENT_LENGTH

        min_length = MIN_TEXT_SEGMENT_LENGTH

    sentences = split_into_sentences(document)

    if target_keywords:
        # Score sentences based on keyword relevance
        scored_sentences = []
        for sentence in sentences:
            if len(sentence) >= min_length:
                score = sum(1 for keyword in target_keywords if keyword.lower() in sentence.lower())
                if score > 0:
                    scored_sentences.append((score, sentence))

        # Sort by relevance score and return top candidates
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return [sentence for _, sentence in scored_sentences[:3]]
    else:
        # Return sentences that meet length requirements
        return [s for s in sentences if len(s) >= min_length]


def suggest_edit_operations(document: str, conflict_type: str) -> Dict[str, Any]:
    """
    Suggest edit operations based on document content and conflict type.

    Args:
        document: The document to analyze
        conflict_type: Type of conflict to create

    Returns:
        Dictionary with suggested edit operations
    """
    from config import (EDIT_SUGGESTION_KEYWORDS, MAX_SUGGESTIONS_PER_CATEGORY,
                        MIN_GENERAL_TEXT_LENGTH)

    suggestions = {
        "medical_condition": [],
        "medication": [],
        "vital_signs": [],
        "lab_values": [],
        "procedures": [],
        "general": [],
    }

    # Find suitable text for each category
    for category, keywords in EDIT_SUGGESTION_KEYWORDS.items():
        suitable_text = find_suitable_text_for_editing(document, keywords, min_length=30)
        suggestions[category] = suitable_text[:MAX_SUGGESTIONS_PER_CATEGORY]

    # General suggestions (longer sentences that might be good anchors)
    suggestions["general"] = find_suitable_text_for_editing(
        document, min_length=MIN_GENERAL_TEXT_LENGTH
    )[:3]

    return suggestions
