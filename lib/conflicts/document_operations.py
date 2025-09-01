import json
import logging
from typing import Any, Dict, List

from config import MIN_TARGET_TEXT_LENGTH
from utils import (
    categorize_text_by_medical_domain,
    find_similar_text,
    suggest_edit_operations,
)


def _extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract and parse JSON data from an LLM response.

    Args:
        response: The LLM's response text

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If JSON cannot be extracted or parsed
    """
    # Try to find JSON between first { and last }
    json_start = response.find("{")
    json_end = response.rfind("}")

    if json_start >= 0 and json_end > json_start:
        json_str = response[json_start : json_end + 1]
        try:
            data = json.loads(json_str)
            logging.info("Successfully parsed response as JSON")
            return data
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code blocks
            if "```json" in response:
                parts = response.split("```json")
                if len(parts) > 1:
                    json_part = parts[1].split("```")[0].strip()
                    data = json.loads(json_part)
                    logging.info("Successfully parsed JSON from markdown code block")
                    return data
                else:
                    raise ValueError(f"Could not extract JSON from markdown: {response}")
            else:
                raise ValueError(f"Invalid JSON format: {json_str}")
    else:
        # Fallback to parsing the entire response
        try:
            data = json.loads(response)
            logging.info("Successfully parsed entire response as JSON")
            return data
        except json.JSONDecodeError as e:
            logging.warning(f"Response is not in JSON format: {e}")
            raise ValueError(f"Invalid JSON format in response: {e}")


def apply_edit_operation(document: str, operation: Dict[str, Any]) -> str:
    """
    Apply edit operation to a document using text-based operations.

    Args:
        document: Original document content
        operation: Edit operation with keys: op, target_text, replacement_text (optional)

    Returns:
        Modified document

    Raises:
        ValueError: If operation type is unknown or target_text not found
    """
    op_type = operation["op"]
    target_text = operation["target_text"]
    replacement_text = operation.get("replacement_text", "")

    # Check if target_text exists in document
    if target_text not in document:
        # Try to find similar text using fuzzy matching
        similar_text = find_similar_text(document, target_text)

        if similar_text:
            logging.warning(
                f"Exact target text not found, using similar text: "
                f"'{target_text[:50]}...' -> '{similar_text[:50]}...'"
            )
            target_text = similar_text
        else:
            # Additional validation: check for common LLM issues
            if target_text.endswith("...") or target_text.endswith("n..."):
                raise ValueError(f"Target text appears to be truncated: '{target_text[:50]}...'")
            if len(target_text.strip()) < MIN_TARGET_TEXT_LENGTH:
                raise ValueError(f"Target text too short to be reliable: '{target_text[:50]}...'")

            # Provide more helpful error message with suggestions
            error_msg = f"Target text not found in document: '{target_text[:100]}...'"
            raise ValueError(error_msg)

    if op_type == "delete":
        return document.replace(target_text, "", 1)
    elif op_type == "insert_after":
        return document.replace(target_text, target_text + replacement_text, 1)
    elif op_type == "replace":
        return document.replace(target_text, replacement_text, 1)
    else:
        raise ValueError(f"Unknown operation type: {op_type}")


def parse_response(response: str, original_doc_1: str, original_doc_2: str) -> Dict[str, Any]:
    """
    Parse the LLM's response to extract operations and apply them to documents.

    Args:
        response: The LLM's response
        original_doc_1: Original content of first document
        original_doc_2: Original content of second document

    Returns:
        Dictionary with modified documents and conflict type

    Raises:
        ValueError: If response cannot be parsed
    """
    try:
        # Log the first 200 chars of response for debugging
        logging.debug(f"Response preview: {response[:200]}...")

        # Extract JSON from response
        data = _extract_json_from_response(response)

        # Check if we have the expected edit operations format
        required_fields = ["doc1", "doc2", "conflict_type"]
        if all(field in data for field in required_fields):
            logging.info("Found edit operations format, applying operations to documents")

            # Apply edit operations to documents
            try:
                modified_doc_1 = apply_edit_operation(original_doc_1, data["doc1"])
                modified_doc_2 = apply_edit_operation(original_doc_2, data["doc2"])
            except ValueError as e:
                # Provide more helpful error message with suggestions
                error_msg = f"Failed to apply edit operations: {e}\n\n"
                error_msg += (
                    "This usually happens when the LLM references text that "
                    "doesn't exist in the documents.\n\n"
                )

                # Add suggestions for document 1
                if "doc1" in data and "target_text" in data["doc1"]:
                    suggestions_1 = suggest_edit_operations(
                        original_doc_1, data.get("conflict_type", "general")
                    )
                    error_msg += "Document 1 suggestions for editing:\n"
                    for category, texts in suggestions_1.items():
                        if texts:
                            error_msg += f"  {category}: {texts[0][:100]}...\n"

                # Add suggestions for document 2
                if "doc2" in data and "target_text" in data["doc2"]:
                    suggestions_2 = suggest_edit_operations(
                        original_doc_2, data.get("conflict_type", "general")
                    )
                    error_msg += "Document 2 suggestions for editing:\n"
                    for category, texts in suggestions_2.items():
                        if texts:
                            error_msg += f"  {category}: {texts[0][:100]}...\n"

                error_msg += (
                    "\nPlease ensure the LLM only references text that actually "
                    "exists in the original documents."
                )
                raise ValueError(error_msg)

            conflict_type = data["conflict_type"]

            return {
                "modified_doc_1": modified_doc_1,
                "modified_doc_2": modified_doc_2,
                "conflict_type": conflict_type,
            }
        else:
            raise ValueError(f"Response missing required fields: {required_fields}")

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse response as JSON: {e}")
        raise ValueError(f"Failed to parse response as JSON: {e}")
    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        raise ValueError(f"Error parsing response: {e}")


def validate_edit_operation(document: str, operation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate an edit operation before applying it.

    Args:
        document: The document to validate against
        operation: The edit operation to validate

    Returns:
        Dictionary with validation results and suggestions
    """
    validation_result = {"is_valid": True, "issues": [], "suggestions": [], "similar_text": None}

    op_type = operation.get("op")
    target_text = operation.get("target_text", "")
    replacement_text = operation.get("replacement_text", "")

    # Check operation type
    if op_type not in ["delete", "insert_after", "replace"]:
        validation_result["is_valid"] = False
        validation_result["issues"].append(f"Invalid operation type: {op_type}")

    # Check if target_text exists
    if target_text not in document:
        validation_result["is_valid"] = False
        validation_result["issues"].append("Target text not found in document")

        # Try to find similar text
        similar_text = find_similar_text(document, target_text)
        if similar_text:
            validation_result["similar_text"] = similar_text
            validation_result["suggestions"].append(
                f"Consider using similar text: '{similar_text[:100]}...'"
            )

        # Provide general suggestions
        suggestions = suggest_edit_operations(document, "general")
        if suggestions["general"]:
            validation_result["suggestions"].append("Available text segments for editing:")
            for i, text in enumerate(suggestions["general"][:3], 1):
                validation_result["suggestions"].append(f"  {i}. {text[:100]}...")

    # Check text quality
    if len(target_text.strip()) < MIN_TARGET_TEXT_LENGTH:
        validation_result["is_valid"] = False
        validation_result["issues"].append(
            f"Target text too short (less than {MIN_TARGET_TEXT_LENGTH} characters)"
        )

    if any(target_text.endswith(indicator) for indicator in ["...", "n..."]):
        validation_result["is_valid"] = False
        validation_result["issues"].append("Target text appears to be truncated")

    # Check replacement text for replace operations
    if op_type == "replace" and not replacement_text:
        validation_result["is_valid"] = False
        validation_result["issues"].append("Replace operation requires replacement_text")

    return validation_result


def extract_text_by_category(document: str) -> Dict[str, List[str]]:
    """
    Extract text from document organized by medical categories.

    Args:
        document: The document to analyze

    Returns:
        Dictionary with text organized by medical categories
    """
    return categorize_text_by_medical_domain(document)
