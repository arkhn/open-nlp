import json
import logging
from typing import Any, Dict, Tuple

from conflicts.utils.utils import find_similar_text


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


def apply_edit_operation(
    document: str,
    operation: Dict[str, Any],
    min_text_length: int,
) -> Tuple[str, str]:
    """
    Apply edit operation to a document using text-based operations.

    Args:
        document: Original document content
        operation: Edit operation with keys: op, target_text, replacement_text (optional)

    Returns:
        Tuple of (modified document, change description)

    Raises:
        ValueError: If operation type is unknown or target_text not found
    """
    op_type = operation["op"]
    target_text = operation["target_text"]
    replacement_text = operation.get("replacement_text", "")

    # Check if target_text exists in document
    similar_text = find_similar_text(document, target_text)

    if not similar_text:
        # Additional validation: check for common LLM issues
        if target_text.endswith("...") or target_text.endswith("n..."):
            raise ValueError(f"Target text appears to be truncated: '{target_text[:50]}...'")
        if len(target_text.strip()) < min_text_length:
            raise ValueError(f"Target text too short to be reliable: '{target_text[:50]}...'")

        # Provide more helpful error message with suggestions
        error_msg = f"Target text not found in document: '{target_text[:100]}...'"
        raise ValueError(error_msg)

    if op_type == "delete":
        modified_doc = document.replace(target_text, "", 1)
        change_description = f"Deleted text: '{target_text[:200]}...'"
        return modified_doc, change_description
    elif op_type == "insert_after":
        modified_doc = document.replace(target_text, target_text + replacement_text, 1)
        change_description = (
            f"Inserted '{replacement_text[:200]}...' after '{target_text[:200]}...'"
        )
        return modified_doc, change_description
    elif op_type == "replace":
        modified_doc = document.replace(target_text, replacement_text, 1)
        change_description = f"Replaced '{target_text[:200]}...' with '{replacement_text[:200]}...'"
        return modified_doc, change_description
    else:
        raise ValueError(f"Unknown operation type: {op_type}")


def parse_response(
    response: str,
    original_doc_1: str,
    original_doc_2: str,
    min_text_length: int,
) -> Dict[str, Any]:
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
                modified_doc_1, change_description_1 = apply_edit_operation(
                    original_doc_1,
                    data["doc1"],
                    min_text_length,
                )
                modified_doc_2, change_description_2 = apply_edit_operation(
                    original_doc_2, data["doc2"], min_text_length
                )
            except ValueError as e:
                # Provide more helpful error message with suggestions
                error_msg = f"Failed to apply edit operations: {e}\n\n"
                error_msg += (
                    "This usually happens when the LLM references text that "
                    "doesn't exist in the documents.\n\n"
                )
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
                "change_info_1": change_description_1,
                "change_info_2": change_description_2,
            }
        else:
            raise ValueError(f"Response missing required fields: {required_fields}")

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse response as JSON: {e}")
        raise ValueError(f"Failed to parse response as JSON: {e}")
    except Exception as e:
        logging.error(f"Error parsing response: {e}")
        raise ValueError(f"Error parsing response: {e}")
