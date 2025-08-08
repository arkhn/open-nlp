import time
from typing import Any, Dict

from base import BaseAgent, ConflictResult, DocumentPair, EditorResult
from config import CONFLICT_TYPES, EDITOR_AGENT_PROMPT_FILE
from prompt_loader import load_prompt


class EditorAgent(BaseAgent):
    """
    Editor Agent creates conflicts by editing/modifying documents based on
    the Doctor Agent's conflict type and instructions.
    """

    def __init__(self):
        super().__init__("Editor")

    def process(
        self, document_pair: DocumentPair, conflict_instructions: ConflictResult
    ) -> EditorResult:
        """
        Modify documents to introduce the specified conflict

        Args:
            document_pair: Original pair of clinical documents
            conflict_instructions: Instructions from Doctor Agent about what conflict to create

        Returns:
            EditorResult containing modified documents and summary of changes
        """
        start_time = time.time()

        self.logger.info(
            f"Editor Agent modifying documents to create \
                '{conflict_instructions.conflict_type}' conflict"
        )

        try:
            # Get conflict type information
            if conflict_instructions.conflict_type not in CONFLICT_TYPES:
                raise ValueError(f"Unknown conflict type: {conflict_instructions.conflict_type}")

            conflict_type_info = CONFLICT_TYPES[conflict_instructions.conflict_type]

            # Load prompt template from file and prepare with documents
            prompt_template = load_prompt(EDITOR_AGENT_PROMPT_FILE)

            prompt = prompt_template.format(
                conflict_type=conflict_type_info.name,
                conflict_description=conflict_type_info.description,
                modification_instructions=conflict_instructions.modification_instructions,
                document1=self._truncate_document(document_pair.doc1_text),
                document2=self._truncate_document(document_pair.doc2_text),
            )

            self.logger.debug(f"Sending modification prompt to API (length: {len(prompt)} chars)")

            # Call Groq API with slightly higher temperature for creative editing
            response = self.client.call_api(prompt, temperature=0.4, max_tokens=6000)

            self.logger.debug(f"Received modification response from API: {response[:200]}...")

            # Parse response
            parsed_response = self._parse_json_response(response)

            # Validate required fields
            required_fields = ["modified_document1", "modified_document2", "changes_made"]
            for field in required_fields:
                if field not in parsed_response:
                    raise ValueError(f"Missing required field '{field}' in Editor Agent response")

            # Validate that modifications were actually made
            if (
                parsed_response["modified_document1"].strip() == document_pair.doc1_text.strip()
                and parsed_response["modified_document2"].strip() == document_pair.doc2_text.strip()
            ):
                self.logger.warning(
                    "Editor Agent returned documents without modifications, retrying..."
                )
                return self._retry_with_emphasis(document_pair, conflict_instructions)

            result = EditorResult(
                modified_document1=parsed_response["modified_document1"],
                modified_document2=parsed_response["modified_document2"],
                changes_made=parsed_response["changes_made"],
            )

            processing_time = time.time() - start_time

            self.logger.info(f"Editor Agent completed modifications in {processing_time:.2f}s")
            self.logger.info(f"Changes made: {result.changes_made[:100]}...")

            # Log document length changes
            orig_len1, orig_len2 = len(document_pair.doc1_text), len(document_pair.doc2_text)
            mod_len1, mod_len2 = len(result.modified_document1), len(result.modified_document2)

            self.logger.debug(
                f"Document 1 length: {orig_len1} -> {mod_len1} ({mod_len1-orig_len1:+d})"
            )
            self.logger.debug(
                f"Document 2 length: {orig_len2} -> {mod_len2} ({mod_len2-orig_len2:+d})"
            )

            return result

        except Exception as e:
            self.logger.error(f"Editor Agent processing failed: {e}")
            raise

    def _retry_with_emphasis(
        self, document_pair: DocumentPair, conflict_instructions: ConflictResult
    ) -> EditorResult:
        """
        Retry editing with more emphasis on making changes

        Args:
            document_pair: Original documents
            conflict_instructions: Conflict instructions

        Returns:
            EditorResult with modifications
        """
        self.logger.info("Retrying with emphasis on making actual modifications")

        conflict_type_info = CONFLICT_TYPES[conflict_instructions.conflict_type]

        # Load prompt template and add emphasis
        prompt_template = load_prompt(EDITOR_AGENT_PROMPT_FILE)

        emphasized_prompt = (
            prompt_template.format(
                conflict_type=conflict_type_info.name,
                conflict_description=conflict_type_info.description,
                modification_instructions=f"IMPORTANT: You MUST make clear, \
                    noticeable changes to create the conflict. \
                        {conflict_instructions.modification_instructions}",
                document1=self._truncate_document(document_pair.doc1_text),
                document2=self._truncate_document(document_pair.doc2_text),
            )
            + "\n\nREMEMBER: The original documents must be MODIFIED to create clear conflicts. \
                Simply returning the original text is not acceptable."
        )

        response = self.client.call_api(emphasized_prompt, temperature=0.6, max_tokens=6000)
        parsed_response = self._parse_json_response(response)

        return EditorResult(
            modified_document1=parsed_response.get("modified_document1", document_pair.doc1_text),
            modified_document2=parsed_response.get("modified_document2", document_pair.doc2_text),
            changes_made=parsed_response.get(
                "changes_made", "Retry attempt - modifications emphasized"
            ),
        )

    def _truncate_document(self, text: str, max_length: int = 3000) -> str:
        """
        Truncate document text to fit within prompt limits while preserving meaning
        Editor agent gets more space since it needs to return full documents

        Args:
            text: Full document text
            max_length: Maximum character length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        # Try to cut at sentence boundaries
        truncated = text[:max_length]
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")

        # Use the latest sentence or line boundary
        cut_point = max(last_period, last_newline)

        if cut_point > max_length * 0.8:  # Only use if we don't lose too much content
            return text[: cut_point + 1]
        else:
            return text[:max_length] + "..."

    def validate_modifications(
        self, original_pair: DocumentPair, modified_result: EditorResult
    ) -> Dict[str, Any]:
        """
        Perform basic validation on the modifications made

        Args:
            original_pair: Original documents
            modified_result: Modified documents

        Returns:
            Dictionary with validation results
        """
        validation = {
            "has_changes_doc1": original_pair.doc1_text != modified_result.modified_document1,
            "has_changes_doc2": original_pair.doc2_text != modified_result.modified_document2,
            "length_change_doc1": len(modified_result.modified_document1)
            - len(original_pair.doc1_text),
            "length_change_doc2": len(modified_result.modified_document2)
            - len(original_pair.doc2_text),
            "has_any_changes": False,
            "changes_summary": modified_result.changes_made,
        }

        validation["has_any_changes"] = (
            validation["has_changes_doc1"] or validation["has_changes_doc2"]
        )

        return validation
