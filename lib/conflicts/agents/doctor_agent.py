import time
from typing import Any, Dict

from base import (BaseAgent, ConflictResult, DocumentPair,
                  format_conflict_types_for_prompt)
from config import CONFLICT_TYPES, DOCTOR_AGENT_PROMPT_FILE
from prompt_loader import load_prompt


class DoctorAgent(BaseAgent):
    """
    Doctor Agent analyzes two clinical documents and decides which type of
    clinical conflict should be introduced between them.
    """

    def __init__(self):
        super().__init__("Doctor")

    def process(self, document_pair: DocumentPair) -> ConflictResult:
        """
        Analyze documents and determine the best conflict type to introduce

        Args:
            document_pair: Pair of clinical documents to analyze

        Returns:
            ConflictResult containing the chosen conflict type and instructions
        """
        start_time = time.time()

        self.logger.info(
            f"Doctor Agent analyzing document pair: \
                {document_pair.doc1_id} & {document_pair.doc2_id}"
        )

        try:
            # Load prompt template from file
            prompt_template = load_prompt(DOCTOR_AGENT_PROMPT_FILE)

            # Prepare prompt with conflict types and documents
            conflict_types_formatted = format_conflict_types_for_prompt(CONFLICT_TYPES)

            prompt = prompt_template.format(
                conflict_types=conflict_types_formatted,
                document1=self._truncate_document(document_pair.doc1_text),
                document2=self._truncate_document(document_pair.doc2_text),
            )

            self.logger.debug(f"Sending prompt to API (length: {len(prompt)} chars)")

            # Call Groq API
            response = self.client.call_api(prompt, temperature=0.3)

            self.logger.debug(f"Received response from API: {response[:200]}...")

            # Parse response
            parsed_response = self._parse_json_response(response)

            # Validate required fields
            required_fields = ["conflict_type", "reasoning", "modification_instructions"]
            for field in required_fields:
                if field not in parsed_response:
                    raise ValueError(f"Missing required field '{field}' in Doctor Agent response")

            # Validate conflict type exists
            if parsed_response["conflict_type"] not in CONFLICT_TYPES:
                self.logger.warning(
                    f"Unknown conflict type '{parsed_response['conflict_type']}', \
                     defaulting to 'opposition'"
                )
                parsed_response["conflict_type"] = "opposition"

            result = ConflictResult(
                conflict_type=parsed_response["conflict_type"],
                reasoning=parsed_response["reasoning"],
                modification_instructions=parsed_response["modification_instructions"],
            )

            processing_time = time.time() - start_time

            self.logger.info(f"Doctor Agent completed analysis in {processing_time:.2f}s")
            self.logger.info(f"Selected conflict type: {result.conflict_type}")
            self.logger.debug(f"Reasoning: {result.reasoning}")

            return result

        except Exception as e:
            self.logger.error(f"Doctor Agent processing failed: {e}")
            raise

    def _truncate_document(self, text: str, max_length: int = 2000) -> str:
        """
        Truncate document text to fit within prompt limits while preserving meaning

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

    def get_conflict_type_info(self, conflict_type: str) -> Dict[str, Any]:
        """
        Get information about a specific conflict type

        Args:
            conflict_type: The conflict type key

        Returns:
            Dictionary with conflict type information
        """
        if conflict_type not in CONFLICT_TYPES:
            raise ValueError(f"Unknown conflict type: {conflict_type}")

        conflict_info = CONFLICT_TYPES[conflict_type]

        return {
            "name": conflict_info.name,
            "description": conflict_info.description,
            "examples": conflict_info.examples,
            "key": conflict_type,
        }

    def list_all_conflict_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available conflict types

        Returns:
            Dictionary mapping conflict type keys to their information
        """
        return {
            key: {
                "name": conflict_type.name,
                "description": conflict_type.description,
                "examples": conflict_type.examples,
            }
            for key, conflict_type in CONFLICT_TYPES.items()
        }
