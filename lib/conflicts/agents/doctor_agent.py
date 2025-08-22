from typing import Dict, Any

from base import BaseAgent
from models import DocumentPair, ConflictResult
from config import CONFLICT_TYPES


def format_conflict_types_for_prompt(conflict_types: Dict) -> str:
    """Format conflict types dictionary for use in prompts"""
    formatted_types = []
    for key, conflict_type in conflict_types.items():
        examples_str = "\n  ".join([f"- {example}" for example in conflict_type.examples])
        formatted_types.append(
            f"""{key}: {conflict_type.name}
            Description: {conflict_type.description}
            Examples:
            {examples_str}
            """
        )
    return "\n".join(formatted_types)


class DoctorAgent(BaseAgent):
    """
    Doctor Agent analyzes two clinical documents and decides which type of
    clinical conflict should be introduced between them.
    """

    def __init__(self, client, model):
        with open("prompts/doctor_agent_system.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        super().__init__("Doctor", client, model, system_prompt)

    def __call__(self, document_pair: DocumentPair) -> ConflictResult:
        """
        Analyze documents and determine the best conflict type to introduce

        Args:
            document_pair: Pair of clinical documents to analyze

        Returns:
            ConflictResult containing the chosen conflict type and instructions
        """
        self.logger.info(
            f"Doctor Agent analyzing document pair: \
                {document_pair.doc1_id} & {document_pair.doc2_id}"
        )

        try:
            # Load prompt template from file
            with open("prompts/doctor_agent.txt", "r", encoding="utf-8") as f:
                prompt_template = f.read().strip()

            # Prepare prompt with conflict types and documents
            conflict_types_formatted = format_conflict_types_for_prompt(CONFLICT_TYPES)

            prompt = prompt_template.format(
                conflict_types=conflict_types_formatted,
                document1=self._truncate_document(document_pair.doc1_text),
                document2=self._truncate_document(document_pair.doc2_text),
            )

            self.logger.debug(f"Sending prompt to API (length: {len(prompt)} chars)")

            # Call Groq API
            response = self._execute_prompt(prompt)

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

            self.logger.info("Doctor Agent completed analysis")
            self.logger.info(f"Selected conflict type: {result.conflict_type}")
            self.logger.debug(f"Reasoning: {result.reasoning}")

            return result

        except Exception as e:
            self.logger.error(f"Doctor Agent processing failed: {e}")
            raise

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
