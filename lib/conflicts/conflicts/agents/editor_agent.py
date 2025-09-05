from pathlib import Path

from ..core.base import BaseAgent
from ..core.document_operations import parse_response
from ..core.models import ConflictResult, DocumentPair, EditorResult

prompts_dir = Path(__file__).parent.parent.parent / "prompts"
EDITOR_SYSTEM_PROMPT_PATH = prompts_dir / "editor_agent_system.txt"


class EditorAgent(BaseAgent):
    """
    Editor Agent creates conflicts by editing/modifying documents based on
    the Doctor Agent's conflict type and instructions.
    """

    def __init__(self, client, model, cfg):
        with open(EDITOR_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        super().__init__("Editor", client, model, cfg, prompt)
        self.min_text_length = cfg.editor.min_text_length

    def __call__(
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
        self.logger.info(f"Creating '{conflict_instructions.conflict_type}' conflict")

        # Try modification with retries on failure
        max_retries = self.cfg.editor.max_retries
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempt {attempt + 1}/{max_retries}")
                return self._perform_modification(document_pair, conflict_instructions)
            except ValueError as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"All {max_retries} attempts failed")
                    raise
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")

    def _perform_modification(
        self, document_pair: DocumentPair, conflict_instructions: ConflictResult
    ) -> EditorResult:
        """Perform a single modification attempt"""
        prompt = self._build_prompt(document_pair, conflict_instructions)
        response = self._execute_prompt(prompt, self.cfg.model.editor_temperature)
        parsed_result = self._parse_and_validate_response(response, document_pair)
        return self._create_result(parsed_result, document_pair)

    def _build_prompt(
        self, document_pair: DocumentPair, conflict_instructions: ConflictResult
    ) -> str:
        """Build the prompt for modification"""
        # Extract specific propositions for each document
        target_propositions_doc1 = self._extract_target_propositions(
            conflict_instructions.proposition_conflicts, "doc1_proposition"
        )
        target_propositions_doc2 = self._extract_target_propositions(
            conflict_instructions.proposition_conflicts, "doc2_proposition"
        )

        editor_instructions_str = (
            "\n".join(conflict_instructions.editor_instructions)
            if conflict_instructions.editor_instructions
            else "No specific instructions provided"
        )

        prompt = self.system_prompt.format(
            input_prompt=conflict_instructions.modification_instructions,
            document_1=self._truncate_document(document_pair.doc1_text),
            document_2=self._truncate_document(document_pair.doc2_text),
            conflict_type=conflict_instructions.conflict_type,
            target_propositions_doc1=target_propositions_doc1,
            target_propositions_doc2=target_propositions_doc2,
            editor_instructions=editor_instructions_str,
        )
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        return prompt

    def _extract_target_propositions(
        self, proposition_conflicts: list[dict], doc_field: str
    ) -> str:
        """Extract specific propositions from a document that need modification

        Args:
            proposition_conflicts: List of proposition conflict dictionaries
            doc_field: Field name to extract ('doc1_proposition' or 'doc2_proposition')

        Returns:
            Formatted string of propositions for the specified document
        """
        if not proposition_conflicts:
            return "No specific propositions identified"

        propositions = [conflict.get(doc_field, "N/A") for conflict in proposition_conflicts]
        return "\n".join([f"- {prop}" for prop in propositions if prop != "N/A"])

    def _parse_and_validate_response(self, response: str, document_pair: DocumentPair) -> dict:
        """Parse response and validate modifications were made"""
        self.logger.debug(f"Received modification response from API: {response[:200]}...")

        try:
            parsed_result = parse_response(
                response,
                document_pair.doc1_text,
                document_pair.doc2_text,
                self.min_text_length,
            )
        except Exception as e:
            self.logger.error(f"Failed to parse response: {e}")
            self.logger.error(f"Response was: {response}")
            raise

        if (
            parsed_result["modified_doc_1"].strip() == document_pair.doc1_text.strip()
            and parsed_result["modified_doc_2"].strip() == document_pair.doc2_text.strip()
        ):
            raise ValueError("No modifications were applied to the documents")

        return parsed_result

    def _create_result(self, parsed_result: dict, document_pair: DocumentPair) -> EditorResult:
        """Create and log the final result"""
        result = EditorResult(
            modified_document1=parsed_result["modified_doc_1"],
            modified_document2=parsed_result["modified_doc_2"],
            changes_made=f"Applied {parsed_result['conflict_type']} conflict modifications",
            change_info_1=parsed_result.get("change_info_1"),
            change_info_2=parsed_result.get("change_info_2"),
            original_excerpt_1=parsed_result.get("original_excerpt_1"),
            modified_excerpt_1=parsed_result.get("modified_excerpt_1"),
            original_excerpt_2=parsed_result.get("original_excerpt_2"),
            modified_excerpt_2=parsed_result.get("modified_excerpt_2"),
        )

        self.logger.info("Editor Agent completed modifications")
        self.logger.info(f"Conflict type: {parsed_result['conflict_type']}")

        # Log document length changes
        orig_len1, orig_len2 = len(document_pair.doc1_text), len(document_pair.doc2_text)
        mod_len1, mod_len2 = len(result.modified_document1), len(result.modified_document2)

        self.logger.debug(f"Document 1 length: {orig_len1} -> {mod_len1} ({mod_len1-orig_len1:+d})")
        self.logger.debug(f"Document 2 length: {orig_len2} -> {mod_len2} ({mod_len2-orig_len2:+d})")

        return result
