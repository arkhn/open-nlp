from base import BaseAgent
from document_operations import parse_response
from models import ConflictResult, DocumentPair, EditorResult


class EditorAgent(BaseAgent):
    """
    Editor Agent creates conflicts by editing/modifying documents based on
    the Doctor Agent's conflict type and instructions.
    """

    def __init__(self, client, model):
        with open("prompts/editor_agent_system.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        super().__init__("Editor", client, model, system_prompt)

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
        return self._try_modification_with_retries(document_pair, conflict_instructions)

    def _try_modification_with_retries(
        self,
        document_pair: DocumentPair,
        conflict_instructions: ConflictResult,
        max_retries: int = 5,
    ) -> EditorResult:
        """Try to modify documents with retries on failure"""
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
        response = self._execute_prompt(prompt, temperature=0.2)
        parsed_result = self._parse_and_validate_response(response, document_pair)
        return self._create_result(parsed_result, document_pair)

    def _build_prompt(
        self, document_pair: DocumentPair, conflict_instructions: ConflictResult
    ) -> str:
        """Build the prompt for modification"""
        prompt = f"""input_prompt: "{conflict_instructions.modification_instructions}"
                document_1: "{self._truncate_document(document_pair.doc1_text)}"
                document_2: "{self._truncate_document(document_pair.doc2_text)}"
                """
        self.logger.debug(f"Prompt length: {len(prompt)} chars")
        return prompt

    def _parse_and_validate_response(self, response: str, document_pair: DocumentPair) -> dict:
        """Parse response and validate modifications were made"""
        self.logger.debug(f"Received modification response from API: {response[:200]}...")

        parsed_result = parse_response(response, document_pair.doc1_text, document_pair.doc2_text)

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
        )

        self.logger.info("Editor Agent completed modifications")
        self.logger.info(f"Conflict type: {parsed_result['conflict_type']}")

        # Log document length changes
        orig_len1, orig_len2 = len(document_pair.doc1_text), len(document_pair.doc2_text)
        mod_len1, mod_len2 = len(result.modified_document1), len(result.modified_document2)

        self.logger.debug(f"Document 1 length: {orig_len1} -> {mod_len1} ({mod_len1-orig_len1:+d})")
        self.logger.debug(f"Document 2 length: {orig_len2} -> {mod_len2} ({mod_len2-orig_len2:+d})")

        return result
