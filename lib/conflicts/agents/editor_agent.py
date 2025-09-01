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

        try:
            prompt = f"""input_prompt: "{conflict_instructions.modification_instructions}"
                    document_1: "{self._truncate_document(document_pair.doc1_text)}"
                    document_2: "{self._truncate_document(document_pair.doc2_text)}"
                    """

            self.logger.debug(f"Prompt length: {len(prompt)} chars")

            # Call API to get edit operations
            response = self._execute_prompt(prompt)

            self.logger.debug(f"Received modification response from API: {response[:200]}...")

            # Parse response using new document operations
            try:
                parsed_result = parse_response(
                    response, document_pair.doc1_text, document_pair.doc2_text
                )
            except ValueError as e:
                self.logger.error(f"Edit operation failed: {e}")
                raise

            # Validate that modifications were actually made
            if (
                parsed_result["modified_doc_1"].strip() == document_pair.doc1_text.strip()
                and parsed_result["modified_doc_2"].strip() == document_pair.doc2_text.strip()
            ):
                self.logger.warning("Editor Agent returned documents without modifications")
                raise ValueError("No modifications were applied to the documents")

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
