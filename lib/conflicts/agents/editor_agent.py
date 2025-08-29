from base import BaseAgent
from document_operations import (extract_text_by_category, parse_response,
                                 suggest_edit_operations)
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

    def _get_document_text_summary(self, document: str, max_sentences: int = 5) -> str:
        """
        Get a summary of available text in the document to help the LLM.

        Args:
            document: The document text
            max_sentences: Maximum number of sentences to include per category

        Returns:
            Summary string with available text segments organized by category
        """
        categorized_text = extract_text_by_category(document)

        summary = "Available text segments by category:\n"

        # Show most relevant categories first
        priority_categories = [
            "assessment",
            "vital_signs",
            "laboratory",
            "medication",
            "procedures",
            "symptoms",
            "general",
        ]

        for category in priority_categories:
            if category in categorized_text and categorized_text[category]:
                texts = categorized_text[category][:max_sentences]
                summary += f"\n{category.replace('_', ' ').title()}:\n"
                for i, text in enumerate(texts, 1):
                    summary += f"  {i}. {text[:150]}...\n"

        return summary

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
            # Prepare prompt with user instructions and documents
            doc1_summary = self._get_document_text_summary(document_pair.doc1_text)
            doc2_summary = self._get_document_text_summary(document_pair.doc2_text)

            prompt = f"""input_prompt: "{conflict_instructions.modification_instructions}"
                    document_1: "{self._truncate_document(document_pair.doc1_text)}"
                    document_2: "{self._truncate_document(document_pair.doc2_text)}"

                    IMPORTANT: Use only text that actually exists in the documents above.

                    Document 1 available text segments:
                    {doc1_summary}

                    Document 2 available text segments:
                    {doc2_summary}

                    Choose target_text from the available segments above. Do NOT invent or
                    generate fictional text.
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
                # If parsing fails due to text matching issues, provide helpful suggestions
                if "Target text not found" in str(e) or "fictional text" in str(e):
                    self.logger.error(f"Text matching failed: {e}")

                    # Provide suggestions for both documents
                    suggestions_1 = suggest_edit_operations(document_pair.doc1_text, "general")
                    suggestions_2 = suggest_edit_operations(document_pair.doc2_text, "general")

                    error_msg = f"Edit operation failed: {e}\n\n"
                    error_msg += "Available text segments in Document 1:\n"
                    for i, text in enumerate(suggestions_1["general"][:3], 1):
                        error_msg += f"  {i}. {text[:100]}...\n"

                    error_msg += "\nAvailable text segments in Document 2:\n"
                    for i, text in enumerate(suggestions_2["general"][:3], 1):
                        error_msg += f"  {i}. {text[:100]}...\n"

                    error_msg += (
                        "\nPlease ensure the LLM only references text that "
                        "actually exists in the documents."
                    )
                    raise ValueError(error_msg)
                else:
                    # Re-raise other parsing errors
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
