import re
from typing import Any, Dict

from conflicts.core.base import BaseAgent
from conflicts.core.models import DocumentPair, EditorResult, ValidationResult


class ModeratorAgent(BaseAgent):
    """
    Moderator Agent validates if the modifications are acceptable and realistic.
    If valid, documents are saved to persistent storage.
    If invalid, they are returned to Editor Agent for re-modification.
    """

    def __init__(self, client, model, cfg, min_validation_score: int = 4):
        with open("prompts/moderator_agent_system.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        super().__init__("Moderator", client, model, cfg, system_prompt)
        self.min_score = min_validation_score

    def __call__(
        self, original_pair: DocumentPair, modified_docs: EditorResult, conflict_type: str
    ) -> ValidationResult:
        """
        Validate the modifications made to clinical documents

        Args:
            original_pair: Original document pair
            modified_docs: Modified documents from Editor Agent
            conflict_type: Type of conflict that was supposed to be created

        Returns:
            ValidationResult indicating whether modifications are acceptable
        """
        self.logger.info(f"Moderator Agent validating '{conflict_type}' conflict modifications")

        try:
            # Load prompt template from file and prepare validation prompt
            with open("prompts/moderator_agent.txt", "r", encoding="utf-8") as f:
                prompt_template = f.read().strip()

            prompt = prompt_template.format(
                context_document_1=self._truncate_document(original_pair.doc1_text),
                context_document_2=self._truncate_document(original_pair.doc2_text),
                conflict_1=modified_docs.change_info_1 or "No change info available",
                conflict_2=modified_docs.change_info_2 or "No change info available",
            )
            print(prompt)

            self.logger.debug(f"Sending validation prompt to API (length: {len(prompt)} chars)")

            # Call Groq API with low temperature for consistent validation
            response = self._execute_prompt(prompt, self.cfg.model.base_temperature)

            self.logger.debug(f"Received validation response from API: {response[:200]}...")

            # Parse response using new score-based format
            parsed_response = self._parse_score_response(response)

            # Extract score (1-5 scale) and reasoning from the new format
            score = parsed_response.get("score", 1)
            reasoning = parsed_response.get("reasoning", "No reasoning provided")

            is_valid = True

            if score < self.min_score:
                is_valid = False
                self.logger.info(
                    f"Validation score {score} below threshold \
                        {self.min_score}, marking as invalid"
                )

            result = ValidationResult(
                is_valid=is_valid,
                score=score,
                reasoning=reasoning,
            )

            self.logger.info("Moderator Agent completed validation")
            self.logger.info(
                f"Validation result: {'VALID' if result.is_valid else 'INVALID'} \
                    (Score: {result.score}/5)"
            )
            self.logger.info(f"Reasoning: {result.reasoning}")

            return result

        except Exception as e:
            self.logger.error(f"Moderator Agent processing failed: {e}")
            # Return a safe invalid result on error
            return ValidationResult(
                is_valid=False,
                score=1,
                reasoning=f"Validation failed due to error: {str(e)}",
            )

    def detailed_validation_check(
        self, original_pair: DocumentPair, modified_docs: EditorResult
    ) -> Dict[str, Any]:
        """
        Perform detailed technical validation checks on the modifications

        Args:
            original_pair: Original documents
            modified_docs: Modified documents

        Returns:
            Dictionary with detailed validation metrics
        """
        checks = {
            "documents_modified": {
                "doc1_changed": original_pair.doc1_text != modified_docs.modified_document1,
                "doc2_changed": original_pair.doc2_text != modified_docs.modified_document2,
                "any_changed": False,
            },
            "length_analysis": {
                "doc1_original_length": len(original_pair.doc1_text),
                "doc1_modified_length": len(modified_docs.modified_document1),
                "doc1_length_change": len(modified_docs.modified_document1)
                - len(original_pair.doc1_text),
                "doc2_original_length": len(original_pair.doc2_text),
                "doc2_modified_length": len(modified_docs.modified_document2),
                "doc2_length_change": len(modified_docs.modified_document2)
                - len(original_pair.doc2_text),
            },
            "content_analysis": {
                "changes_description_length": len(modified_docs.changes_made),
                "has_changes_description": len(modified_docs.changes_made.strip()) > 0,
            },
            "potential_issues": [],
        }

        # Calculate if any document was changed
        checks["documents_modified"]["any_changed"] = (
            checks["documents_modified"]["doc1_changed"]
            or checks["documents_modified"]["doc2_changed"]
        )

        # Identify potential issues
        if not checks["documents_modified"]["any_changed"]:
            checks["potential_issues"].append("No modifications detected in either document")

        if (
            abs(checks["length_analysis"]["doc1_length_change"])
            > len(original_pair.doc1_text) * 0.5
        ):
            checks["potential_issues"].append("Document 1 length changed by more than 50%")

        if (
            abs(checks["length_analysis"]["doc2_length_change"])
            > len(original_pair.doc2_text) * 0.5
        ):
            checks["potential_issues"].append("Document 2 length changed by more than 50%")

        if not checks["content_analysis"]["has_changes_description"]:
            checks["potential_issues"].append("No description of changes provided")

        return checks

    def _parse_score_response(self, response: str) -> dict:
        """
        Parse the new response format that expects reasoning + "Score: X"
        """
        try:
            # Find the score using regex
            score = 1
            if "score:" in response.lower():
                score_match = re.search(r"score:\s*(\d)", response, re.IGNORECASE)
                if score_match:
                    score = int(score_match.group(1))
                    score = max(1, min(5, score))

            return {
                "reasoning": response.strip(),
                "score": score,
            }

        except Exception as e:
            self.logger.error(f"Failed to parse score response: {e}")
            return {
                "reasoning": response.strip() if response else "Failed to parse response",
                "score": 1,
            }
