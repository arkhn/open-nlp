from typing import Any, Dict

from base import BaseAgent
from config import CONFLICT_TYPES
from models import DocumentPair, EditorResult, ValidationResult


class ModeratorAgent(BaseAgent):
    """
    Moderator Agent validates if the modifications are acceptable and realistic.
    If valid, documents are saved to persistent storage.
    If invalid, they are returned to Editor Agent for re-modification.
    """

    def __init__(self, client, model, min_validation_score: int = 70):
        with open("prompts/moderator_agent_system.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        super().__init__("Moderator", client, model, system_prompt)
        self.min_validation_score = min_validation_score

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
            # Get conflict type information for context
            if conflict_type not in CONFLICT_TYPES:
                self.logger.warning(f"Unknown conflict type '{conflict_type}' for validation")
                conflict_name = "Unknown Conflict Type"
            else:
                conflict_name = CONFLICT_TYPES[conflict_type].name

            # Load prompt template from file and prepare validation prompt
            with open("prompts/moderator_agent.txt", "r", encoding="utf-8") as f:
                prompt_template = f.read().strip()

            prompt = prompt_template.format(
                original_doc1=self._truncate_document(original_pair.doc1_text),
                original_doc2=self._truncate_document(original_pair.doc2_text),
                modified_doc1=self._truncate_document(modified_docs.modified_document1),
                modified_doc2=self._truncate_document(modified_docs.modified_document2),
                conflict_type=conflict_name,
                changes_made=modified_docs.changes_made,
            )

            self.logger.debug(f"Sending validation prompt to API (length: {len(prompt)} chars)")

            # Call Groq API with low temperature for consistent validation
            response = self._execute_prompt(prompt)

            self.logger.debug(f"Received validation response from API: {response[:200]}...")

            # Parse response
            parsed_response = self._parse_json_response(response)

            # Validate required fields
            required_fields = ["is_valid", "validation_score", "feedback", "approval_reasoning"]
            for field in required_fields:
                if field not in parsed_response:
                    self.logger.warning(
                        f"Missing field '{field}' in Moderator response, using default"
                    )

            # Handle missing issues_found field
            issues_found = parsed_response.get("issues_found", [])
            if isinstance(issues_found, str):
                issues_found = [issues_found]

            # Ensure validation score is within valid range
            validation_score = parsed_response.get("validation_score", 0)
            try:
                validation_score = int(validation_score)
                validation_score = max(0, min(100, validation_score))
            except (ValueError, TypeError):
                self.logger.warning(
                    f"Invalid validation score '{validation_score}', defaulting to 0"
                )
                validation_score = 0

            # Determine final validity based on score and explicit validation
            is_valid = parsed_response.get("is_valid", False)
            if isinstance(is_valid, str):
                is_valid = is_valid.lower() in ["true", "yes", "1", "valid"]

            # Apply minimum score threshold
            if validation_score < self.min_validation_score:
                is_valid = False
                self.logger.info(
                    f"Validation score {validation_score} below threshold \
                        {self.min_validation_score}, marking as invalid"
                )

            result = ValidationResult(
                is_valid=bool(is_valid),
                validation_score=validation_score,
                feedback=parsed_response.get("feedback", "No feedback provided"),
                issues_found=issues_found,
                approval_reasoning=parsed_response.get(
                    "approval_reasoning", "No reasoning provided"
                ),
            )

            self.logger.info("Moderator Agent completed validation")
            self.logger.info(
                f"Validation result: {'VALID' if result.is_valid else 'INVALID'}"
                "(Score: {result.validation_score}/100)"
            )

            if not result.is_valid:
                self.logger.info(f"Issues found: {result.issues_found}")

            return result

        except Exception as e:
            self.logger.error(f"Moderator Agent processing failed: {e}")
            # Return a safe invalid result on error
            return ValidationResult(
                is_valid=False,
                validation_score=0,
                feedback=f"Validation failed due to error: {str(e)}",
                issues_found=[f"Processing error: {str(e)}"],
                approval_reasoning="Validation could not be completed due to technical error",
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

    def get_validation_summary(self, validation_result: ValidationResult) -> str:
        """
        Generate a human-readable summary of the validation result

        Args:
            validation_result: Result from validation process

        Returns:
            String summary of validation
        """
        status = "APPROVED" if validation_result.is_valid else "REJECTED"

        summary = f"""
=== VALIDATION SUMMARY ===
Status: {status}
Score: {validation_result.validation_score}/100

Feedback: {validation_result.feedback}

Reasoning: {validation_result.approval_reasoning}
"""

        if validation_result.issues_found:
            summary += "\nIssues Found:\n"
            for issue in validation_result.issues_found:
                summary += f"  - {issue}\n"

        return summary
