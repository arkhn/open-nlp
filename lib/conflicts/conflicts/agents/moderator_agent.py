import re
from pathlib import Path

from ..core.base import BaseAgent
from ..core.models import DocumentPair, EditorResult, ValidationResult

prompts_dir = Path(__file__).parent.parent.parent / "prompts"
MODERATOR_SYSTEM_PROMPT_PATH = prompts_dir / "moderator_agent_system.txt"


class ModeratorAgent(BaseAgent):
    """
    Moderator Agent validates if the modifications are acceptable and realistic.
    If valid, documents are saved to persistent storage.
    If invalid, they are returned to Editor Agent for re-modification.
    """

    def __init__(self, client, model, cfg, min_validation_score: int = 4):
        with open(MODERATOR_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        super().__init__("Moderator", client, model, cfg, prompt)
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
            prompt = self.system_prompt.format(
                context_document_1=self._truncate_document(modified_docs.modified_document1),
                context_document_2=self._truncate_document(modified_docs.modified_document2),
                conflict_1=modified_docs.change_info_1 or "No change info available",
                conflict_2=modified_docs.change_info_2 or "No change info available",
                original_excerpt_1=modified_docs.original_excerpt_1
                or "No original excerpt available",
                modified_excerpt_1=modified_docs.modified_excerpt_1
                or "No modified excerpt available",
                original_excerpt_2=modified_docs.original_excerpt_2
                or "No original excerpt available",
                modified_excerpt_2=modified_docs.modified_excerpt_2
                or "No modified excerpt available",
            )

            self.logger.debug(f"Sending validation prompt to API (length: {len(prompt)} chars)")

            # Call Groq API with low temperature for consistent validation
            response = self._execute_prompt(prompt, self.cfg.model.base_temperature)

            self.logger.debug(f"Received validation response from API: {response[:200]}...")

            # Parse response using new score-based format
            parsed_response = self._parse_score_response(response)

            # Extract scores and reasoning
            overall_score = parsed_response.get("overall_score", 1)
            clinical_plausibility_score = parsed_response.get("clinical_plausibility_score", 1)
            record_realism_score = parsed_response.get("record_realism_score", 1)
            clinical_significance_score = parsed_response.get("clinical_significance_score", 1)
            reasoning = parsed_response.get("reasoning", "No reasoning provided")

            # Determine validity based on score threshold
            is_valid = overall_score >= self.min_score

            if not is_valid:
                self.logger.info(
                    f"Overall validation score {overall_score} below"
                    f" threshold {self.min_score}, marking as invalid"
                )

            result = ValidationResult(
                is_valid=is_valid,
                overall_score=overall_score,
                reasoning=reasoning,
                clinical_plausibility_score=clinical_plausibility_score,
                record_realism_score=record_realism_score,
                clinical_significance_score=clinical_significance_score,
            )

            self.logger.info(
                f"Moderator validation completed: {'VALID' if result.is_valid else 'INVALID'} "
                f"(Overall: {result.overall_score}/5, Clinical: {clinical_plausibility_score}/5, "
                f"Realism: {record_realism_score}/5, Significance: {clinical_significance_score}/5)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Moderator Agent processing failed: {e}")
            return self._create_error_result(str(e))

    def _create_error_result(self, error_message: str) -> ValidationResult:
        """Create a standardized error result"""
        return ValidationResult(
            is_valid=False,
            overall_score=1.0,
            reasoning=f"Validation failed due to error: {error_message}",
            clinical_plausibility_score=1.0,
            record_realism_score=1.0,
            clinical_significance_score=1.0,
        )

    def _parse_score_response(self, response: str) -> dict:
        """
        Parse response to extract individual scores and overall average.
        """
        try:
            # Initialize with defaults
            scores = {"clinical": 1.0, "realism": 1.0, "significance": 1.0}
            reasoning = response.strip()

            # Extract scores using flexible patterns (order matters - more specific first)
            patterns = {
                "significance": r"(?:clinical.*?significance|significance).*?(\d+(?:\.\d+)?)",
                "clinical": r"(?:clinical.*?plausibility|plausibility).*?(\d+(?:\.\d+)?)",
                "realism": r"(?:record.*?realism|realism).*?(\d+(?:\.\d+)?)",
                "overall_score": r"(?:overall.*?score|score).*?(\d+(?:\.\d+)?)",
            }

            for score_type, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if 1 <= score <= 5:
                        scores[score_type] = score

            return {
                "reasoning": reasoning,
                "overall_score": scores["overall_score"],
                "clinical_plausibility_score": round(scores["clinical"], 1),
                "record_realism_score": round(scores["realism"], 1),
                "clinical_significance_score": round(scores["significance"], 1),
            }

        except Exception as e:
            self.logger.error(f"Failed to parse score response: {e}")
            return {
                "reasoning": response.strip() if response else "Failed to parse response",
                "overall_score": 1.0,
                "clinical_plausibility_score": 1.0,
                "record_realism_score": 1.0,
                "clinical_significance_score": 1.0,
            }
