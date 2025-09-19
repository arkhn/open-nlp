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
            )

            self.logger.debug(f"Sending validation prompt to API (length: {len(prompt)} chars)")

            # Call Groq API with low temperature for consistent validation
            response = self._execute_prompt(prompt, self.cfg.model.base_temperature)

            self.logger.debug(f"Received validation response from API: {response[:200]}...")

            # Parse response using new score-based format
            parsed_response = self._parse_score_response(response)

            # Extract scores and reasoning from the new format
            overall_score = parsed_response.get("score", 1)
            clinical_plausibility_score = parsed_response.get("clinical_plausibility_score", 1)
            record_realism_score = parsed_response.get("record_realism_score", 1)
            clinical_significance_score = parsed_response.get("clinical_significance_score", 1)
            reasoning = parsed_response.get("reasoning", "No reasoning provided")

            is_valid = True

            if overall_score < self.min_score:
                is_valid = False
                self.logger.info(
                    f"Overall validation score {overall_score} below threshold \
                        {self.min_score}, marking as invalid"
                )

            result = ValidationResult(
                is_valid=is_valid,
                score=overall_score,
                reasoning=reasoning,
                clinical_plausibility_score=clinical_plausibility_score,
                record_realism_score=record_realism_score,
                clinical_significance_score=clinical_significance_score,
            )

            self.logger.info(
                f"Moderator validation completed: {'VALID' if result.is_valid else 'INVALID'} "
                f"(Overall: {result.score}/5, Clinical: {clinical_plausibility_score}/5, "
                f"Realism: {record_realism_score}/5, Significance: {clinical_significance_score}/5)"
            )

            return result

        except Exception as e:
            self.logger.error(f"Moderator Agent processing failed: {e}")
            # Return a safe invalid result on error
            return ValidationResult(
                is_valid=False,
                score=1.0,
                reasoning=f"Validation failed due to error: {str(e)}",
                clinical_plausibility_score=1.0,
                record_realism_score=1.0,
                clinical_significance_score=1.0,
                retry_attempt=1,
            )

    def _parse_score_response(self, response: str) -> dict:
        """
        Parse response to extract individual scores and overall average.
        """
        try:
            # Initialize with defaults
            clinical_score = 1.0
            realism_score = 1.0
            significance_score = 1.0
            reasoning = response.strip()

            # Extract scores using simple patterns
            patterns = {
                "clinical": r"(?:clinical.*?plausibility|plausibility).*?(\d+(?:\.\d+)?)",
                "realism": r"(?:record.*?realism|realism).*?(\d+(?:\.\d+)?)",
                "significance": r"(?:clinical.*?significance|significance).*?(\d+(?:\.\d+)?)",
                "overall": r"(?:overall|total).*?score.*?(\d+(?:\.\d+)?)",
            }

            for score_type, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    if 1 <= score <= 5:
                        if score_type == "clinical":
                            clinical_score = score
                        elif score_type == "realism":
                            realism_score = score
                        elif score_type == "significance":
                            significance_score = score

            # Calculate overall score as average
            overall_score = (clinical_score + realism_score + significance_score) / 3

            return {
                "reasoning": reasoning,
                "score": round(overall_score, 1),
                "clinical_plausibility_score": round(clinical_score, 1),
                "record_realism_score": round(realism_score, 1),
                "clinical_significance_score": round(significance_score, 1),
            }

        except Exception as e:
            self.logger.error(f"Failed to parse score response: {e}")
            return {
                "reasoning": response.strip() if response else "Failed to parse response",
                "score": 1.0,
                "clinical_plausibility_score": 1.0,
                "record_realism_score": 1.0,
                "clinical_significance_score": 1.0,
            }
