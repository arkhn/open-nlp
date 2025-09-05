from pathlib import Path

from ..core.base import BaseAgent
from ..core.models import PropositionResult

prompts_dir = Path(__file__).parent.parent.parent / "prompts"
PROPOSITION_SYSTEM_PROMPT_PATH = prompts_dir / "proposition_agent_system.txt"


class PropositionAgent(BaseAgent):
    """
    Proposition Agent decomposes long-form clinical text into discrete propositions
    that can be used for further analysis, particularly for identifying potential
    conflicts between documents.
    """

    def __init__(self, client, model, cfg):
        with open(PROPOSITION_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        super().__init__("Proposition", client, model, cfg, prompt)

    def __call__(self, text: str) -> PropositionResult:
        """
        Decompose clinical text into propositions

        Args:
            text: Long-form clinical text to decompose

        Returns:
            PropositionResult containing the decomposed propositions
        """
        self.logger.info(f"Decomposing text into propositions (length: {len(text)} chars)")

        try:
            truncated_text = self._truncate_document(text)

            # Create prompt using system prompt template
            prompt = self.system_prompt.format(text=truncated_text)

            # Execute the prompt
            response = self._execute_prompt(prompt, temperature=0.3)

            # Parse the JSON response
            result_data = self._parse_json_response(response)

            # Create the result
            result = PropositionResult(
                propositions=result_data["propositions"],
                reasoning=result_data["reasoning"],
                total_propositions=result_data["total_propositions"],
            )

            self.logger.info(
                f"Successfully decomposed text into {result.total_propositions} propositions"
            )
            return result

        except Exception as e:
            self.logger.error(f"Failed to decompose text into propositions: {e}")
            raise

    def decompose_document_pair(
        self, doc1_text: str, doc2_text: str
    ) -> tuple[PropositionResult, PropositionResult]:
        """
        Decompose both documents in a document pair into propositions

        Args:
            doc1_text: Text of the first document
            doc2_text: Text of the second document

        Returns:
            Tuple of PropositionResult for each document
        """
        self.logger.info("Decomposing document pair into propositions")

        # Decompose both documents
        propositions1 = self(doc1_text)
        propositions2 = self(doc2_text)

        return propositions1, propositions2
