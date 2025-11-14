from pathlib import Path

from ..core.base import BaseAgent
from ..core.constants import (
    PRE_POST_CARE_CONFLICT_TYPE,
    SPECIALIZED_CONFLICT_TYPES,
    TEMPORALITY_CONFLICT_TYPE,
)
from ..core.models import ConflictResult, DocumentPair, PropositionResult

prompts_dir = Path(__file__).parent.parent.parent / "prompts"
DOCTOR_PRE_POST_CARE_PROMPT_PATH = prompts_dir / "doctor_agent_pre_post_care_system.txt"
DOCTOR_TEMPORALITY_PROMPT_PATH = prompts_dir / "doctor_agent_temporality_system.txt"


class DoctorAgent(BaseAgent):
    """
    Doctor Agent analyzes two clinical documents and decides which type of
    clinical conflict should be introduced between them.
    """

    def __init__(self, client, model, cfg, conflict_type: str):
        if conflict_type not in SPECIALIZED_CONFLICT_TYPES:
            raise ValueError(
                f"Invalid conflict_type: {conflict_type}. "
                f"Must be one of: {SPECIALIZED_CONFLICT_TYPES}"
            )

        # Load appropriate prompt based on conflict type
        prompt_path, agent_name = self._get_prompt_path_and_name(conflict_type)

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        super().__init__(agent_name, client, model, cfg, prompt)
        self.conflict_type_specialization = conflict_type

    @staticmethod
    def _get_prompt_path_and_name(conflict_type: str):
        """Get prompt path and agent name based on conflict type"""
        if conflict_type == PRE_POST_CARE_CONFLICT_TYPE:
            return DOCTOR_PRE_POST_CARE_PROMPT_PATH, "Doctor-PrePostCare"
        elif conflict_type == TEMPORALITY_CONFLICT_TYPE:
            return DOCTOR_TEMPORALITY_PROMPT_PATH, "Doctor-Temporality"
        else:
            raise ValueError(f"Unknown conflict type: {conflict_type}")

    def __call__(
        self,
        document_pair: DocumentPair,
        propositions1: PropositionResult = None,
        propositions2: PropositionResult = None,
        conflict_type: str = None,  # Kept for API compatibility but not used
    ) -> ConflictResult:
        """
        Analyze documents and choose proposition pairs for the specialized conflict type

        Args:
            document_pair: Pair of clinical documents to analyze
            propositions1: Optional PropositionResult from document 1
            propositions2: Optional PropositionResult from document 2
            conflict_type: Not used (kept for API compatibility)

        Returns:
            ConflictResult containing the chosen proposition pairs and instructions
        """
        self.logger.info(
            f"Analyzing document pair: {document_pair.doc1_id} & {document_pair.doc2_id}"
            f" for conflict type: {self.conflict_type_specialization}"
        )

        try:
            # Prepare propositions strings
            propositions1_str = (
                "\n".join([f"{i}. {prop}" for i, prop in enumerate(propositions1.propositions, 1)])
                if propositions1 and propositions1.propositions
                else "No propositions provided"
            )
            propositions2_str = (
                "\n".join([f"{i}. {prop}" for i, prop in enumerate(propositions2.propositions, 1)])
                if propositions2 and propositions2.propositions
                else "No propositions provided"
            )

            prompt = self.system_prompt.format(
                document1=self._truncate_document(document_pair.doc1_text),
                document2=self._truncate_document(document_pair.doc2_text),
                propositions1=propositions1_str,
                propositions2=propositions2_str,
            )

            self.logger.debug(f"Prompt length: {len(prompt)} chars")

            # Call Groq API
            response = self._execute_prompt(prompt, self.cfg.model.base_temperature)

            # Parse response
            parsed_response = self._parse_json_response(response)

            # Validate required fields
            required_fields = ["reasoning", "modification_instructions", "proposition_pairs"]
            for field in required_fields:
                if field not in parsed_response:
                    raise ValueError(f"Missing required field '{field}' in Doctor Agent response")

            # Use the specialized conflict type
            result_conflict_type = self.conflict_type_specialization

            result = ConflictResult(
                conflict_type=result_conflict_type,
                reasoning=parsed_response["reasoning"],
                modification_instructions=parsed_response["modification_instructions"],
                editor_instructions=parsed_response.get("editor_instructions", []),
                proposition_conflicts=parsed_response.get("proposition_pairs", []),
            )

            self.logger.info("Doctor Agent completed analysis")
            self.logger.info(f"Selected conflict type: {result.conflict_type}")

            return result

        except Exception as e:
            self.logger.error(f"Error in Doctor Agent: {e}")
            raise
