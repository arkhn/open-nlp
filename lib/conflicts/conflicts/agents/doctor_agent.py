from pathlib import Path

from ..core.base import BaseAgent
from ..core.constants import (
    BIOMARKER_MONITORING_CONFLICT_TYPE,
    CLINICAL_HISTORY_CONFLICT_TYPE,
    PRE_POST_CARE_CONFLICT_TYPE,
    SPECIALIZED_CONFLICT_TYPES,
    TEMPORALITY_CONFLICT_TYPE,
)
from ..core.exceptions import DoctorAgentError
from ..core.models import ConflictResult, DocumentPair, PropositionResult

prompts_dir = Path(__file__).parent.parent.parent / "prompts"
DOCTOR_PRE_POST_CARE_PROMPT_PATH = prompts_dir / "doctor_agent_pre_post_care_system.txt"
DOCTOR_TEMPORALITY_PROMPT_PATH = prompts_dir / "doctor_agent_temporality_system.txt"
DOCTOR_CLINICAL_HISTORY_PROMPT_PATH = prompts_dir / "doctor_agent_clinical_history_system.txt"
DOCTOR_BIOMARKER_MONITORING_PROMPT_PATH = (
    prompts_dir / "doctor_agent_biomarker_monitoring_system.txt"
)

# Configuration mapping for conflict types
CONFLICT_TYPE_CONFIG = {
    PRE_POST_CARE_CONFLICT_TYPE: (DOCTOR_PRE_POST_CARE_PROMPT_PATH, "Doctor-PrePostCare"),
    TEMPORALITY_CONFLICT_TYPE: (DOCTOR_TEMPORALITY_PROMPT_PATH, "Doctor-Temporality"),
    CLINICAL_HISTORY_CONFLICT_TYPE: (DOCTOR_CLINICAL_HISTORY_PROMPT_PATH, "Doctor-ClinicalHistory"),
    BIOMARKER_MONITORING_CONFLICT_TYPE: (
        DOCTOR_BIOMARKER_MONITORING_PROMPT_PATH,
        "Doctor-BiomarkerMonitoring",
    ),
}


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
        if conflict_type not in CONFLICT_TYPE_CONFIG:
            raise ValueError(f"Unknown conflict type: {conflict_type}")
        return CONFLICT_TYPE_CONFIG[conflict_type]

    def __call__(
        self,
        document_pair: DocumentPair,
        propositions1: PropositionResult = None,
        propositions2: PropositionResult = None,
    ) -> ConflictResult:
        """
        Analyze documents and choose proposition pairs for the specialized conflict type

        Args:
            document_pair: Pair of clinical documents to analyze
            propositions1: Optional PropositionResult from document 1
            propositions2: Optional PropositionResult from document 2

        Returns:
            ConflictResult containing the chosen proposition pairs and instructions
        """
        self.logger.info(
            f"Analyzing document pair: {document_pair.doc1_id} & {document_pair.doc2_id}"
            f" for conflict type: {self.conflict_type_specialization}"
        )

        try:
            # Prepare propositions strings
            propositions1_str = self._format_propositions(propositions1)
            propositions2_str = self._format_propositions(propositions2)

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
            if isinstance(e, DoctorAgentError):
                raise
            raise DoctorAgentError(f"Doctor Agent failed: {e}") from e

    def _format_propositions(self, proposition_result: PropositionResult = None) -> str:
        """
        Format propositions for prompt inclusion

        Args:
            proposition_result: Optional PropositionResult to format

        Returns:
            Formatted string of propositions or "No propositions provided"
        """
        if proposition_result and proposition_result.propositions:
            return "\n".join(
                [f"{i}. {prop}" for i, prop in enumerate(proposition_result.propositions, 1)]
            )
        return "No propositions provided"
