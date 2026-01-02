import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig

from ..agents.doctor_agent import DoctorAgent
from ..agents.editor_agent import EditorAgent
from ..agents.moderator_agent import ModeratorAgent
from ..agents.proposition_agent import PropositionAgent
from ..core.models import EditorResult
from .base import Annotation, ConflictDataItem, DatasetManager
from .constants import EXCERPT_NUMBERS, SPECIALIZED_CONFLICT_TYPES
from .data_loader import DataLoader
from .exceptions import (
    DoctorAgentError,
    EditorAgentError,
    ModeratorAgentError,
    PropositionAgentError,
)
from .models import DocumentPair, PropositionResult, ValidationResult

load_dotenv()


class Pipeline:
    """
    Main pipeline controller that manages the three-agent workflow
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the pipeline

        Args:
            cfg: Hydra configuration object
        """
        # Store configuration
        self.cfg = cfg

        # Use Hydra config values
        self.max_retries = cfg.pipeline.max_retries
        self.early_exit_on_success = cfg.pipeline.get("early_exit_on_success", False)

        # Setup logging - Hydra already configures root logger
        # Just get a logger for this module
        self.logger = logging.getLogger(__name__)

        # Initialize components
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()[:8]
        )
        date_str = datetime.now().strftime("%d%m%Y")
        processed_dir = Path(__file__).parent.parent.parent / "processed"
        filename = processed_dir / f"{git_sha}_{date_str}.json"

        self.dataset_manager = DatasetManager(filename)

        # Create shared OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))

        # Initialize agents with shared client and configuration
        self.proposition_agent = PropositionAgent(self.client, cfg.model.name, cfg)

        # Initialize specialized doctor agents for all conflict types using dict-based approach
        self.doctor_agents = {
            conflict_type: DoctorAgent(
                self.client, cfg.model.name, cfg, conflict_type=conflict_type
            )
            for conflict_type in SPECIALIZED_CONFLICT_TYPES
        }

        # Initialize a single editor agent (works for all conflict types)
        self.editor_agent = EditorAgent(self.client, cfg.model.name, cfg)

        self.moderator_agent = ModeratorAgent(
            self.client,
            cfg.model.name,
            cfg,
            min_validation_score=cfg.validation.min_validation_score,
        )

        self.data_loader = DataLoader(cfg)

        stats = self.data_loader.get_data_statistics()
        self.logger.info(
            f"Loaded dataset with {stats['total_documents']} documents"
            f"from {stats['unique_subjects']} subjects"
        )

    def _execute_agent(self, agent, document_pair: DocumentPair, *extra_args):
        """
        Wrapper to execute an agent with timing

        Args:
            agent: The agent instance to call
            document_pair: The document pair to process
            *extra_args: Additional arguments for agent()

        Returns:
            Tuple of (result, processing_time)
        """
        start_time = time.time()
        result = agent(document_pair, *extra_args)
        processing_time = time.time() - start_time

        return result, processing_time

    def _save_attempts_to_database(
        self,
        pair_id: str,
        document_pair: DocumentPair,
        best_result: Dict[str, Any],
        all_attempts: List[Dict[str, Any]],
        attempt_stats: Dict[str, Any],
    ) -> bool:
        """
        Save all attempts to database as separate entries for each attempt

        Args:
            pair_id: Document pair ID for logging
            document_pair: Original document pair
            best_result: Best result with editor_result, validation_result, conflict_type, etc.
            all_attempts: List of all attempts with metadata
            attempt_stats: Pre-calculated statistics (to avoid recalculation)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            for attempt in all_attempts:
                self._save_single_attempt(attempt, document_pair, best_result)

            self._log_save_statistics(pair_id, best_result, attempt_stats)
            return True

        except Exception as e:
            self.logger.error(f"Failed to save document pair {pair_id}: {e}", exc_info=True)
            return False

    def _save_single_attempt(
        self, attempt: Dict[str, Any], document_pair: DocumentPair, best_result: Dict[str, Any]
    ):
        """Save a single attempt to database"""
        editor_result = attempt["editor_result"]
        validation_result = attempt["validation_result"]
        conflict_type = attempt["conflict_type"]

        # Determine if this is the best result overall
        is_best_conflict = attempt == best_result

        # Create annotations for this specific attempt
        annotations = self._create_annotations_for_attempt(attempt)

        # Create document data using this specific attempt
        doc_data = self.dataset_manager._create_document_data(
            editor_result, document_pair, validation_result, conflict_type, is_best_conflict
        )

        # Create the complete item for this attempt
        conflict_item = ConflictDataItem(data=doc_data, annotations=annotations)

        # Save to database
        self.dataset_manager.save_item(conflict_item)

    def _create_annotations_for_attempt(self, attempt: Dict[str, Any]) -> List[Annotation]:
        """Create annotations for both excerpts in an attempt"""
        annotations = []
        editor_result = attempt["editor_result"]
        validation_result = attempt["validation_result"]
        conflict_type = attempt["conflict_type"]
        attempt_num = attempt["attempt_num"]

        for excerpt_num in EXCERPT_NUMBERS:
            annotation = self.dataset_manager._create_annotation_for_excerpt(
                editor_result, validation_result, conflict_type, excerpt_num, attempt_num
            )
            if annotation:
                annotations.append(annotation)
        return annotations

    def _log_save_statistics(
        self, pair_id: str, best_result: Dict[str, Any], attempt_stats: Dict[str, Any]
    ):
        """
        Log statistics about saved attempts

        Args:
            pair_id: Document pair ID
            best_result: Best result with validation info
            attempt_stats: Pre-calculated statistics (to avoid recalculation)
        """
        status = "VALID" if best_result["validation_result"].is_valid else "INVALID"
        self.logger.info(
            f"Document pair {pair_id} saved (Status: {status}, "
            f"{attempt_stats['valid_attempts']}/{attempt_stats['total_attempts']}"
            " attempts meet threshold, "
            f"{attempt_stats['successful_conflict_types']}/{attempt_stats['total_conflict_types']}"
            " conflict types successful)"
        )

    def _calculate_attempt_statistics(self, all_attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics from all attempts"""
        valid_attempts = sum(1 for a in all_attempts if a["validation_result"].is_valid)
        successful_types = set(
            a["conflict_type"] for a in all_attempts if a["validation_result"].is_valid
        )
        total_types = set(a["conflict_type"] for a in all_attempts)

        return {
            "valid_attempts": valid_attempts,
            "total_attempts": len(all_attempts),
            "successful_conflict_types": len(successful_types),
            "total_conflict_types": len(total_types),
        }

    @property
    def _conflict_type_configs(self) -> List[Tuple[str, DoctorAgent]]:
        """Get list of conflict type configurations"""
        return [(conflict_type, agent) for conflict_type, agent in self.doctor_agents.items()]

    def _process_all_conflict_types(
        self,
        document_pair: DocumentPair,
        proposition_result: Tuple[PropositionResult, PropositionResult],
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Process all conflict types and return all attempts + best result"""
        all_attempts = []
        best_result = None

        for conflict_type, doctor_agent in self._conflict_type_configs:
            self.logger.info(f"Processing conflict type: {conflict_type}")

            # Process conflict type with early exit support
            attempts, _ = self._process_single_conflict_type(
                conflict_type, doctor_agent, document_pair, proposition_result
            )
            all_attempts.extend(attempts)

            # Update best result from this conflict type's attempts
            best_result = self._update_best_result(best_result, attempts)

            # Early exit: Check success ngay sau khi process conflict type
            if self.early_exit_on_success and best_result is not None:
                if best_result["validation_result"].is_valid:
                    self.logger.info(
                        f"Early exit enabled & successful conflict found "
                        f"({best_result['conflict_type']}) - stopping "
                        "all conflict types processing"
                    )
                    break

        return all_attempts, best_result

    def _update_best_result(
        self, best_result: Optional[Dict[str, Any]], attempts: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Update best result from list of attempts"""
        for attempt in attempts:
            if attempt["validation_result"].is_valid:
                if (
                    best_result is None
                    or attempt["validation_result"].overall_score
                    > best_result["validation_result"].overall_score
                ):
                    best_result = attempt
        return best_result

    def _process_single_conflict_type(
        self,
        conflict_type: str,
        doctor_agent: DoctorAgent,
        document_pair: DocumentPair,
        proposition_result: Tuple[PropositionResult, PropositionResult],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Process one conflict type with retry logic

        Returns:
            Tuple of (attempts list, found_valid flag)
            found_valid: True if a valid result was found (for early exit tracking)
        """
        attempts = []
        found_valid = False

        # Doctor Agent chooses proposition pairs for this conflict type
        try:
            conflict_result, doctor_time = self._execute_agent(
                doctor_agent, document_pair, proposition_result[0], proposition_result[1]
            )
        except DoctorAgentError as e:
            self.logger.error(f"Doctor Agent failed for conflict type {conflict_type}: {e}")
            return attempts, found_valid
        except Exception as e:
            self.logger.error(
                f"Unexpected error in Doctor Agent for conflict type {conflict_type}: {e}"
            )
            return attempts, found_valid

        # Editor and Moderator agents with retry logic for this conflict type
        for attempt_num in range(1, self.max_retries + 1):
            attempt_data = self._make_single_attempt(
                document_pair, conflict_result, conflict_type, doctor_time, attempt_num
            )
            attempts.append(attempt_data)

            self._log_attempt_result(attempt_data, attempt_num, conflict_type)

            if attempt_data["validation_result"].is_valid:
                found_valid = True
                if self.early_exit_on_success:
                    self.logger.info(
                        f"Validation passed for {conflict_type} (attempt {attempt_num}), "
                        f"early exit enabled - stopping conflict type iteration"
                    )
                    break

            if attempt_num < self.max_retries and not attempt_data["validation_result"].is_valid:
                self.logger.warning(f"Validation failed for {conflict_type}, retrying...")
                time.sleep(1)

        return attempts, found_valid

    def _make_single_attempt(
        self,
        document_pair: DocumentPair,
        conflict_result,
        conflict_type: str,
        doctor_time: float,
        attempt_num: int,
    ) -> Dict[str, Any]:
        """Make a single attempt: execute editor and moderator agents"""
        # Execute editor agent
        try:
            editor_result, editor_time = self._execute_agent(
                self.editor_agent, document_pair, conflict_result
            )
        except EditorAgentError as e:
            self.logger.warning(
                f"Editor agent failed for {conflict_type} (attempt {attempt_num}): {e}"
            )
            # Create failed validation result when editor fails
            validation_result = self._create_failed_validation_result(f"Editor agent failed: {e}")
            # Create a minimal editor result for tracking

            editor_result = EditorResult(
                modified_document1=document_pair.doc1_text,
                modified_document2=document_pair.doc2_text,
                changes_made=f"Editor agent failed: {e}",
                change_info_1="No changes made - editor agent failed",
                change_info_2="No changes made - editor agent failed",
            )
            editor_time = 0
            moderator_time = 0
        else:
            # Execute moderator agent for validation
            try:
                validation_result, moderator_time = self._execute_agent(
                    self.moderator_agent, document_pair, editor_result, conflict_type
                )
            except ModeratorAgentError as e:
                self.logger.warning(
                    f"Moderator agent failed for {conflict_type} (attempt {attempt_num}): {e}"
                )
                validation_result = self._create_failed_validation_result(
                    f"Moderator agent failed: {e}"
                )
                moderator_time = 0

        return {
            "conflict_type": conflict_type,
            "conflict_result": conflict_result,
            "editor_result": editor_result,
            "validation_result": validation_result,
            "doctor_time": doctor_time,
            "editor_time": editor_time,
            "moderator_time": moderator_time,
            "attempt_num": attempt_num,
        }

    def _log_attempt_result(
        self, attempt_data: Dict[str, Any], attempt_num: int, conflict_type: str
    ):
        """Log the result of a single attempt"""
        validation_result = attempt_data["validation_result"]
        self.logger.info(
            f"Attempt {attempt_num} for {conflict_type}: "
            f"valid={validation_result.is_valid}, "
            f"overall={validation_result.overall_score}/5, "
            f"clinical={validation_result.clinical_plausibility_score}/5, "
            f"temporal={validation_result.temporal_appropriateness_score}/5, "
            f"significance={validation_result.clinical_significance_score}/5"
        )

    def _create_failed_validation_result(self, reason: str) -> ValidationResult:
        """Create a ValidationResult indicating failure"""
        return ValidationResult(
            is_valid=False,
            overall_score=1.0,
            reasoning=reason,
            clinical_plausibility_score=1.0,
            temporal_appropriateness_score=1.0,
            clinical_significance_score=1.0,
        )

    def _initialize_result_data(self, pair_id: str) -> Dict[str, Any]:
        """Initialize result data dictionary with default values"""
        return {
            "pair_id": pair_id,
            "success": False,
            "conflict_type": None,
            "processing_time": 0,
            "proposition_result": None,
            "doctor_result": None,
            "editor_result": None,
            "moderator_result": None,
            "proposition_time": 0,
            "doctor_time": 0,
            "editor_time": 0,
            "moderator_time": 0,
        }

    def _create_failed_result(self, pair_id: str, error: str) -> Dict[str, Any]:
        """Create a failed result dictionary with error information"""
        result = self._initialize_result_data(pair_id)
        result["error"] = error
        return result

    def process_document_pair(self, document_pair: DocumentPair) -> Tuple[bool, Dict[str, Any]]:
        """
        Process a single document pair through the complete pipeline

        Args:
            document_pair: Pair of clinical documents to process

        Returns:
            Tuple of (success, result_data)
        """
        pair_id = f"{document_pair.doc1_id}_{document_pair.doc2_id}"
        start_time = time.time()

        self.logger.info(f"Processing document pair {pair_id}")

        result_data = self._initialize_result_data(pair_id)

        # Step 1: Extract propositions from documents
        proposition_start_time = time.time()
        self.logger.info("Decomposing document pair into propositions")
        propositions1 = self.proposition_agent(document_pair.doc1_text)
        propositions2 = self.proposition_agent(document_pair.doc2_text)
        proposition_result = (propositions1, propositions2)
        proposition_time = time.time() - proposition_start_time
        result_data["proposition_result"] = proposition_result
        result_data["proposition_time"] = proposition_time

        # Step 2: Process both specialized conflict types for this proposition set
        all_attempts, best_result = self._process_all_conflict_types(
            document_pair, proposition_result
        )

        # Use best result if found, otherwise use the last attempt
        final_result = best_result if best_result else (all_attempts[-1] if all_attempts else None)

        if not final_result:
            self.logger.error("No attempts were made - cannot process document pair")
            result_data["processing_time"] = time.time() - start_time
            return False, result_data

        # Update result data from final result
        self._update_result_data_from_final(result_data, final_result)

        # Calculate statistics once and reuse
        attempt_stats = self._calculate_attempt_statistics(all_attempts)

        # Step 4: Save all attempts to database
        is_success = self._save_attempts_to_database(
            pair_id, document_pair, final_result, all_attempts, attempt_stats
        )

        # Track all attempts for analysis
        result_data["all_attempts"] = self._build_saved_attempts_list(all_attempts, is_success)
        result_data["success"] = final_result["validation_result"].is_valid
        result_data["processing_time"] = time.time() - start_time

        # Summary log
        self._log_processing_summary(
            pair_id, result_data, final_result, all_attempts, proposition_result, attempt_stats
        )

        return result_data["success"], result_data

    def _update_result_data_from_final(
        self, result_data: Dict[str, Any], final_result: Dict[str, Any]
    ):
        """Update result_data dictionary from final_result"""
        result_data["doctor_result"] = final_result["conflict_result"]
        result_data["doctor_time"] = final_result["doctor_time"]
        result_data["conflict_type"] = final_result["conflict_type"]
        result_data["editor_result"] = final_result["editor_result"]
        result_data["editor_time"] = final_result["editor_time"]
        result_data["moderator_result"] = final_result["validation_result"]
        result_data["moderator_time"] = final_result["moderator_time"]

    def _build_saved_attempts_list(
        self, all_attempts: List[Dict[str, Any]], is_success: bool
    ) -> List[Dict[str, Any]]:
        """Build list of saved attempts metadata"""
        saved_attempts = []
        for attempt in all_attempts:
            saved_attempts.append(
                {
                    "attempt": attempt["attempt_num"],
                    "saved": is_success,
                    "valid": attempt["validation_result"].is_valid,
                    "overall_score": attempt["validation_result"].overall_score,
                }
            )
        return saved_attempts

    def _log_processing_summary(
        self,
        pair_id: str,
        result_data: Dict[str, Any],
        final_result: Dict[str, Any],
        all_attempts: List[Dict[str, Any]],
        proposition_result: Tuple[PropositionResult, PropositionResult],
        attempt_stats: Dict[str, Any],
    ):
        """
        Log summary of processing results

        Args:
            pair_id: Document pair ID
            result_data: Result data dictionary
            final_result: Final result with validation info
            all_attempts: List of all attempts (for reference, not recalculated)
            proposition_result: Proposition results from both documents
            attempt_stats: Pre-calculated statistics (to avoid recalculation)
        """
        status = "SUCCESS" if result_data["success"] else "FAILED"
        total_propositions = len(proposition_result[0].propositions) + len(
            proposition_result[1].propositions
        )

        self.logger.info(
            f"Pair {pair_id}: {status} - {final_result['conflict_type']} conflict (best of "
            f"{attempt_stats['successful_conflict_types']}/{attempt_stats['total_conflict_types']} "
            f"successful types), {total_propositions} propositions, "
            f"{attempt_stats['valid_attempts']}/{attempt_stats['total_attempts']} valid"
        )

    def execute(
        self,
        dataset_size: int = 5,
        category_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a batch of document pairs

        Args:
            dataset_size: Number of document pairs to process
            category_filter: List of categories to filter documents by

        Returns:
            Dictionary with batch processing results
        """
        self.logger.info(f"Starting batch processing of {dataset_size} document pairs")

        batch_start_time = time.time()
        document_pairs = self.data_loader.get_random_document_pairs(
            dataset_size=dataset_size, category_filter=category_filter
        )

        results = []
        successful = 0

        for doc_pair in document_pairs:
            try:
                success, result_data = self.process_document_pair(doc_pair)
                results.append(result_data)

                if success:
                    successful += 1
            except (
                PropositionAgentError,
                DoctorAgentError,
                EditorAgentError,
                ModeratorAgentError,
            ) as e:
                self.logger.error(
                    "Agent error processing document pair "
                    f"{doc_pair.doc1_id}_{doc_pair.doc2_id}: {e}"
                )
                pair_id = f"{doc_pair.doc1_id}_{doc_pair.doc2_id}"
                failed_result = self._create_failed_result(pair_id, f"{type(e).__name__}: {e}")
                results.append(failed_result)
            except Exception as e:
                self.logger.error(
                    "Unexpected error processing document pair "
                    f"{doc_pair.doc1_id}_{doc_pair.doc2_id}: {e}",
                    exc_info=True,
                )
                pair_id = f"{doc_pair.doc1_id}_{doc_pair.doc2_id}"
                failed_result = self._create_failed_result(pair_id, f"Unexpected error: {e}")
                results.append(failed_result)

        batch_time = time.time() - batch_start_time
        success_rate = (successful / len(document_pairs)) * 100

        self.logger.info(
            f"Batch completed: {successful}/{len(document_pairs)} successful ({success_rate:.1f}%)"
        )

        return {
            "total_pairs": len(document_pairs),
            "successful": successful,
            "failed": len(document_pairs) - successful,
            "success_rate": success_rate,
            "total_processing_time": batch_time,
            "results": results,
        }

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline performance

        Returns:
            Dictionary with pipeline statistics
        """
        total_validated = self.dataset_manager.get_validated_documents_count()
        data_stats = self.data_loader.get_data_statistics()

        # Build doctor agents statistics dynamically
        doctor_agents_stats = {
            f"doctor_{conflict_type}": {
                "name": agent.name,
                "conflict_type": conflict_type,
            }
            for conflict_type, agent in self.doctor_agents.items()
        }

        return {
            "validated_documents": total_validated,
            "dataset_statistics": data_stats,
            "agents": {
                **doctor_agents_stats,
                "editor": {"name": self.editor_agent.name},
                "moderator": {
                    "name": self.moderator_agent.name,
                    "min_score": self.moderator_agent.min_score,
                },
            },
            "configuration": {
                "max_retries": self.max_retries,
                "dataset_path": str(self.dataset_manager.json_path),
            },
        }
