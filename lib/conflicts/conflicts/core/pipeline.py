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
from .base import ConflictDataItem, DatasetManager
from .constants import PRE_POST_CARE_CONFLICT_TYPE, TEMPORALITY_CONFLICT_TYPE
from .data_loader import DataLoader
from .models import DocumentPair, ValidationResult

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

        # Initialize specialized doctor agents for the two conflict types
        self.doctor_agent_pre_post_care = DoctorAgent(
            self.client, cfg.model.name, cfg, conflict_type=PRE_POST_CARE_CONFLICT_TYPE
        )
        self.doctor_agent_temporality = DoctorAgent(
            self.client, cfg.model.name, cfg, conflict_type=TEMPORALITY_CONFLICT_TYPE
        )

        # Initialize specialized editor agents for the two conflict types
        self.editor_agent_pre_post_care = EditorAgent(
            self.client, cfg.model.name, cfg, conflict_type=PRE_POST_CARE_CONFLICT_TYPE
        )
        self.editor_agent_temporality = EditorAgent(
            self.client, cfg.model.name, cfg, conflict_type=TEMPORALITY_CONFLICT_TYPE
        )

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

    def _save_to_database(
        self,
        pair_id: str,
        document_pair: DocumentPair,
        editor_result,
        conflict_type: str,
        validation_result,
    ) -> bool:
        """
        Save documents to database (both successful and failed attempts)

        Args:
            pair_id: Document pair ID for logging
            document_pair: The document pair
            editor_result: Result from editor agent
            conflict_type: Type of conflict identified
            validation_result: Result from moderator agent

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            doc_id = self.dataset_manager.save_validated_documents(
                document_pair, editor_result, conflict_type, validation_result
            )
            status = "VALID" if validation_result.is_valid else "INVALID"
            threshold_status = "MEETS" if validation_result.is_valid else "BELOW"
            self.logger.info(
                f"Document pair {pair_id} saved (DB ID: {doc_id}, Status: {status},"
                f" Threshold: {threshold_status})"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to save document pair {pair_id}: {e}")
            return False

    def _save_attempts_to_database(
        self,
        pair_id: str,
        document_pair: DocumentPair,
        best_result: Dict[str, Any],
        all_attempts: List[Dict[str, Any]],
    ) -> bool:
        """
        Save all attempts to database as separate entries for each attempt

        Args:
            pair_id: Document pair ID for logging
            document_pair: Original document pair
            best_result: Best result with editor_result, validation_result, conflict_type, etc.
            all_attempts: List of all attempts with metadata

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create a separate entry for each retry attempt
            for attempt in all_attempts:
                editor_result = attempt["editor_result"]
                validation_result = attempt["validation_result"]
                conflict_type = attempt["conflict_type"]
                attempt_num = attempt["attempt_num"]

                # Determine if this is the best result overall
                is_best_conflict = attempt == best_result

                # Create annotations for this specific attempt
                annotations = []
                for excerpt_num in [1, 2]:
                    annotation = self.dataset_manager._create_annotation_for_excerpt(
                        editor_result,
                        validation_result,
                        conflict_type,
                        excerpt_num,
                        attempt_num,
                    )
                    if annotation:
                        annotations.append(annotation)

                # Create document data using this specific attempt
                doc_data = self.dataset_manager._create_document_data(
                    editor_result, document_pair, validation_result, conflict_type, is_best_conflict
                )

                # Create the complete item for this attempt
                conflict_item = ConflictDataItem(
                    data=doc_data,
                    annotations=annotations,
                )

                # Save to database
                self.dataset_manager.save_item(conflict_item)

            # Count statistics
            valid_attempts = sum(
                1 for attempt in all_attempts if attempt["validation_result"].is_valid
            )
            successful_conflict_types = len(
                set(
                    attempt["conflict_type"]
                    for attempt in all_attempts
                    if attempt["validation_result"].is_valid
                )
            )
            total_conflict_types = len(set(attempt["conflict_type"] for attempt in all_attempts))

            status = "VALID" if best_result["validation_result"].is_valid else "INVALID"
            self.logger.info(
                f"Document pair {pair_id} saved (Status: {status},"
                f" {valid_attempts}/{len(all_attempts)} attempts meet threshold,"
                f" {successful_conflict_types}/{total_conflict_types} conflict types successful)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to save document pair {pair_id}: {e}")
            return False

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

        result_data = {
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

        # Step 1: Proposition Agent decomposes documents into propositions
        start_prop_time = time.time()
        proposition_result = self.proposition_agent.decompose_document_pair(
            document_pair.doc1_text, document_pair.doc2_text
        )
        proposition_time = time.time() - start_prop_time
        result_data["proposition_result"] = proposition_result
        result_data["proposition_time"] = proposition_time

        # Step 2: Process both specialized conflict types for this proposition set
        all_attempts = []
        best_result = None

        # Process both specialized conflict types
        conflict_type_configs = [
            (
                PRE_POST_CARE_CONFLICT_TYPE,
                self.doctor_agent_pre_post_care,
                self.editor_agent_pre_post_care,
            ),
            (
                TEMPORALITY_CONFLICT_TYPE,
                self.doctor_agent_temporality,
                self.editor_agent_temporality,
            ),
        ]

        for conflict_type, doctor_agent, editor_agent in conflict_type_configs:
            self.logger.info(f"Processing conflict type: {conflict_type}")

            # Doctor Agent chooses proposition pairs for this conflict type
            try:
                conflict_result, doctor_time = self._execute_agent(
                    doctor_agent,
                    document_pair,
                    proposition_result[0],
                    proposition_result[1],
                    conflict_type,
                )
            except Exception as e:
                self.logger.error(f"Doctor Agent failed for conflict type {conflict_type}: {e}")
                continue

            # Step 3: Editor and Moderator agents with retry logic for this conflict type
            for attempt in range(1, self.max_retries + 1):
                # Execute editor agent
                editor_result, editor_time = self._execute_agent(
                    editor_agent, document_pair, conflict_result
                )

                # Check if editor agent failed to create modifications
                if "Failed to create conflict" in editor_result.changes_made:
                    self.logger.warning(
                        f"Editor agent failed to create modifications for {conflict_type},"
                        "skipping moderator validation"
                    )
                    validation_result = ValidationResult(
                        is_valid=False,
                        overall_score=1.0,
                        reasoning="Editor agent failed to modify - no changes to validate",
                        clinical_plausibility_score=1.0,
                        temporal_appropriateness_score=1.0,
                        clinical_significance_score=1.0,
                    )
                else:
                    # Execute moderator agent for validation
                    validation_result, moderator_time = self._execute_agent(
                        self.moderator_agent, document_pair, editor_result, conflict_type
                    )

                # Store this attempt with all metadata
                attempt_data = {
                    "conflict_type": conflict_type,
                    "conflict_result": conflict_result,
                    "editor_result": editor_result,
                    "validation_result": validation_result,
                    "doctor_time": doctor_time,
                    "editor_time": editor_time,
                    "moderator_time": moderator_time if "moderator_time" in locals() else 0,
                    "attempt_num": attempt,
                }
                all_attempts.append(attempt_data)

                self.logger.info(
                    f"Attempt {attempt} for {conflict_type}: "
                    f"valid={validation_result.is_valid}, "
                    f"overall={validation_result.overall_score}/5, "
                    f"clinical={validation_result.clinical_plausibility_score}/5, "
                    f"temporal={validation_result.temporal_appropriateness_score}/5, "
                    f"significance={validation_result.clinical_significance_score}/5"
                )

                # Check if this attempt was successful and update best result
                if validation_result.is_valid:
                    if (
                        best_result is None
                        or validation_result.overall_score
                        > best_result["validation_result"].overall_score
                    ):
                        best_result = attempt_data
                        result_data["success"] = True

                    if self.early_exit_on_success:
                        self.logger.info(
                            f"Validation passed for {conflict_type}, early exit enabled - "
                            f"stopping conflict type iteration"
                        )
                        break

                if attempt < self.max_retries and not validation_result.is_valid:
                    self.logger.warning(f"Validation failed for {conflict_type}, retrying...")
                    time.sleep(1)

            # Early exit if we found success and early exit is enabled
            if self.early_exit_on_success and best_result:
                self.logger.info(
                    "Early exit enabled & successful conflict found - "
                    "stopping all conflict processing"
                )
                break

        # Use best result if found, otherwise use the last attempt
        final_result = best_result if best_result else all_attempts[-1] if all_attempts else None

        if not final_result:
            self.logger.error("No attempts were made - cannot process document pair")
            result_data["processing_time"] = time.time() - start_time
            return False, result_data

        # Update result data
        result_data["doctor_result"] = final_result["conflict_result"]
        result_data["doctor_time"] = final_result["doctor_time"]
        result_data["conflict_type"] = final_result["conflict_type"]
        result_data["editor_result"] = final_result["editor_result"]
        result_data["editor_time"] = final_result["editor_time"]
        result_data["moderator_result"] = final_result["validation_result"]
        result_data["moderator_time"] = final_result["moderator_time"]

        # Step 4: Save all attempts to database
        is_success = self._save_attempts_to_database(
            pair_id,
            document_pair,
            final_result,
            all_attempts,
        )

        # Track all attempts for analysis
        saved_attempts = []
        for attempt in all_attempts:
            saved_attempts.append(
                {
                    "attempt": attempt["attempt_num"],
                    "saved": is_success,
                    "valid": attempt["validation_result"].is_valid,
                    "overall_score": attempt["validation_result"].overall_score,
                    "meets_threshold": attempt["validation_result"].is_valid,
                }
            )

        result_data["all_attempts"] = saved_attempts
        result_data["success"] = final_result["validation_result"].is_valid
        result_data["processing_time"] = time.time() - start_time

        # Summary log
        status = "SUCCESS" if result_data["success"] else "FAILED"
        total_attempts = len(saved_attempts)
        successful_attempts = sum(1 for attempt in saved_attempts if attempt["valid"])
        successful_conflict_types = len(
            set(
                attempt["conflict_type"]
                for attempt in all_attempts
                if attempt["validation_result"].is_valid
            )
        )
        total_conflict_types = len(set(attempt["conflict_type"] for attempt in all_attempts))

        self.logger.info(
            f"Pair {pair_id}: {status} - {final_result['conflict_type']} conflict "
            f"(best of {successful_conflict_types}/{total_conflict_types} successful types), "
            f"{proposition_result[0].total_propositions + proposition_result[1].total_propositions}"
            f" propositions, {successful_attempts}/{total_attempts} valid"
        )

        return result_data["success"], result_data

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
            except Exception as e:
                self.logger.error(
                    f"Failed to process document pair {doc_pair.doc1_id}_{doc_pair.doc2_id}: {e}"
                )
                failed_result = {
                    "pair_id": f"{doc_pair.doc1_id}_{doc_pair.doc2_id}",
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
                    "error": str(e),
                }
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

        return {
            "validated_documents": total_validated,
            "dataset_statistics": data_stats,
            "agents": {
                "doctor_pre_post_care": {
                    "name": self.doctor_agent_pre_post_care.name,
                    "conflict_type": PRE_POST_CARE_CONFLICT_TYPE,
                },
                "doctor_temporality": {
                    "name": self.doctor_agent_temporality.name,
                    "conflict_type": TEMPORALITY_CONFLICT_TYPE,
                },
                "editor_pre_post_care": {"name": self.editor_agent_pre_post_care.name},
                "editor_temporality": {"name": self.editor_agent_temporality.name},
                "moderator": {
                    "name": self.moderator_agent.name,
                    "min_score": self.moderator_agent.min_score,
                },
            },
            "configuration": {
                "max_retries": self.max_retries,
                "dataset_path": self.dataset_manager.json_path,
            },
        }
