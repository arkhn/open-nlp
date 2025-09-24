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
        self.doctor_agent = DoctorAgent(self.client, cfg.model.name, cfg)
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

    def _save_all_attempts_to_database(
        self,
        pair_id: str,
        document_pair: DocumentPair,
        all_attempts: List[Tuple],
        final_conflict_type: str,
        all_conflict_attempts: List[Dict],
    ) -> bool:
        """
        Save all attempts to database as multiple annotations within a single document pair

        Args:
            pair_id: Document pair ID for logging
            document_pair: Original document pair
            all_attempts: List of (editor_result, validation_result, attempt_num) tuples
            final_conflict_type: Type of conflict that was selected as best
            all_conflict_attempts: List of all conflict type attempts (for counting)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Use the first attempt's editor result for the document content
            first_editor_result = all_attempts[0][0]

            # Count successful conflict types for logging
            successful_conflict_types = []
            for conflict_attempt in all_conflict_attempts:
                attempts = conflict_attempt["attempts"]
                for editor_result_attempt, validation_result_attempt, attempt_num in attempts:
                    if validation_result_attempt.is_valid:
                        successful_conflict_types.append(conflict_attempt["conflict_type"])
                        break  # Only count each conflict type once

            # Create annotations for all attempts
            all_annotations = []
            for editor_result_attempt, validation_result_attempt, attempt_num in all_attempts:
                # Create annotations for both excerpts using helper method
                for excerpt_num in [1, 2]:
                    annotation = self.dataset_manager._create_annotation_for_excerpt(
                        editor_result_attempt,
                        validation_result_attempt,
                        final_conflict_type,
                        excerpt_num,
                        attempt_num,
                    )
                    if annotation:
                        all_annotations.append(annotation)

            # Create document data using the first attempt's content
            doc_data = self.dataset_manager._create_document_data(
                first_editor_result, document_pair, all_attempts[0][1], final_conflict_type
            )

            # Create the complete item with all annotations
            conflict_item = ConflictDataItem(
                data=doc_data, annotations=[{"result": all_annotations}]
            )

            # Save to database
            doc_id = self.dataset_manager.save_item(conflict_item)

            status = "VALID" if any(attempt[1].is_valid for attempt in all_attempts) else "INVALID"
            threshold_count = sum(1 for attempt in all_attempts if attempt[1].is_valid)

            self.logger.info(
                f"Document pair {pair_id} saved (DB ID: {doc_id}, Status: {status},"
                f" {threshold_count}/{len(all_attempts)} attempts meet threshold,"
                f" {len(successful_conflict_types)}/{len(all_conflict_attempts)}"
                " conflict types successful)"
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

        # Step 2: Try each conflict type with Doctor Agent choosing proposition pairs
        all_conflict_attempts = []
        successful_conflict = None

        # Get all available conflict types
        conflict_types = list(self.doctor_agent.list_all_conflict_types().keys())

        if not conflict_types:
            self.logger.error("No conflict types available - cannot process document pair")
            result_data["processing_time"] = time.time() - start_time
            return False, result_data

        for conflict_type in conflict_types:
            self.logger.info(f"Trying conflict type: {conflict_type}")

            # Doctor Agent chooses proposition pairs for this conflict type
            try:
                conflict_result, doctor_time = self._execute_agent(
                    self.doctor_agent,
                    document_pair,
                    proposition_result[0],
                    proposition_result[1],
                    conflict_type,
                )
            except Exception as e:
                self.logger.error(f"Doctor Agent failed for conflict type {conflict_type}: {e}")
                # Skip this conflict type and continue to next one
                continue

            # Step 3: Editor and Moderator agents with retry logic for this conflict type
            validation_result = None
            editor_result = None
            conflict_attempts = []  # Store attempts for this conflict type

            for attempt in range(1, self.max_retries + 1):
                # Execute editor agent
                editor_result, editor_time = self._execute_agent(
                    self.editor_agent, document_pair, conflict_result
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
                    conflict_attempts.append((editor_result, validation_result, attempt))
                    break

                # Execute moderator agent for validation
                validation_result, moderator_time = self._execute_agent(
                    self.moderator_agent, document_pair, editor_result, conflict_type
                )

                # Store this attempt
                conflict_attempts.append((editor_result, validation_result, attempt))

                self.logger.info(
                    f"Attempt {attempt} for {conflict_type}: "
                    f"valid={validation_result.is_valid}, "
                    f"overall={validation_result.overall_score}/5, "
                    f"clinical={validation_result.clinical_plausibility_score}/5, "
                    f"temporal={validation_result.temporal_appropriateness_score}/5, "
                    f"significance={validation_result.clinical_significance_score}/5"
                )

                # Check if this attempt was successful
                if validation_result.is_valid:
                    successful_conflict = {
                        "conflict_type": conflict_type,
                        "conflict_result": conflict_result,
                        "editor_result": editor_result,
                        "validation_result": validation_result,
                        "doctor_time": doctor_time,
                        "editor_time": editor_time,
                        "moderator_time": moderator_time,
                        "attempts": conflict_attempts,
                    }
                    result_data["success"] = True
                    if self.early_exit_on_success:
                        self.logger.info(
                            f"Validation passed for {conflict_type}, early exit enabled - "
                            f"stopping conflict type iteration"
                        )
                        break
                    else:
                        self.logger.info(
                            f"Validation passed for {conflict_type}, "
                            f"continuing to next conflict type..."
                        )
                        break

                if attempt < self.max_retries:
                    self.logger.warning(f"Validation failed for {conflict_type}, retrying...")
                    time.sleep(1)

            # Store all attempts for this conflict type
            all_conflict_attempts.append(
                {
                    "conflict_type": conflict_type,
                    "conflict_result": conflict_result,
                    "doctor_time": doctor_time,
                    "attempts": conflict_attempts,
                }
            )

            # Continue to next conflict type unless early exit is enabled and we found success
            if self.early_exit_on_success and successful_conflict:
                self.logger.info(
                    "Early exit enabled & successful conflict found - "
                    "stopping all conflict processing"
                )
                break

        # Choose the best successful conflict (highest validation score) or the last attempted one
        if successful_conflict:
            # If we have multiple successful conflicts, choose the one with highest score
            best_conflict = successful_conflict
            for conflict_attempt in all_conflict_attempts:
                for (
                    editor_result_attempt,
                    validation_result_attempt,
                    attempt_num,
                ) in conflict_attempt["attempts"]:
                    if (
                        validation_result_attempt.is_valid
                        and validation_result_attempt.overall_score
                        > best_conflict["validation_result"].overall_score
                    ):
                        best_conflict = {
                            "conflict_type": conflict_attempt["conflict_type"],
                            "conflict_result": conflict_attempt["conflict_result"],
                            "editor_result": editor_result_attempt,
                            "validation_result": validation_result_attempt,
                            "doctor_time": conflict_attempt["doctor_time"],
                            "editor_time": 0,  # We don't track individual attempt times
                            "moderator_time": 0,
                            "attempts": conflict_attempt["attempts"],
                        }
            final_conflict = best_conflict
        else:
            # Use the last conflict type attempted
            final_conflict = all_conflict_attempts[-1]
            final_conflict["editor_result"] = (
                final_conflict["attempts"][-1][0] if final_conflict["attempts"] else None
            )
            final_conflict["validation_result"] = (
                final_conflict["attempts"][-1][1] if final_conflict["attempts"] else None
            )
            final_conflict["editor_time"] = 0
            final_conflict["moderator_time"] = 0

        # Update result data
        result_data["doctor_result"] = final_conflict["conflict_result"]
        result_data["doctor_time"] = final_conflict["doctor_time"]
        result_data["conflict_type"] = final_conflict["conflict_type"]
        result_data["editor_result"] = final_conflict["editor_result"]
        result_data["editor_time"] = final_conflict["editor_time"]
        result_data["moderator_result"] = final_conflict["validation_result"]
        result_data["moderator_time"] = final_conflict["moderator_time"]
        result_data["all_conflict_attempts"] = all_conflict_attempts

        # Flatten all attempts for database saving
        all_attempts = []
        for conflict_attempt in all_conflict_attempts:
            for editor_result_attempt, validation_result_attempt, attempt_num in conflict_attempt[
                "attempts"
            ]:
                all_attempts.append((editor_result_attempt, validation_result_attempt, attempt_num))

        # Step 4: Save all attempts to database (no duplication, all attempts in one document pair)
        saved_attempts = []
        is_success = self._save_all_attempts_to_database(
            pair_id,
            document_pair,
            all_attempts,
            final_conflict["conflict_type"],
            all_conflict_attempts,  # Pass all conflict types data
        )

        # Track all attempts for analysis
        for editor_result_attempt, validation_result_attempt, attempt_num in all_attempts:
            saved_attempts.append(
                {
                    "attempt": attempt_num,
                    "saved": is_success,
                    "valid": validation_result_attempt.is_valid,
                    "overall_score": validation_result_attempt.overall_score,
                    "meets_threshold": validation_result_attempt.is_valid,
                }
            )

        result_data["all_attempts"] = saved_attempts
        result_data["success"] = validation_result.is_valid if validation_result else False

        result_data["processing_time"] = time.time() - start_time

        # Summary log
        status = "SUCCESS" if result_data["success"] else "FAILED"
        total_attempts = len(saved_attempts)
        successful_attempts = sum(1 for attempt in saved_attempts if attempt["valid"])
        threshold_attempts = sum(1 for attempt in saved_attempts if attempt["meets_threshold"])

        # Count successful conflict types
        successful_conflict_types = []
        for conflict_attempt in all_conflict_attempts:
            for editor_result_attempt, validation_result_attempt, attempt_num in conflict_attempt[
                "attempts"
            ]:
                if validation_result_attempt.is_valid:
                    successful_conflict_types.append(conflict_attempt["conflict_type"])
                    break  # Only count each conflict type once

        self.logger.info(
            f"Pair {pair_id}: {status} - {final_conflict['conflict_type']} conflict "
            f"(best of {len(successful_conflict_types)}/{len(conflict_types)} successful types), "
            f"{proposition_result[0].total_propositions + proposition_result[1].total_propositions}"
            f" propositions, {successful_attempts}/{total_attempts} valid,"
            f" {threshold_attempts}/{total_attempts} meet threshold"
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
                "doctor": {
                    "name": self.doctor_agent.name,
                    "conflict_types_available": list(
                        self.doctor_agent.list_all_conflict_types().keys()
                    ),
                },
                "editor": {"name": self.editor_agent.name},
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
