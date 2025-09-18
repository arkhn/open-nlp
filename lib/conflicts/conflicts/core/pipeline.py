import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import openai
from dotenv import load_dotenv
from omegaconf import DictConfig

from ..agents.doctor_agent import DoctorAgent
from ..agents.editor_agent import EditorAgent
from ..agents.moderator_agent import ModeratorAgent
from ..agents.proposition_agent import PropositionAgent
from .base import DatasetManager
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

        # Setup logging - Hydra already configures root logger
        # Just get a logger for this module
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.dataset_manager = DatasetManager("validated_documents.json")

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
        Save validated documents to database

        Args:
            pair_id: Document pair ID for logging
            document_pair: The document pair
            editor_result: Result from editor agent
            conflict_type: Type of conflict identified
            validation_result: Result from moderator agent

        Returns:
            True if saved successfully, False otherwise
        """
        if validation_result.is_valid:
            doc_id = self.dataset_manager.save_validated_documents(
                document_pair, editor_result, conflict_type, validation_result
            )
            self.logger.info(f"Document pair {pair_id} processed successfully (DB ID: {doc_id})")
            return True
        else:
            self.logger.warning(
                f"Document pair {pair_id} failed validation after {self.max_retries} attempts"
            )
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
            "conflict_pairs": 0,
            "successful_pairs": 0,
            "total_pairs": 0,
            "processing_time": 0,
            "proposition_result": None,
            "doctor_result": None,
            "conflict_pair_results": [],
            "proposition_time": 0,
            "doctor_time": 0,
        }

        # Step 1: Proposition Agent decomposes documents into propositions
        start_prop_time = time.time()
        proposition_result = self.proposition_agent.decompose_document_pair(
            document_pair.doc1_text, document_pair.doc2_text
        )
        proposition_time = time.time() - start_prop_time
        result_data["proposition_result"] = proposition_result
        result_data["proposition_time"] = proposition_time

        # Step 2: Doctor Agent identifies conflict pairs using propositions
        conflict_result, doctor_time = self._execute_agent(
            self.doctor_agent, document_pair, proposition_result[0], proposition_result[1]
        )
        result_data["doctor_result"] = conflict_result
        result_data["doctor_time"] = doctor_time
        result_data["conflict_pairs"] = len(conflict_result.conflict_pairs)

        # Step 3: Process each conflict pair through Editor and Moderator agents
        all_results = []
        successful_pairs = 0

        for i, conflict_pair in enumerate(conflict_result.conflict_pairs):
            self.logger.info(
                f"Processing conflict pair {i+1}/{len(conflict_result.conflict_pairs)}:"
                f" {conflict_pair.conflict_type}"
            )

            pair_result = {
                "conflict_type": conflict_pair.conflict_type,
                "success": False,
                "editor_result": None,
                "moderator_result": None,
                "editor_time": 0,
                "moderator_time": 0,
                "attempts": 0,
            }

            validation_result = None
            editor_result = None

            for attempt in range(1, self.max_retries + 1):
                pair_result["attempts"] = attempt

                # Execute editor agent
                editor_result, editor_time = self._execute_agent(
                    self.editor_agent, document_pair, conflict_pair
                )
                pair_result["editor_result"] = editor_result
                pair_result["editor_time"] = editor_time

                # Check if editor agent failed to create modifications
                if "Failed to create conflict" in editor_result.changes_made:
                    self.logger.warning(
                        f"Editor agent failed to create modifications for "
                        f"{conflict_pair.conflict_type}, skipping moderator validation"
                    )
                    validation_result = ValidationResult(
                        is_valid=False,
                        score=1,
                        reasoning="Editor agent failed to modify - no changes to validate",
                    )
                    pair_result["moderator_result"] = validation_result
                    pair_result["moderator_time"] = 0
                    break

                # Execute moderator agent for validation
                validation_result, moderator_time = self._execute_agent(
                    self.moderator_agent, document_pair, editor_result, conflict_pair.conflict_type
                )
                pair_result["moderator_result"] = validation_result
                pair_result["moderator_time"] = moderator_time

                self.logger.info(
                    f"Attempt {attempt}: {conflict_pair.conflict_type} conflict, "
                    f"valid={validation_result.is_valid}, score={validation_result.score}/5"
                )

                if validation_result.is_valid:
                    pair_result["success"] = True
                    successful_pairs += 1
                    break

                if attempt < self.max_retries:
                    self.logger.warning(
                        f"Validation failed for {conflict_pair.conflict_type}, retrying..."
                    )
                    time.sleep(1)

            # Save to database if validation passed
            if validation_result and validation_result.is_valid:
                is_success = self._save_to_database(
                    f"{pair_id}_{conflict_pair.conflict_type}",
                    document_pair,
                    editor_result,
                    conflict_pair.conflict_type,
                    validation_result,
                )
                pair_result["success"] = is_success

            all_results.append(pair_result)

        # Update result data with all conflict pair results
        result_data["conflict_pair_results"] = all_results
        result_data["successful_pairs"] = successful_pairs
        result_data["total_pairs"] = len(conflict_result.conflict_pairs)
        result_data["success"] = successful_pairs > 0  # Success if at least one pair succeeded

        result_data["processing_time"] = time.time() - start_time

        # Summary log
        status = "SUCCESS" if result_data["success"] else "FAILED"
        self.logger.info(
            f"{pair_id}: {status} - {result_data['successful_pairs']}/{result_data['total_pairs']} "
            "conflict pairs successful, "
            f"{proposition_result[0].total_propositions + proposition_result[1].total_propositions}"
            f" propositions"
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
                    "conflict_pairs": 0,
                    "successful_pairs": 0,
                    "total_pairs": 0,
                    "processing_time": 0,
                    "proposition_result": None,
                    "doctor_result": None,
                    "conflict_pair_results": [],
                    "proposition_time": 0,
                    "doctor_time": 0,
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
