import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import openai
from agents.doctor_agent import DoctorAgent
from agents.editor_agent import EditorAgent
from agents.moderator_agent import ModeratorAgent
from base import DatabaseManager
from config import (
    API_KEY,
    BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MIN_VALIDATION_SCORE,
    MODEL,
)
from data_loader import DataLoader
from models import DocumentPair

LOG_LEVEL = "INFO"
LOG_FILE = "pipeline.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
PIPELINE_LOGGER_NAME = "ClinicalPipeline"


class Pipeline:
    """
    Main pipeline controller that manages the three-agent workflow
    """

    def __init__(self, max_retries: int = None, min_validation_score: int = None):
        """
        Initialize the pipeline

        Args:
            max_retries: Maximum number of retry attempts for validation failures
            min_validation_score: Minimum score required for validation approval
        """
        # Use default values from config if not provided
        self.max_retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
        min_validation_score = (
            min_validation_score
            if min_validation_score is not None
            else DEFAULT_MIN_VALIDATION_SCORE
        )

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL.upper()),
            format=LOG_FORMAT,
            handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(PIPELINE_LOGGER_NAME)

        # Initialize components
        self.db_manager = DatabaseManager()

        # Create shared OpenAI client
        self.client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

        # Initialize agents with shared client
        self.doctor_agent = DoctorAgent(self.client, MODEL)
        self.editor_agent = EditorAgent(self.client, MODEL)
        self.moderator_agent = ModeratorAgent(
            self.client, MODEL, min_validation_score=min_validation_score
        )

        self.data_loader = DataLoader()

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
            doc_id = self.db_manager.save_validated_documents(
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
            "conflict_type": None,
            "processing_time": 0,
            "doctor_result": None,
            "editor_result": None,
            "moderator_result": None,
            "doctor_time": 0,
            "editor_time": 0,
            "moderator_time": 0,
        }

        # Step 1: Doctor Agent identifies conflict type
        conflict_result, doctor_time = self._execute_agent(self.doctor_agent, document_pair)
        result_data["doctor_result"] = conflict_result
        result_data["doctor_time"] = doctor_time
        result_data["conflict_type"] = conflict_result.conflict_type

        # Step 2: Editor and Moderator agents with retry logic for editor only
        validation_result = None
        editor_result = None

        for attempt in range(1, self.max_retries + 1):
            # Execute editor agent
            editor_result, editor_time = self._execute_agent(
                self.editor_agent, document_pair, conflict_result
            )
            result_data["editor_result"] = editor_result
            result_data["editor_time"] = editor_time

            # Execute moderator agent for validation
            validation_result, moderator_time = self._execute_agent(
                self.moderator_agent, document_pair, editor_result, conflict_result.conflict_type
            )
            result_data["moderator_result"] = validation_result
            result_data["moderator_time"] = moderator_time

            if validation_result.is_valid:
                self.logger.info(f"Validation successful on attempt {attempt}")
                result_data["success"] = True
                break

            if attempt < self.max_retries:
                self.logger.warning(f"Validation failed on attempt {attempt}, retrying editor...")
                time.sleep(1)

        # Step 3: Save to database if validation passed
        if validation_result and validation_result.is_valid:
            is_success = self._save_to_database(
                pair_id,
                document_pair,
                editor_result,
                conflict_result.conflict_type,
                validation_result,
            )
            result_data["success"] = is_success

        result_data["processing_time"] = time.time() - start_time
        return result_data["success"], result_data

    def process_batch(
        self,
        batch_size: int = 5,
        category_filter: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process a batch of document pairs

        Args:
            batch_size: Number of document pairs to process
            category_filter: List of categories to filter documents by

        Returns:
            Dictionary with batch processing results
        """
        self.logger.info(f"Starting batch processing of {batch_size} document pairs")

        batch_start_time = time.time()
        document_pairs = self.data_loader.get_random_document_pairs(
            count=batch_size, category_filter=category_filter
        )

        results = []
        successful = 0

        for doc_pair in document_pairs:
            success, result_data = self.process_document_pair(doc_pair)
            results.append(result_data)

            if success:
                successful += 1

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
        total_validated = self.db_manager.get_validated_documents_count()
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
                    "min_validation_score": self.moderator_agent.min_validation_score,
                },
            },
            "configuration": {
                "max_retries": self.max_retries,
                "database_path": self.db_manager.db_path,
            },
        }
