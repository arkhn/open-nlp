import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import openai
from agents.doctor_agent import DoctorAgent
from agents.editor_agent import EditorAgent
from agents.moderator_agent import ModeratorAgent
from base import DatabaseManager
from config import API_KEY, BASE_URL, LOG_FILE, LOG_LEVEL, MODEL
from data_loader import DataLoader
from models import DocumentPair


class Pipeline:
    """
    Main pipeline controller that manages the three-agent workflow
    """

    def __init__(self, max_retries: int = 0, min_validation_score: int = 70):
        """
        Initialize the pipeline

        Args:
            max_retries: Maximum number of retry attempts for validation failures
            min_validation_score: Minimum score required for validation approval
        """
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
        )
        self.logger = logging.getLogger("ClinicalPipeline")

        # Initialize components
        self.max_retries = max_retries
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
            f"Loaded dataset with {stats['total_documents']} documents \
                from {stats['unique_subjects']} subjects"
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
        }

        conflict_result, doctor_time = self._execute_agent(self.doctor_agent, document_pair)

        doctor_log_data = {
            "conflict_type": conflict_result.conflict_type,
            "reasoning": conflict_result.reasoning,
        }
        self.db_manager.log_processing_step(pair_id, "Doctor", doctor_log_data, doctor_time)

        result_data["conflict_type"] = conflict_result.conflict_type

        validation_result = None
        editor_result = None

        for attempt in range(1, self.max_retries + 1):
            editor_result, editor_time = self._execute_agent(
                self.editor_agent, document_pair, conflict_result
            )

            editor_log_data = {"attempt": attempt, "changes_made": editor_result.changes_made}
            self.db_manager.log_processing_step(pair_id, "Editor", editor_log_data, editor_time)

            validation_result, moderator_time = self._execute_agent(
                self.moderator_agent, document_pair, editor_result, conflict_result.conflict_type
            )

            moderator_log_data = {
                "attempt": attempt,
                "is_valid": validation_result.is_valid,
                "validation_score": validation_result.validation_score,
                "issues_found": validation_result.issues_found,
            }
            self.db_manager.log_processing_step(
                pair_id, "Moderator", moderator_log_data, moderator_time
            )

            if validation_result.is_valid:
                self.logger.info(f"Validation successful on attempt {attempt}")
                break

            if attempt < self.max_retries:
                self.logger.warning(f"Validation failed on attempt {attempt}, retrying...")
                time.sleep(1)

        # Step 3: Save to database if validation passed
        is_success = self._save_to_database(
            pair_id, document_pair, editor_result, conflict_result.conflict_type, validation_result
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
        total_pairs = len(document_pairs)

        self.logger.info(
            f"Batch completed: {successful}/{total_pairs} \
                successful ({successful/total_pairs*100:.1f}%)"
        )

        return {
            "total_pairs": total_pairs,
            "successful": successful,
            "failed": total_pairs - successful,
            "success_rate": successful / total_pairs * 100,
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
