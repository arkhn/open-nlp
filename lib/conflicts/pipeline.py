#!/usr/bin/env python3
"""
Clinical Document Conflict Pipeline - Main Controller
Orchestrates the three-agent workflow for clinical document editing and validation
"""

import time
from typing import List, Optional, Dict, Any, Tuple

from base import (
    PipelineLogger, DatabaseManager, DocumentPair, 
    ConflictResult, EditorResult, ValidationResult
)
from config import MAX_RETRY_ATTEMPTS, LOG_LEVEL, LOG_FILE
from data_loader import ClinicalDataLoader, create_sample_data_if_missing
from agents.doctor_agent import DoctorAgent
from agents.editor_agent import EditorAgent
from agents.moderator_agent import ModeratorAgent


class ClinicalConflictPipeline:
    """
    Main pipeline controller that manages the three-agent workflow
    """
    
    def __init__(self, 
                 max_retries: int = MAX_RETRY_ATTEMPTS,
                 min_validation_score: int = 70):
        """
        Initialize the pipeline
        
        Args:
            max_retries: Maximum number of retry attempts for validation failures
            min_validation_score: Minimum score required for validation approval
        """
        # Setup logging
        self.logger = PipelineLogger.setup_logging(LOG_LEVEL, LOG_FILE)
        
        # Initialize components
        self.max_retries = max_retries
        self.db_manager = DatabaseManager()
        
        # Initialize agents
        self.doctor_agent = DoctorAgent()
        self.editor_agent = EditorAgent()
        self.moderator_agent = ModeratorAgent(min_validation_score=min_validation_score)
        
        # Initialize data loader
        try:
            self.data_loader = ClinicalDataLoader()
        except FileNotFoundError:
            self.logger.warning("Clinical data not found, attempting to create sample data")
            create_sample_data_if_missing()
            self.data_loader = ClinicalDataLoader()
        
        self.logger.info("Clinical Conflict Pipeline initialized successfully")
        
        # Log data statistics
        stats = self.data_loader.get_data_statistics()
        self.logger.info(f"Loaded dataset with {stats['total_documents']} documents from {stats['unique_subjects']} subjects")
    
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
            'pair_id': pair_id,
            'doc1_id': document_pair.doc1_id,
            'doc2_id': document_pair.doc2_id,
            'success': False,
            'attempts': 0,
            'final_validation_score': 0,
            'conflict_type': None,
            'processing_time': 0,
            'error': None
        }
        
        try:
            # Step 1: Doctor Agent Analysis
            self.logger.info("Step 1: Doctor Agent analyzing documents...")
            doctor_start_time = time.time()
            
            conflict_result = self.doctor_agent.process(document_pair)
            
            doctor_processing_time = time.time() - doctor_start_time
            self.db_manager.log_processing_step(
                pair_id, 
                "Doctor", 
                {
                    'conflict_type': conflict_result.conflict_type,
                    'reasoning': conflict_result.reasoning
                },
                doctor_processing_time
            )
            
            result_data['conflict_type'] = conflict_result.conflict_type
            
            # Step 2: Editor and Moderator Loop
            validation_result = None
            editor_result = None
            attempt = 0
            
            while attempt < self.max_retries:
                attempt += 1
                result_data['attempts'] = attempt
                
                self.logger.info(f"Step 2: Editor Agent modifying documents (attempt {attempt}/{self.max_retries})...")
                editor_start_time = time.time()
                
                editor_result = self.editor_agent.process(document_pair, conflict_result)
                
                editor_processing_time = time.time() - editor_start_time
                self.db_manager.log_processing_step(
                    pair_id,
                    "Editor",
                    {
                        'attempt': attempt,
                        'changes_made': editor_result.changes_made
                    },
                    editor_processing_time
                )
                
                self.logger.info(f"Step 3: Moderator Agent validating modifications (attempt {attempt}/{self.max_retries})...")
                moderator_start_time = time.time()
                
                validation_result = self.moderator_agent.process(
                    document_pair, 
                    editor_result, 
                    conflict_result.conflict_type
                )
                
                moderator_processing_time = time.time() - moderator_start_time
                self.db_manager.log_processing_step(
                    pair_id,
                    "Moderator",
                    {
                        'attempt': attempt,
                        'is_valid': validation_result.is_valid,
                        'validation_score': validation_result.validation_score,
                        'issues_found': validation_result.issues_found
                    },
                    moderator_processing_time
                )
                
                result_data['final_validation_score'] = validation_result.validation_score
                
                if validation_result.is_valid:
                    self.logger.info(f"Validation successful on attempt {attempt}")
                    break
                else:
                    self.logger.warning(f"Validation failed on attempt {attempt}: {validation_result.feedback}")
                    
                    if attempt < self.max_retries:
                        self.logger.info(f"Retrying... ({attempt + 1}/{self.max_retries})")
                        # Brief pause before retry
                        time.sleep(1)
            
            # Step 4: Save to database if validation passed
            if validation_result and validation_result.is_valid:
                self.logger.info("Saving validated documents to database...")
                
                doc_id = self.db_manager.save_validated_documents(
                    document_pair,
                    editor_result,
                    conflict_result.conflict_type,
                    validation_result
                )
                
                result_data['success'] = True
                result_data['database_id'] = doc_id
                
                self.logger.info(f"Document pair {pair_id} processed successfully (DB ID: {doc_id})")
            else:
                result_data['success'] = False
                result_data['error'] = f"Validation failed after {self.max_retries} attempts"
                self.logger.error(f"Document pair {pair_id} failed validation after {self.max_retries} attempts")
        
        except Exception as e:
            result_data['success'] = False
            result_data['error'] = str(e)
            self.logger.error(f"Error processing document pair {pair_id}: {e}")
        
        finally:
            result_data['processing_time'] = time.time() - start_time
        
        return result_data['success'], result_data
    
    def process_batch(self, 
                      batch_size: int = 5,
                      same_subject: bool = False,
                      category_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a batch of document pairs
        
        Args:
            batch_size: Number of document pairs to process
            same_subject: If True, try to pair documents from the same subject
            category_filter: List of categories to filter documents by
            
        Returns:
            Dictionary with batch processing results
        """
        self.logger.info(f"Starting batch processing of {batch_size} document pairs")
        
        batch_start_time = time.time()
        
        # Get document pairs
        try:
            document_pairs = self.data_loader.get_random_document_pairs(
                count=batch_size,
                same_subject=same_subject,
                category_filter=category_filter
            )
        except Exception as e:
            self.logger.error(f"Failed to load document pairs: {e}")
            return {
                'success': False,
                'error': f"Data loading failed: {e}",
                'processed': 0,
                'failed': 0
            }
        
        # Process each pair
        results = []
        successful = 0
        failed = 0
        
        for i, doc_pair in enumerate(document_pairs, 1):
            self.logger.info(f"Processing pair {i}/{batch_size}")
            
            success, result_data = self.process_document_pair(doc_pair)
            results.append(result_data)
            
            if success:
                successful += 1
            else:
                failed += 1
            
            # Log progress
            if i % 5 == 0 or i == len(document_pairs):
                self.logger.info(f"Batch progress: {i}/{batch_size} processed, {successful} successful, {failed} failed")
        
        batch_time = time.time() - batch_start_time
        
        # Calculate statistics
        batch_summary = {
            'success': True,
            'total_pairs': len(document_pairs),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(document_pairs) * 100,
            'total_processing_time': batch_time,
            'average_processing_time': batch_time / len(document_pairs),
            'results': results
        }
        
        self.logger.info(f"Batch processing completed: {successful}/{len(document_pairs)} successful ({batch_summary['success_rate']:.1f}%)")
        self.logger.info(f"Total time: {batch_time:.2f}s, Average per pair: {batch_summary['average_processing_time']:.2f}s")
        
        return batch_summary
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline performance
        
        Returns:
            Dictionary with pipeline statistics
        """
        total_validated = self.db_manager.get_validated_documents_count()
        data_stats = self.data_loader.get_data_statistics()
        
        return {
            'validated_documents': total_validated,
            'dataset_statistics': data_stats,
            'agents': {
                'doctor': {
                    'name': self.doctor_agent.name,
                    'conflict_types_available': list(self.doctor_agent.list_all_conflict_types().keys())
                },
                'editor': {
                    'name': self.editor_agent.name
                },
                'moderator': {
                    'name': self.moderator_agent.name,
                    'min_validation_score': self.moderator_agent.min_validation_score
                }
            },
            'configuration': {
                'max_retries': self.max_retries,
                'database_path': self.db_manager.db_path
            }
        }
    
    def test_single_conflict_type(self, conflict_type: str, count: int = 3) -> Dict[str, Any]:
        """
        Test the pipeline with a specific conflict type by forcing the Doctor Agent's decision
        
        Args:
            conflict_type: The conflict type to test
            count: Number of document pairs to test with
            
        Returns:
            Test results
        """
        self.logger.info(f"Testing pipeline with forced conflict type: {conflict_type}")
        
        # Validate conflict type exists
        available_types = self.doctor_agent.list_all_conflict_types()
        if conflict_type not in available_types:
            return {
                'success': False,
                'error': f"Unknown conflict type '{conflict_type}'. Available: {list(available_types.keys())}"
            }
        
        # Get test document pairs
        document_pairs = self.data_loader.get_random_document_pairs(count=count)
        
        test_results = []
        successful = 0
        
        for doc_pair in document_pairs:
            # Create a forced conflict result
            conflict_info = available_types[conflict_type]
            forced_conflict = ConflictResult(
                conflict_type=conflict_type,
                reasoning=f"Testing forced conflict type: {conflict_info['name']}",
                modification_instructions=f"Create a {conflict_info['name']} conflict between these documents based on the description: {conflict_info['description']}"
            )
            
            # Process through Editor and Moderator only
            try:
                editor_result = self.editor_agent.process(doc_pair, forced_conflict)
                validation_result = self.moderator_agent.process(doc_pair, editor_result, conflict_type)
                
                test_result = {
                    'pair_id': f"{doc_pair.doc1_id}_{doc_pair.doc2_id}",
                    'conflict_type': conflict_type,
                    'editor_success': True,
                    'validation_success': validation_result.is_valid,
                    'validation_score': validation_result.validation_score,
                    'issues_found': validation_result.issues_found
                }
                
                if validation_result.is_valid:
                    successful += 1
                
            except Exception as e:
                test_result = {
                    'pair_id': f"{doc_pair.doc1_id}_{doc_pair.doc2_id}",
                    'conflict_type': conflict_type,
                    'editor_success': False,
                    'validation_success': False,
                    'error': str(e)
                }
            
            test_results.append(test_result)
        
        return {
            'success': True,
            'conflict_type_tested': conflict_type,
            'total_pairs': count,
            'successful_validations': successful,
            'success_rate': successful / count * 100,
            'results': test_results
        }
