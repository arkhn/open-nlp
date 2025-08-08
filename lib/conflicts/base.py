#!/usr/bin/env python3
"""
Base classes and utilities for the Clinical Document Pipeline
"""

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import openai
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL, DATABASE_PATH


@dataclass
class DocumentPair:
    """Represents a pair of clinical documents"""
    doc1_id: str
    doc2_id: str
    doc1_text: str
    doc2_text: str
    subject_id: Optional[str] = None
    category1: Optional[str] = None
    category2: Optional[str] = None


@dataclass
class ConflictResult:
    """Result from the Doctor Agent"""
    conflict_type: str
    reasoning: str
    modification_instructions: str


@dataclass
class EditorResult:
    """Result from the Editor Agent"""
    modified_document1: str
    modified_document2: str
    changes_made: str


@dataclass
class ValidationResult:
    """Result from the Moderator Agent"""
    is_valid: bool
    validation_score: int
    feedback: str
    issues_found: list
    approval_reasoning: str


class GroqAPIClient:
    """Client for interacting with Groq API"""
    
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Initialize OpenAI client with Groq configuration
        self.client = openai.OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL
        )
        self.model = GROQ_MODEL
    
    def call_api(self, prompt: str, temperature: float = 0.1, max_tokens: int = 4000) -> str:
        """Make a call to the Groq API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant specialized in clinical document analysis. Always respond in valid JSON format when requested."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Groq API call failed: {e}")
            raise


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.client = GroqAPIClient()
        self.logger = logging.getLogger(f"Agent.{name}")
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process the input and return result"""
        pass
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from API, handling potential formatting issues"""
        try:
            # Try to find JSON content in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON content found in response")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"Response content: {response}")
            raise ValueError(f"Invalid JSON response from {self.name}: {e}")


class DatabaseManager:
    """Manager for SQLite database operations"""
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create table for validated documents
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validated_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_doc1_id TEXT,
                    original_doc2_id TEXT,
                    original_doc1_text TEXT,
                    original_doc2_text TEXT,
                    modified_doc1_text TEXT,
                    modified_doc2_text TEXT,
                    conflict_type TEXT,
                    validation_score INTEGER,
                    changes_made TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create table for processing history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_pair_id TEXT,
                    agent_name TEXT,
                    result_data TEXT,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def save_validated_documents(self, 
                                original_pair: DocumentPair,
                                modified_docs: EditorResult,
                                conflict_type: str,
                                validation_result: ValidationResult) -> int:
        """Save validated documents to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO validated_documents 
                (original_doc1_id, original_doc2_id, original_doc1_text, original_doc2_text,
                 modified_doc1_text, modified_doc2_text, conflict_type, validation_score, changes_made)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                original_pair.doc1_id,
                original_pair.doc2_id,
                original_pair.doc1_text,
                original_pair.doc2_text,
                modified_docs.modified_document1,
                modified_docs.modified_document2,
                conflict_type,
                validation_result.validation_score,
                modified_docs.changes_made
            ))
            
            doc_id = cursor.lastrowid
            conn.commit()
            
            return doc_id
    
    def log_processing_step(self, doc_pair_id: str, agent_name: str, result_data: Dict[str, Any], processing_time: float):
        """Log a processing step to history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO processing_history (doc_pair_id, agent_name, result_data, processing_time)
                VALUES (?, ?, ?, ?)
            """, (doc_pair_id, agent_name, json.dumps(result_data), processing_time))
            
            conn.commit()
    
    def get_validated_documents_count(self) -> int:
        """Get count of validated documents"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM validated_documents")
            return cursor.fetchone()[0]


class PipelineLogger:
    """Custom logger for the pipeline"""
    
    @staticmethod
    def setup_logging(log_level: str = "INFO", log_file: str = "pipeline.log"):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Create logger for the pipeline
        logger = logging.getLogger("ClinicalPipeline")
        return logger


def format_conflict_types_for_prompt(conflict_types: Dict) -> str:
    """Format conflict types dictionary for use in prompts"""
    formatted_types = []
    for key, conflict_type in conflict_types.items():
        examples_str = "\n  ".join([f"- {example}" for example in conflict_type.examples])
        formatted_types.append(f"""
{key}: {conflict_type.name}
Description: {conflict_type.description}
Examples:
  {examples_str}
""")
    return "\n".join(formatted_types)
