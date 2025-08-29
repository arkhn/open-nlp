import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Dict

import openai
import pandas as pd
from config import DATABASE_PATH
from models import DocumentPair, EditorResult, ValidationResult


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, name: str, client: openai.OpenAI, model: str, system_prompt: str = ""):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Process the input and return result"""
        pass

    def _execute_prompt(self, prompt: str, temperature: float = 0.7) -> str:
        """Execute a single prompt"""
        try:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            raise

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from API, handling potential formatting issues"""
        try:
            # Try to find JSON content in the response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON content found in response")

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response from {self.name}: {e}")

    def _truncate_document(self, text: str, max_length: int = 2000) -> str:
        """truncate document text to fit within prompt limits"""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


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
            cursor.execute(
                """
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
                    doc1_timestamp TEXT,
                    doc2_timestamp TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create table for processing history
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_pair_id TEXT,
                    agent_name TEXT,
                    result_data TEXT,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()

    def save_validated_documents(
        self,
        original_pair: DocumentPair,
        modified_docs: EditorResult,
        conflict_type: str,
        validation_result: ValidationResult,
    ) -> int:
        """Save validated documents to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO validated_documents
                (original_doc1_id, original_doc2_id, original_doc1_text, original_doc2_text, \
                    modified_doc1_text, modified_doc2_text, conflict_type, validation_score, \
                        changes_made, doc1_timestamp, doc2_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    original_pair.doc1_id,
                    original_pair.doc2_id,
                    original_pair.doc1_text,
                    original_pair.doc2_text,
                    modified_docs.modified_document1,
                    modified_docs.modified_document2,
                    conflict_type,
                    validation_result.validation_score,
                    modified_docs.changes_made,
                    str(original_pair.doc1_timestamp) if original_pair.doc1_timestamp else None,
                    str(original_pair.doc2_timestamp) if original_pair.doc2_timestamp else None,
                ),
            )

            doc_id = cursor.lastrowid
            conn.commit()

            return doc_id

    def log_processing_step(
        self, doc_pair_id: str, agent_name: str, result_data: Dict[str, Any], processing_time: float
    ):
        """Log a processing step to history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO processing_history (doc_pair_id, agent_name, \
                    result_data, processing_time)
                VALUES (?, ?, ?, ?)
            """,
                (doc_pair_id, agent_name, json.dumps(result_data), processing_time),
            )

            conn.commit()

    def get_validated_documents_count(self) -> int:
        """Get count of validated documents"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM validated_documents")
            return cursor.fetchone()[0]

    def save_to_parquet(self, output_path: str = "validated_documents.parquet"):
        """Save all validated documents to a parquet file"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM validated_documents"
            df = pd.read_sql_query(query, conn)
            df.to_parquet(output_path, index=False)
            print(f"Database saved to parquet file: {output_path}")
