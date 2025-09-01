import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

import openai
import pandas as pd
from conflicts.models import DocumentPair, EditorResult, ValidationResult


class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, name: str, client: openai.OpenAI, model: str, cfg, system_prompt: str = ""):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")
        self.client = client
        self.model = model
        self.cfg = cfg
        self.max_length = cfg.model.max_length
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

    def _truncate_document(self, text: str) -> str:
        """truncate document text to fit within prompt limits"""
        if len(text) <= self.max_length:
            return text
        return text[: self.max_length] + "..."


class DatasetManager:
    """Manager for Parquet dataset operations"""

    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        self.logger = logging.getLogger("DatasetManager")
        self.df = self._load_or_create_dataframe()

    def _load_or_create_dataframe(self) -> pd.DataFrame:
        """Load existing parquet file or create empty DataFrame with proper schema"""
        if os.path.exists(self.parquet_path):
            df = pd.read_parquet(self.parquet_path)
            self.logger.info(f"Loaded {len(df)} records from {self.parquet_path}")
            return df
        else:
            # Create empty DataFrame with expected schema
            df = pd.DataFrame(
                columns=[
                    "id",
                    "original_doc1_id",
                    "original_doc2_id",
                    "original_doc1_text",
                    "original_doc2_text",
                    "modified_doc1_text",
                    "modified_doc2_text",
                    "conflict_type",
                    "score",
                    "changes_made",
                    "doc1_timestamp",
                    "doc2_timestamp",
                    "created_at",
                ]
            )
            self.logger.info("Created empty DataFrame for new dataset")
            return df

    def save_to_parquet(self, output_path: str = None):
        """Save current DataFrame to parquet file"""
        output_path = output_path or self.parquet_path
        self.df.to_parquet(output_path, index=False)
        self.logger.info(f"Saved {len(self.df)} records to {output_path}")

    def save_validated_documents(
        self,
        original_pair: DocumentPair,
        modified_docs: EditorResult,
        conflict_type: str,
        validation_result: ValidationResult,
    ) -> int:
        """Add validated documents to DataFrame"""
        doc_id = len(self.df) + 1

        new_record = {
            "id": doc_id,
            "original_doc1_id": original_pair.doc1_id,
            "original_doc2_id": original_pair.doc2_id,
            "original_doc1_text": original_pair.doc1_text,
            "original_doc2_text": original_pair.doc2_text,
            "modified_doc1_text": modified_docs.modified_document1,
            "modified_doc2_text": modified_docs.modified_document2,
            "conflict_type": conflict_type,
            "score": validation_result.score,
            "changes_made": modified_docs.changes_made,
            "doc1_timestamp": str(original_pair.doc1_timestamp)
            if original_pair.doc1_timestamp
            else None,
            "doc2_timestamp": str(original_pair.doc2_timestamp)
            if original_pair.doc2_timestamp
            else None,
            "created_at": datetime.now().isoformat(),
        }

        self.df = pd.concat([self.df, pd.DataFrame([new_record])], ignore_index=True)
        self.logger.info(f"Added document with ID: {doc_id} to dataset")
        return doc_id

    def get_validated_documents_count(self) -> int:
        """Get count of validated documents"""
        return len(self.df)
