import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import openai
import pandas as pd

from .models import DocumentPair, EditorResult, ValidationResult


@dataclass
class AnnotationValue:
    """Annotation value structure for Label Studio"""

    start: int
    end: int
    text: str
    labels: List[str]


@dataclass
class Annotation:
    """Single annotation in Label Studio format"""

    from_name: str
    to_name: str
    type: str
    moderator_score: int
    conflict_type: str
    value: AnnotationValue


@dataclass
class DocumentData:
    """Document data structure matching processed JSON format"""

    doc_1: str
    doc_2: str
    orig_doc_1: str
    orig_doc_2: str
    timestamp_1: Optional[str]
    timestamp_2: Optional[str]
    created_at: Optional[str]


@dataclass
class ConflictDataItem:
    """Complete conflict data item matching processed JSON format exactly"""

    data: DocumentData
    annotations: List[Dict[str, List[Annotation]]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "data": asdict(self.data),
            "annotations": [
                {"result": [asdict(ann) for ann in group["result"]]} for group in self.annotations
            ],
        }


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
    """Manager for JSON dataset operations"""

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.logger = logging.getLogger("DatasetManager")
        self.data = self._load_or_create_data()

    def _load_or_create_data(self) -> list:
        """Load existing JSON file or create empty list"""
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} records from {self.json_path}")
            return data
        else:
            self.logger.info("Created empty list for new dataset")
            return []

    def find_text_positions(self, text: str, excerpt: str):
        """Find start and end positions of excerpt in text"""
        if not excerpt or pd.isna(excerpt):
            return None, None

        start_pos = text.find(excerpt)
        if start_pos == -1:
            return None, None

        end_pos = start_pos + len(excerpt)
        return start_pos, end_pos

    def save_to_json(self, output_path: str = None):
        """Save current data to JSON file"""
        output_path = output_path or self.json_path
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved {len(self.data)} records to {output_path}")

    def save_to_parquet(self, output_path: str = None):
        """Save current data to parquet file (legacy method, calls save_to_json)"""
        self.save_to_json(output_path)

    def save_validated_documents(
        self,
        original_pair: DocumentPair,
        modified_docs: EditorResult,
        conflict_type: str,
        validation_result: ValidationResult,
    ) -> int:
        """Add validated documents to dataset in Label Studio format"""
        doc_id = len(self.data) + 1

        # Create document data
        doc_data = DocumentData(
            doc_1=modified_docs.modified_document1,
            doc_2=modified_docs.modified_document2,
            orig_doc_1=original_pair.doc1_text,
            orig_doc_2=original_pair.doc2_text,
            created_at=datetime.now().isoformat(),
            timestamp_1=str(original_pair.doc1_timestamp) if original_pair.doc1_timestamp else None,
            timestamp_2=str(original_pair.doc2_timestamp) if original_pair.doc2_timestamp else None,
        )

        # Create annotations list
        annotations = []

        # Add annotation for excerpt 1 if exists
        if (
            modified_docs.modified_excerpt_1
            and not pd.isna(modified_docs.modified_excerpt_1)
            and modified_docs.modified_excerpt_1.strip()
        ):
            start_pos, end_pos = self.find_text_positions(
                modified_docs.modified_document1, modified_docs.modified_excerpt_1
            )
            if start_pos is not None:
                annotation = Annotation(
                    from_name="labels_doc1",
                    to_name="doc_1",
                    type="labels",
                    moderator_score=validation_result.score,
                    conflict_type=conflict_type,
                    value=AnnotationValue(
                        start=start_pos,
                        end=end_pos,
                        text=modified_docs.modified_excerpt_1,
                        labels=["Conflict"],
                    ),
                )
                annotations.append(annotation)

        # Add annotation for excerpt 2 if exists
        if (
            modified_docs.modified_excerpt_2
            and not pd.isna(modified_docs.modified_excerpt_2)
            and modified_docs.modified_excerpt_2.strip()
        ):
            start_pos, end_pos = self.find_text_positions(
                modified_docs.modified_document2, modified_docs.modified_excerpt_2
            )
            if start_pos is not None:
                annotation = Annotation(
                    from_name="labels_doc2",
                    to_name="doc_2",
                    type="labels",
                    moderator_score=validation_result.score,
                    conflict_type=conflict_type,
                    value=AnnotationValue(
                        start=start_pos,
                        end=end_pos,
                        text=modified_docs.modified_excerpt_2,
                        labels=["Conflict"],
                    ),
                )
                annotations.append(annotation)

        # Create the complete item
        conflict_item = ConflictDataItem(data=doc_data, annotations=[{"result": annotations}])

        # Add to data list
        self.data.append(conflict_item.to_dict())
        self.logger.info(f"Added document with ID: {doc_id} to dataset")
        return doc_id

    def get_validated_documents_count(self) -> int:
        """Get count of validated documents"""
        return len(self.data)
