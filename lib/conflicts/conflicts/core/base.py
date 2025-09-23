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
    moderator_score: float
    moderator_reasoning: str
    conflict_type: str
    value: AnnotationValue
    clinical_plausibility_score: float = 1.0
    temporal_appropriateness_score: float = 1.0
    clinical_significance_score: float = 1.0
    is_valid: bool = False
    retry_attempt: int = 1


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
    moderator_score: Optional[float] = None
    moderator_reasoning: Optional[str] = None
    conflict_type: Optional[str] = None
    all_conflict_types: Optional[Dict[str, Any]] = None
    final_selected_type: Optional[str] = None


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
        # Create document data using helper method
        doc_data = self._create_document_data(
            modified_docs, original_pair, validation_result, conflict_type
        )

        # Create annotations using helper methods
        annotations = []
        for excerpt_num in [1, 2]:
            annotation = self._create_annotation_for_excerpt(
                modified_docs, validation_result, conflict_type, excerpt_num
            )
            if annotation:
                annotations.append(annotation)

        # Create the complete item and save
        conflict_item = ConflictDataItem(data=doc_data, annotations=[{"result": annotations}])
        return self.save_item(conflict_item)

    def _create_document_data(
        self,
        modified_docs: EditorResult,
        original_pair: DocumentPair,
        validation_result: ValidationResult,
        conflict_type: str,
    ) -> DocumentData:
        """Create DocumentData object with standard fields"""
        return DocumentData(
            doc_1=modified_docs.modified_document1,
            doc_2=modified_docs.modified_document2,
            orig_doc_1=original_pair.doc1_text,
            orig_doc_2=original_pair.doc2_text,
            created_at=datetime.now().isoformat(),
            timestamp_1=str(original_pair.doc1_timestamp) if original_pair.doc1_timestamp else None,
            timestamp_2=str(original_pair.doc2_timestamp) if original_pair.doc2_timestamp else None,
            moderator_score=validation_result.overall_score,
            moderator_reasoning=validation_result.reasoning,
            conflict_type=conflict_type,
        )

    def _create_annotation_for_excerpt(
        self,
        modified_docs: EditorResult,
        validation_result: ValidationResult,
        conflict_type: str,
        excerpt_num: int,
        retry_attempt: int = 1,
    ) -> Optional[Annotation]:
        """Create annotation for a specific excerpt if it exists"""
        if excerpt_num == 1:
            excerpt = modified_docs.modified_excerpt_1
            document = modified_docs.modified_document1
            to_name = "doc_1"
            from_name = (
                f"labels_doc1_attempt_{retry_attempt}" if retry_attempt > 1 else "labels_doc1"
            )
        else:
            excerpt = modified_docs.modified_excerpt_2
            document = modified_docs.modified_document2
            to_name = "doc_2"
            from_name = (
                f"labels_doc2_attempt_{retry_attempt}" if retry_attempt > 1 else "labels_doc2"
            )

        if not excerpt or pd.isna(excerpt) or not excerpt.strip():
            return None

        start_pos, end_pos = self.find_text_positions(document, excerpt)
        if start_pos is None:
            return None

        return Annotation(
            from_name=from_name,
            to_name=to_name,
            type="labels",
            moderator_score=validation_result.overall_score,
            moderator_reasoning=validation_result.reasoning,
            conflict_type=conflict_type,
            clinical_plausibility_score=validation_result.clinical_plausibility_score,
            temporal_appropriateness_score=validation_result.temporal_appropriateness_score,
            clinical_significance_score=validation_result.clinical_significance_score,
            is_valid=validation_result.is_valid,
            retry_attempt=retry_attempt,
            value=AnnotationValue(
                start=start_pos,
                end=end_pos,
                text=excerpt,
                labels=["Conflict"],
            ),
        )

    def save_item(self, conflict_item: ConflictDataItem) -> int:
        """Add a conflict item to the dataset"""
        doc_id = len(self.data) + 1
        self.data.append(conflict_item.to_dict())
        self.logger.info(f"Added document with ID: {doc_id} to dataset")
        return doc_id

    def get_validated_documents_count(self) -> int:
        """Get count of validated documents"""
        return len(self.data)
