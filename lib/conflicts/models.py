from dataclasses import dataclass
from typing import Optional


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