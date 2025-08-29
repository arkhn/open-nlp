import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
BASE_URL = os.getenv("BASE_URL", "https://api.groq.com/openai/v1")

# Data Configuration
DATA_PATH = "data/mimic-iii-verifact-bhc.parquet"
DATABASE_PATH = "validated_documents.db"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "pipeline.log"

# Pipeline Configuration
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
DEFAULT_MIN_VALIDATION_SCORE = 70
DEFAULT_MAX_RETRIES = 0

# Logging Format Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
PIPELINE_LOGGER_NAME = "ClinicalPipeline"

# Text Processing Configuration
MIN_SENTENCE_LENGTH = 10
MIN_TARGET_TEXT_LENGTH = 20
MIN_TEXT_SEGMENT_LENGTH = 30
MIN_GENERAL_TEXT_LENGTH = 50
SIMILARITY_THRESHOLD = 0.6
WORD_OVERLAP_THRESHOLD = 2
MAX_SUGGESTIONS_PER_CATEGORY = 3
MAX_GENERAL_SUGGESTIONS = 5

# Edit Operation Types
EDIT_OPERATIONS = {"DELETE": "delete", "INSERT_AFTER": "insert_after", "REPLACE": "replace"}

# Validation Rules
VALIDATION_RULES = {
    "MIN_TARGET_LENGTH": MIN_TARGET_TEXT_LENGTH,
    "TRUNCATION_INDICATORS": ["...", "n..."],
    "REQUIRED_FIELDS": ["op", "target_text"],
    "REPLACE_REQUIRES_REPLACEMENT": True,
}

# Medical Category Keywords and Patterns
MEDICAL_CATEGORIES = {
    "assessment": ["assessment", "diagnosis", "impression", "findings", "clinical picture"],
    "vital_signs": [
        "vital signs",
        "blood pressure",
        "heart rate",
        "pulse",
        "temperature",
        "respiratory rate",
        "oxygen saturation",
        "sp02",
        "bp",
        "hr",
        "temp",
    ],
    "laboratory": [
        "lab",
        "laboratory",
        "test",
        "result",
        "value",
        "level",
        "ng/ml",
        "mg/dl",
        "mmol/l",
        "units",
        "reference range",
    ],
    "medication": [
        "medication",
        "drug",
        "prescription",
        "dose",
        "mg",
        "tablet",
        "capsule",
        "injection",
        "iv",
        "oral",
        "subcutaneous",
    ],
    "procedures": [
        "procedure",
        "surgery",
        "operation",
        "intervention",
        "treatment",
        "catheterization",
        "intubation",
        "ventilation",
    ],
    "symptoms": [
        "symptoms",
        "complaints",
        "pain",
        "shortness of breath",
        "dyspnea",
        "nausea",
        "vomiting",
        "dizziness",
        "weakness",
    ],
}

# Edit Operation Keywords for Suggestions
EDIT_SUGGESTION_KEYWORDS = {
    "medical_condition": ["diagnosis", "condition", "disease", "syndrome", "assessment"],
    "medication": ["medication", "drug", "prescription", "dose", "mg", "tablet"],
    "vital_signs": ["blood pressure", "heart rate", "temperature", "pulse", "respiratory"],
    "lab_values": ["lab", "laboratory", "test", "result", "value", "level"],
    "procedures": ["procedure", "surgery", "operation", "intervention", "treatment"],
}

# Sentence Splitting Configuration
SENTENCE_SPLIT_PATTERN = r"(?<=[.!?])\s+"

# JSON Parsing Configuration
JSON_PARSING = {
    "MARKDOWN_CODE_BLOCK": "```json",
    "REQUIRED_FIELDS": ["doc1", "doc2", "conflict_type"],
    "RESPONSE_PREVIEW_LENGTH": 200,
}

# Error Messages
ERROR_MESSAGES = {
    "TARGET_TEXT_NOT_FOUND": "Target text not found in document",
    "TARGET_TEXT_TOO_SHORT": "Target text too short to be reliable",
    "TARGET_TEXT_TRUNCATED": "Target text appears to be truncated",
    "INVALID_OPERATION_TYPE": "Unknown operation type",
    "REPLACE_MISSING_REPLACEMENT": "Replace operation requires replacement_text",
    "JSON_PARSE_FAILED": "Failed to parse response as JSON",
    "MISSING_REQUIRED_FIELDS": "Response missing required fields: doc1, doc2, conflict_type",
}


@dataclass
class ConflictType:
    """Represents a type of clinical conflict"""

    name: str
    description: str
    examples: List[str]


# Conflict Types and Descriptions
CONFLICT_TYPES = {
    "opposition": ConflictType(
        name="Opposition Conflicts",
        description="Contradictory findings about the same clinical entity",
        examples=[
            "Normal vs abnormal findings of same body structure: Left breast: \
                Unremarkable <> Left breast demonstrates persistent circumscribed masses",
            "Negative vs positive statements: No cardiopulmonary disease \
                <> Bibasilar atelectasis",
            "Lab/vital sign interpretation: Low blood sugar at admission \
                <> Patient admitted with hyperglycemia",
            "Opposite disorders: Hypernatremia <> Hyponatremia",
            "Sex information opposites: Female patient <> Testis: Unremarkable",
        ],
    ),
    "anatomical": ConflictType(
        name="Anatomical Conflicts",
        description="Contradictions regarding body structures and their presence/absence",
        examples=[
            "Absent vs present structures: Cholelithiasis \
                <> The gallbladder is absent",
            "History of removal vs present structure: \
                Bilat mastectomy (2010) <> Left breast: solid mass",
            "Imaging vs clinical finding: Procedure: \
                Chest XR <> Brain lesion",
            "Laterality mismatch: Stable ductal carcinoma of \
                left breast <> Right breast carcinoma",
        ],
    ),
    "value": ConflictType(
        name="Value Conflicts",
        description="Contradictory measurements, lab values, or quantitative findings",
        examples=[
            "Condition vs measurement: Hypoglycemia <> Blood glucose 145",
            "Conflicting lab measurements: 02/11/2022 WBC 8.0 <> 02/11/2022 WBC 5.5",
        ],
    ),
    "contraindication": ConflictType(
        name="Contraindication Conflicts",
        description="Conflicts between allergies/contraindications and treatments",
        examples=[
            "Allergy vs prescribed medication: \
                Allergic to acetaminophen <> Home meds include Tylenol"
        ],
    ),
    "comparison": ConflictType(
        name="Comparison Conflicts",
        description="Contradictory comparative statements or temporal changes",
        examples=[
            "Increased/decreased vs measurements: Ultrasound shows \
                3 cm lesion, previously 4 cm, indicating increase"
        ],
    ),
    "descriptive": ConflictType(
        name="Descriptive Conflicts",
        description="Contradictory descriptive statements about the same condition",
        examples=[
            "Positive vs unlikely statements: Lungs: Pleural effusion unlikely \
                <> Assessment: Pleural effusion",
            "Conflicting characteristics: Stable small pleural effusion \
                <> Impression: Small pleural effusion",
            "Multiple vs single statements: Findings: 9 mm lesion right kidney \
                <> Assessment: Right renal lesions",
        ],
    ),
}

# Prompt Configuration
PROMPTS_DIR = "prompts"

# Prompt file names (without .txt extension)
DOCTOR_AGENT_PROMPT_FILE = "doctor_agent"
EDITOR_AGENT_PROMPT_FILE = "editor_agent"
MODERATOR_AGENT_PROMPT_FILE = "moderator_agent"
