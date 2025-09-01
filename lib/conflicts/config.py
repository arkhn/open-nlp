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

# Pipeline Configuration
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
DEFAULT_MIN_VALIDATION_SCORE = 70
DEFAULT_MAX_RETRIES = 3

# Essential Text Processing Thresholds
MIN_TARGET_TEXT_LENGTH = 20
SIMILARITY_THRESHOLD = 0.6


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
