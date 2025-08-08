#!/usr/bin/env python3
"""
Configuration file for the Clinical Document Conflict Pipeline
"""

import os
from typing import Dict, List
from dataclasses import dataclass

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-70b-versatile"  # Using Groq's LLaMA model instead of GPT-OSS
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Data Configuration
DATA_PATH = "data/mimic-iii-verifact-bhc.parquet"
DATABASE_PATH = "validated_documents.db"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FILE = "pipeline.log"

# Pipeline Configuration
MAX_RETRY_ATTEMPTS = 3
BATCH_SIZE = 2  # Number of documents to process together

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
            "Normal vs abnormal findings of same body structure: Left breast: Unremarkable <> Left breast demonstrates persistent circumscribed masses",
            "Negative vs positive statements: No cardiopulmonary disease <> Bibasilar atelectasis",
            "Lab/vital sign interpretation: Low blood sugar at admission <> Patient admitted with hyperglycemia",
            "Opposite disorders: Hypernatremia <> Hyponatremia",
            "Sex information opposites: Female patient <> Testis: Unremarkable"
        ]
    ),
    "anatomical": ConflictType(
        name="Anatomical Conflicts",
        description="Contradictions regarding body structures and their presence/absence",
        examples=[
            "Absent vs present structures: Cholelithiasis <> The gallbladder is absent",
            "History of removal vs present structure: Bilat mastectomy (2010) <> Left breast: solid mass",
            "Imaging vs clinical finding: Procedure: Chest XR <> Brain lesion",
            "Laterality mismatch: Stable ductal carcinoma of left breast <> Right breast carcinoma"
        ]
    ),
    "value": ConflictType(
        name="Value Conflicts",
        description="Contradictory measurements, lab values, or quantitative findings",
        examples=[
            "Condition vs measurement: Hypoglycemia <> Blood glucose 145",
            "Conflicting lab measurements: 02/11/2022 WBC 8.0 <> 02/11/2022 WBC 5.5"
        ]
    ),
    "contraindication": ConflictType(
        name="Contraindication Conflicts",
        description="Conflicts between allergies/contraindications and treatments",
        examples=[
            "Allergy vs prescribed medication: Allergic to acetaminophen <> Home meds include Tylenol"
        ]
    ),
    "comparison": ConflictType(
        name="Comparison Conflicts",
        description="Contradictory comparative statements or temporal changes",
        examples=[
            "Increased/decreased vs measurements: Ultrasound shows 3 cm lesion, previously 4 cm, indicating increase"
        ]
    ),
    "descriptive": ConflictType(
        name="Descriptive Conflicts",
        description="Contradictory descriptive statements about the same condition",
        examples=[
            "Positive vs unlikely statements: Lungs: Pleural effusion unlikely <> Assessment: Pleural effusion",
            "Conflicting characteristics: Stable small pleural effusion <> Impression: Small pleural effusion",
            "Multiple vs single statements: Findings: 9 mm lesion right kidney <> Assessment: Right renal lesions"
        ]
    )
}

# Prompt Templates
DOCTOR_AGENT_PROMPT = """
You are a clinical expert analyzing two medical documents to determine what type of conflict should be introduced between them.

Available conflict types:
{conflict_types}

Document 1:
{document1}

Document 2:
{document2}

Analyze these documents and choose the most appropriate conflict type to introduce. Consider:
1. The clinical content and findings in both documents
2. What type of conflict would be realistic and educational
3. Which conflict type best matches the available clinical information

Respond with JSON format:
{{
    "conflict_type": "chosen_conflict_type_key",
    "reasoning": "explanation of why this conflict type was chosen",
    "modification_instructions": "specific instructions for the Editor Agent on how to create the conflict"
}}
"""

EDITOR_AGENT_PROMPT = """
You are a clinical document editor. Your task is to modify clinical documents to introduce specific conflicts as instructed.

Conflict Type: {conflict_type}
Conflict Description: {conflict_description}
Modification Instructions: {modification_instructions}

Document 1:
{document1}

Document 2:
{document2}

Modify these documents to create the specified conflict while:
1. Maintaining clinical realism and proper medical terminology
2. Ensuring the conflict is clear and detectable
3. Preserving the overall structure and format of the documents
4. Making minimal changes necessary to create the conflict

Respond with JSON format:
{{
    "modified_document1": "complete modified document 1",
    "modified_document2": "complete modified document 2",
    "changes_made": "summary of specific changes made to create the conflict"
}}
"""

MODERATOR_AGENT_PROMPT = """
You are a clinical document validator. Your task is to validate whether the modifications made to create conflicts are acceptable and realistic.

Original Documents:
Document 1: {original_doc1}
Document 2: {original_doc2}

Modified Documents:
Document 1: {modified_doc1}
Document 2: {modified_doc2}

Expected Conflict Type: {conflict_type}
Changes Made: {changes_made}

Validate these modifications by checking:
1. Are the conflicts realistic and clinically plausible?
2. Do the modifications preserve the overall medical context?
3. Are the changes appropriate for the specified conflict type?
4. Would these conflicts be useful for training/validation purposes?
5. Are there any unrealistic or impossible medical combinations?

Respond with JSON format:
{{
    "is_valid": true/false,
    "validation_score": 0-100,
    "feedback": "detailed feedback on the modifications",
    "issues_found": ["list of any issues if invalid"],
    "approval_reasoning": "explanation of validation decision"
}}
"""
