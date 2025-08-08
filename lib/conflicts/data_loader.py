import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from base import DocumentPair
from config import BATCH_SIZE, DATA_PATH


class ClinicalDataLoader:
    """
    Loads and manages clinical document data from the preprocessed MIMIC-III dataset
    """

    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = Path(data_path)
        self.logger = logging.getLogger("DataLoader")
        self.data_df: Optional[pd.DataFrame] = None
        self.loaded = False

        # Load data on initialization
        self._load_data()

    def _load_data(self):
        """Load the preprocessed clinical documents data"""
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")

            self.logger.info(f"Loading clinical documents from: {self.data_path}")

            # Load parquet file
            self.data_df = pd.read_parquet(self.data_path)

            self.logger.info(f"Loaded {len(self.data_df)} clinical documents")
            self.logger.info(f"Categories: {self.data_df['category'].value_counts().to_dict()}")
            self.logger.info(f"Unique subjects: {self.data_df['subject_id'].nunique()}")

            # Clean data - remove very short documents
            initial_count = len(self.data_df)
            self.data_df = self.data_df[self.data_df["text"].str.len() >= 100]
            final_count = len(self.data_df)

            if initial_count != final_count:
                self.logger.info(
                    f"Filtered out {initial_count - final_count} documents < 100 characters"
                )

            self.loaded = True

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def get_random_document_pairs(
        self,
        count: int = BATCH_SIZE,
        same_subject: bool = False,
        category_filter: Optional[List[str]] = None,
        min_text_length: int = 200,
    ) -> List[DocumentPair]:
        """
        Get random pairs of clinical documents

        Args:
            count: Number of document pairs to return
            same_subject: If True, pair documents from the same subject
            category_filter: List of categories to filter by (e.g., ['Discharge summary'])
            min_text_length: Minimum text length for each document

        Returns:
            List of DocumentPair objects
        """
        if not self.loaded:
            raise RuntimeError("Data not loaded successfully")

        self.logger.info(f"Generating {count} random document pairs")

        # Apply filters
        filtered_df = self.data_df.copy()

        if category_filter:
            filtered_df = filtered_df[filtered_df["category"].isin(category_filter)]
            self.logger.debug(
                f"Filtered to {len(filtered_df)} documents matching categories: {category_filter}"
            )

        if min_text_length > 0:
            filtered_df = filtered_df[filtered_df["text"].str.len() >= min_text_length]
            self.logger.debug(
                f"Filtered to {len(filtered_df)} documents with min length {min_text_length}"
            )

        if len(filtered_df) < 2:
            raise ValueError("Not enough documents available after filtering")

        document_pairs = []

        for i in range(count):
            if same_subject:
                # Find subjects with multiple documents
                subject_counts = filtered_df["subject_id"].value_counts()
                subjects_with_multiple = subject_counts[subject_counts >= 2].index.tolist()

                if not subjects_with_multiple:
                    self.logger.warning(
                        "No subjects with multiple documents, falling back to random pairing"
                    )
                    pair = self._get_random_pair(filtered_df)
                else:
                    # Pick a random subject with multiple documents
                    subject = random.choice(subjects_with_multiple)
                    subject_docs = filtered_df[filtered_df["subject_id"] == subject].sample(n=2)
                    pair = self._create_document_pair(subject_docs.iloc[0], subject_docs.iloc[1])
            else:
                pair = self._get_random_pair(filtered_df)

            document_pairs.append(pair)

        self.logger.info(f"Generated {len(document_pairs)} document pairs")
        return document_pairs

    def _get_random_pair(self, df: pd.DataFrame) -> DocumentPair:
        """Get a random pair of documents from the dataframe"""
        if len(df) < 2:
            raise ValueError("Need at least 2 documents to create a pair")

        # Sample two different documents
        sampled = df.sample(n=2)
        return self._create_document_pair(sampled.iloc[0], sampled.iloc[1])

    def _create_document_pair(self, doc1: pd.Series, doc2: pd.Series) -> DocumentPair:
        """Create a DocumentPair from two pandas Series"""
        return DocumentPair(
            doc1_id=str(doc1["row_id"]),
            doc2_id=str(doc2["row_id"]),
            doc1_text=doc1["text"],
            doc2_text=doc2["text"],
            subject_id=f"{doc1['subject_id']},{doc2['subject_id']}",
            category1=doc1["category"],
            category2=doc2["category"],
        )

    def get_documents_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get documents of a specific category

        Args:
            category: Category to filter by
            limit: Maximum number of documents to return

        Returns:
            List of document dictionaries
        """
        if not self.loaded:
            raise RuntimeError("Data not loaded successfully")

        filtered_df = self.data_df[self.data_df["category"] == category].head(limit)

        return filtered_df.to_dict("records")

    def get_subject_documents(self, subject_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get all documents for a specific subject

        Args:
            subject_id: Subject ID to filter by
            limit: Maximum number of documents to return

        Returns:
            List of document dictionaries
        """
        if not self.loaded:
            raise RuntimeError("Data not loaded successfully")

        filtered_df = self.data_df[self.data_df["subject_id"] == int(subject_id)].head(limit)

        return filtered_df.to_dict("records")

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dataset

        Returns:
            Dictionary with dataset statistics
        """
        if not self.loaded:
            raise RuntimeError("Data not loaded successfully")

        return {
            "total_documents": len(self.data_df),
            "unique_subjects": self.data_df["subject_id"].nunique(),
            "categories": self.data_df["category"].value_counts().to_dict(),
            "text_length_stats": {
                "mean": self.data_df["text"].str.len().mean(),
                "median": self.data_df["text"].str.len().median(),
                "min": self.data_df["text"].str.len().min(),
                "max": self.data_df["text"].str.len().max(),
            },
            "sample_categories": list(self.data_df["category"].unique()),
        }

    def create_balanced_pairs(
        self, count: int = BATCH_SIZE, balance_categories: bool = True
    ) -> List[DocumentPair]:
        """
        Create balanced pairs of documents for training

        Args:
            count: Number of pairs to create
            balance_categories: If True, try to balance across categories

        Returns:
            List of DocumentPair objects
        """
        if not self.loaded:
            raise RuntimeError("Data not loaded successfully")

        document_pairs = []

        if balance_categories:
            # Get available categories
            categories = self.data_df["category"].unique().tolist()
            pairs_per_category = count // len(categories)
            remaining_pairs = count % len(categories)

            for category in categories:
                category_docs = self.data_df[self.data_df["category"] == category]

                if len(category_docs) >= 2:
                    # Create pairs within this category
                    pairs_to_create = pairs_per_category + (1 if remaining_pairs > 0 else 0)
                    if remaining_pairs > 0:
                        remaining_pairs -= 1

                    for _ in range(pairs_to_create):
                        if len(category_docs) >= 2:
                            pair = self._get_random_pair(category_docs)
                            document_pairs.append(pair)

            # Fill any remaining pairs with random documents
            while len(document_pairs) < count:
                pair = self._get_random_pair(self.data_df)
                document_pairs.append(pair)

        else:
            # Just create random pairs
            document_pairs = self.get_random_document_pairs(count)

        return document_pairs

    def reload_data(self):
        """Reload the data from file"""
        self.loaded = False
        self.data_df = None
        self._load_data()


def create_sample_data_if_missing(data_path: str = DATA_PATH) -> bool:
    """
    Create sample clinical documents if the main data file is missing
    This is useful for testing when the full MIMIC-III dataset is not available

    Args:
        data_path: Path where to create sample data

    Returns:
        True if sample data was created, False if real data exists
    """
    data_file = Path(data_path)

    if data_file.exists():
        return False

    logger = logging.getLogger("DataLoader.Sample")
    logger.warning(f"Main data file {data_path} not found, creating sample data for testing")

    # Create sample clinical documents
    sample_data = [
        {
            "row_id": 1,
            "subject_id": 1001,
            "category": "Discharge summary",
            "text": """DISCHARGE SUMMARY

Patient: John Doe, 65-year-old male
Admission Date: 01/15/2023
Discharge Date: 01/20/2023

CHIEF COMPLAINT: Chest pain and shortness of breath

HISTORY OF PRESENT ILLNESS:
The patient presented to the emergency department with a 2-day history of \
chest pain and shortness of breath. Pain was described as crushing, substernal, \
and radiating to the left arm. No known cardiac disease history.

PHYSICAL EXAMINATION:
Vital signs: BP 140/90, HR 88, RR 18, O2 sat 95% on room air
Cardiovascular: Regular rate and rhythm, no murmurs
Pulmonary: Clear to auscultation bilaterally
Extremities: No edema

LABORATORY RESULTS:
Troponin I: 0.02 ng/mL (normal)
BNP: 45 pg/mL (normal)
WBC: 7.5 K/uL

IMAGING:
Chest X-ray: No acute cardiopulmonary process
ECG: Normal sinus rhythm

ASSESSMENT AND PLAN:
Non-cardiac chest pain, likely musculoskeletal. Discharged home with pain medications.""",
        },
        {
            "row_id": 2,
            "subject_id": 1001,
            "category": "Progress note",
            "text": """PROGRESS NOTE - Day 2

Patient continues to complain of chest pain. Pain is now localized to the right side and \
is sharp in nature. Vital signs stable.

EXAMINATION:
Cardiovascular: Irregular heart rhythm noted, possible atrial fibrillation
Pulmonary: Bilateral rales present in lower lobes

NEW ORDERS:
- Cardiac monitor
- Repeat ECG
- Chest CT with contrast

ASSESSMENT:
Chest pain, rule out pulmonary embolism
Possible atrial fibrillation""",
        },
        {
            "row_id": 3,
            "subject_id": 1002,
            "category": "Radiology",
            "text": """RADIOLOGY REPORT

Examination: CT Chest with IV contrast
Date: 01/16/2023

TECHNIQUE: Axial CT images of the chest obtained with IV contrast.

FINDINGS:
Lungs: No pulmonary nodules or masses identified. No pleural effusion.
Heart: Normal cardiac size and contour.
Mediastinum: No lymphadenopathy.
Bones: No acute fractures.

IMPRESSION:
Normal chest CT. No evidence of pulmonary embolism.""",
        },
        {
            "row_id": 4,
            "subject_id": 1002,
            "category": "Laboratory",
            "text": """LABORATORY REPORT

Patient: Jane Smith
Date: 01/16/2023

COMPLETE METABOLIC PANEL:
Glucose: 120 mg/dL (normal)
BUN: 18 mg/dL (normal)
Creatinine: 0.9 mg/dL (normal)
Sodium: 140 mmol/L (normal)
Potassium: 4.2 mmol/L (normal)
Chloride: 102 mmol/L (normal)

LIPID PANEL:
Total cholesterol: 180 mg/dL
HDL: 45 mg/dL
LDL: 120 mg/dL
Triglycerides: 150 mg/dL

INTERPRETATION: All values within normal limits.""",
        },
    ]

    # Create directory if it doesn't exist
    data_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame and save
    df = pd.DataFrame(sample_data)
    df.to_parquet(data_file, index=False)

    logger.info(f"Created sample data with {len(sample_data)} documents at {data_path}")
    return True
