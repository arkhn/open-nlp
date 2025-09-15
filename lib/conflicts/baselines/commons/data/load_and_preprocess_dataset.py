import json
import logging
from typing import Dict, List

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

from ..config import CONFLICT_TYPES, get_conflicts_data_path

logger = logging.getLogger(__name__)


def load_conflict_dataset(data_path: str = "processed/_12092025.json") -> pd.DataFrame:
    """
    Load the conflict dataset from processed JSON file.

    Args:
        data_path: Path to the processed JSON file

    Returns:
        DataFrame with conflict data
    """
    full_data_path = get_conflicts_data_path(data_path)

    if not full_data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {full_data_path}")

    logger.info(f"Loading dataset from {full_data_path}")

    # Load JSON data
    with open(full_data_path, "r") as f:
        data = json.load(f)

    # Convert to DataFrame
    df = process_conflict_json_data(data)

    return df


def process_conflict_json_data(data: List[Dict]) -> pd.DataFrame:
    """
    Process the conflict JSON data into a DataFrame.

    Args:
        data: List of conflict data entries from JSON

    Returns:
        DataFrame with conflict data
    """
    conflict_pairs = []

    for entry in data:
        # Extract data from the entry
        doc_data = entry["data"]
        annotations = entry.get("annotations", [])

        # Get document texts
        doc1_text = doc_data.get("doc_1", "")
        doc2_text = doc_data.get("doc_2", "")

        # Create combined text for classification
        combined_text = f"Document 1: {doc1_text[:500]} [SEP] Document 2: {doc2_text[:500]}"

        # Extract conflict information from annotations
        conflict_type = "opposition"  # Default
        conflict_label = CONFLICT_TYPES[conflict_type]

        if annotations and len(annotations) > 0:
            # Get conflict type from the first annotation
            first_annotation = annotations[0]
            if "result" in first_annotation and len(first_annotation["result"]) > 0:
                result = first_annotation["result"][0]
                if "conflict_type" in result:
                    conflict_type = result["conflict_type"]
                    conflict_label = CONFLICT_TYPES.get(conflict_type, 0)

        # Create unique IDs for documents
        doc1_id = f"doc1_{len(conflict_pairs)}"
        doc2_id = f"doc2_{len(conflict_pairs)}"

        conflict_pairs.append(
            {
                "doc1_id": doc1_id,
                "doc2_id": doc2_id,
                "doc1_text": doc1_text,
                "doc2_text": doc2_text,
                "combined_text": combined_text,
                "conflict_type": conflict_type,
                "conflict_label": conflict_label,
                "subject_id": f"subject_{len(conflict_pairs)}",  # Generate unique subject ID
                "category1": "processed_doc1",
                "category2": "processed_doc2",
                "timestamp_1": doc_data.get("timestamp_1", ""),
                "timestamp_2": doc_data.get("timestamp_2", ""),
                "created_at": doc_data.get("created_at", ""),
            }
        )

    # Convert to DataFrame
    conflict_df = pd.DataFrame(conflict_pairs)

    logger.info(f"Processed {len(conflict_df)} conflict pairs from JSON data")
    logger.info(f"Conflict type distribution:\n{conflict_df['conflict_type'].value_counts()}")

    return conflict_df


def preprocess_text(text: str, max_length: int = 512) -> str:
    """
    Preprocess text for classification.

    Args:
        text: Input text
        max_length: Maximum length to truncate

    Returns:
        Preprocessed text
    """
    # Basic text cleaning
    text = text.strip()

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]

    return text


def load_and_preprocess_dataset(
    data_path: str = "processed/_12092025.json",
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    max_length: int = 512,
    random_state: int = 42,
):
    """
    Load and preprocess the conflict detection dataset.

    Args:
        data_path: Path to the processed JSON file
        test_ratio: Ratio for test split
        val_ratio: Ratio for validation split (from remaining data)
        max_length: Maximum text length
        random_state: Random state for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load the dataset
    df = load_conflict_dataset(data_path)

    # Preprocess text
    df["combined_text"] = df["combined_text"].apply(lambda x: preprocess_text(x, max_length))

    # Check class distribution
    class_counts = df["conflict_type"].value_counts()
    logger.info(f"Class distribution: {class_counts.to_dict()}")

    # Check if we can stratify (each class needs at least 2 samples)
    min_class_count = class_counts.min()
    can_stratify = min_class_count >= 2

    if can_stratify:
        logger.info("Using stratified splitting")
        # Split the data with stratification
        train_df, test_df = train_test_split(
            df, test_size=test_ratio, random_state=random_state, stratify=df["conflict_type"]
        )

        train_df, val_df = train_test_split(
            train_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_df["conflict_type"],
        )
    else:
        logger.warning(
            f"Some classes have only {min_class_count} sample(s). "
            f"Using random splitting without stratification."
        )
        # Split the data without stratification
        train_df, test_df = train_test_split(df, test_size=test_ratio, random_state=random_state)

        train_df, val_df = train_test_split(
            train_df, test_size=val_ratio, random_state=random_state
        )

    # Create datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    logger.info(
        f"Dataset splits - Train: {len(train_dataset)},"
        f" Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset
