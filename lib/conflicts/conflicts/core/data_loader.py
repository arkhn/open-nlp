import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .models import DocumentPair


class DataLoader:
    """
    Loads and manages clinical document data from the preprocessed MIMIC-III dataset
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_path = Path(cfg.data.path)
        self.logger = logging.getLogger("DataLoader")
        self.data_df = pd.read_parquet(self.data_path)

    def get_random_document_pairs(
        self,
        dataset_size: int,
        category_filter: Optional[List[str]] = None,
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
        self.logger.info(f"Generating {dataset_size} random document pairs")

        # Apply filters
        filtered_df = self.data_df.copy()

        if category_filter:
            filtered_df = filtered_df[filtered_df["category"].isin(category_filter)]
            self.logger.debug(
                f"Filtered to {len(filtered_df)} documents matching categories: {category_filter}"
            )
        document_pairs = []

        for _ in range(dataset_size):
            subject_counts = filtered_df["subject_id"].value_counts()
            subject = random.choice(subject_counts.index.tolist())
            subject_docs = filtered_df[filtered_df["subject_id"] == subject].sample(n=2)
            pair = self._create_document_pair(subject_docs.iloc[0], subject_docs.iloc[1])
            document_pairs.append(pair)

        self.logger.info(f"Generated {len(document_pairs)} document pairs")
        return document_pairs

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
            doc1_timestamp=doc1.get("chart_time") if "chart_time" in doc1 else None,
            doc2_timestamp=doc2.get("chart_time") if "chart_time" in doc2 else None,
        )

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dataset

        Returns:
            Dictionary with dataset statistics
        """

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
