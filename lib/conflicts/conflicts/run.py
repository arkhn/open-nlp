"""
run.py — Run the PDEM pipeline on same-patient document pairs from classify_documents output.

OVERVIEW
--------
For each conflict type, this script:
  1. Loads the per-type document list produced by classify_documents.py
     (data/documents_per_conflict_type.json).
  2. Joins with the source parquet to retrieve document text and timestamps.
  3. Joins with the per-type score CSVs to rank documents by relevance.
  4. Groups documents by patient (subject_id).
  5. For each patient with ≥ 2 matching documents, takes the top-K highest-scoring
     documents and generates all C(K, 2) pairs (K set by pipeline.top_k_per_subject).
  6. Processes each pair using only the Doctor agent specialised for that conflict type.

SCALE (approximate, 100 patients)
----------------------------------
  top_k_per_subject=2  →   1 pair/patient →   400 pairs total
  top_k_per_subject=5  →  10 pairs/patient → 4,000 pairs total
  top_k_per_subject=10 →  45 pairs/patient → 18,000 pairs total

USAGE
-----
  python -m conflicts.run                               # default top_k=5
  python -m conflicts.run pipeline.top_k_per_subject=3 # override via Hydra

PREREQUISITES
-------------
  Run classify_documents.py first to produce:
    data/documents_per_conflict_type.json
    data/conflict_type_documents/<type>_documents.csv
"""

import json
import logging
from itertools import combinations
from pathlib import Path

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from .core.constants import (
    BIOMARKER_MONITORING_CONFLICT_TYPE,
    CLINICAL_HISTORY_CONFLICT_TYPE,
    PRE_POST_CARE_CONFLICT_TYPE,
    TEMPORALITY_CONFLICT_TYPE,
)
from .core.exceptions import PropositionAgentError
from .core.models import DocumentPair
from .core.pipeline import Pipeline

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Map short names from classify_documents.py output to pipeline constants
PARQUET_TO_CONSTANT = {
    "biomarker": BIOMARKER_MONITORING_CONFLICT_TYPE,
    "clinical_history": CLINICAL_HISTORY_CONFLICT_TYPE,
    "pre_post_care": PRE_POST_CARE_CONFLICT_TYPE,
    "temporality": TEMPORALITY_CONFLICT_TYPE,
    # Also accept full names in case JSON uses them
    BIOMARKER_MONITORING_CONFLICT_TYPE: BIOMARKER_MONITORING_CONFLICT_TYPE,
    CLINICAL_HISTORY_CONFLICT_TYPE: CLINICAL_HISTORY_CONFLICT_TYPE,
    PRE_POST_CARE_CONFLICT_TYPE: PRE_POST_CARE_CONFLICT_TYPE,
    TEMPORALITY_CONFLICT_TYPE: TEMPORALITY_CONFLICT_TYPE,
}


def load_source_documents() -> pd.DataFrame:
    """Load the MIMIC-III source parquet indexed by row_id.

    Returns:
        DataFrame with columns: subject_id, category, text, chart_time, indexed by row_id.
    """
    path = DATA_DIR / "mimic-iii-verifact-bhc.parquet"
    df = pd.read_parquet(path, columns=["row_id", "subject_id", "category", "text", "chart_time"])
    return df.set_index("row_id")


def load_type_scores(short_name: str) -> pd.Series:
    """Load similarity scores for a conflict type from the per-type CSV.

    Args:
        short_name: Short conflict type key (e.g. "biomarker").

    Returns:
        Series of cosine similarity scores indexed by row_id.
    """
    csv_path = DATA_DIR / "conflict_type_documents" / f"{short_name}_documents.csv"
    scores_df = pd.read_csv(csv_path, usecols=["row_id", "score"], index_col="row_id")
    return scores_df["score"]


def build_pairs_for_type(
    short_name: str,
    row_ids: list,
    source_df: pd.DataFrame,
    top_k: int = 5,
) -> list[DocumentPair]:
    """Build all C(top_k, 2) same-patient document pairs for a given conflict type.

    For each patient with ≥ 2 matching documents, selects the top-K highest-scoring
    documents and generates every combination of 2, maximising both coverage and
    relevance to the conflict type.

    Args:
        short_name:  Short conflict type key (e.g. "biomarker").
        row_ids:     List of row_ids from documents_per_conflict_type.json.
        source_df:   Source parquet DataFrame indexed by row_id.
        top_k:       Number of top-scoring docs to consider per patient.
                     C(top_k, 2) pairs are produced per patient.
                     E.g. top_k=5 → 10 pairs/patient, top_k=10 → 45 pairs/patient.

    Returns:
        List of DocumentPair objects.
    """
    scores = load_type_scores(short_name)

    type_df = source_df[source_df.index.isin(row_ids)].copy()
    type_df["score"] = type_df.index.map(scores)

    pairs = []
    skipped = 0

    for subject_id, subject_docs in type_df.groupby("subject_id"):
        if len(subject_docs) < 2:
            skipped += 1
            continue

        topk_docs = subject_docs.nlargest(min(top_k, len(subject_docs)), "score")

        for doc1, doc2 in combinations(topk_docs.itertuples(), 2):
            pairs.append(
                DocumentPair(
                    doc1_id=str(doc1.Index),
                    doc2_id=str(doc2.Index),
                    doc1_text=doc1.text,
                    doc2_text=doc2.text,
                    subject_id=str(subject_id),
                    category1=doc1.category,
                    category2=doc2.category,
                    doc1_timestamp=doc1.chart_time if pd.notna(doc1.chart_time) else None,
                    doc2_timestamp=doc2.chart_time if pd.notna(doc2.chart_time) else None,
                )
            )

    if skipped:
        log.debug(f"  Skipped {skipped} patients with < 2 matching documents")

    return pairs


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig) -> None:
    """Process all same-patient document pairs from classify_documents.py output."""

    hydra_cfg = HydraConfig.get()
    log.info(f"Working directory: {hydra_cfg.runtime.output_dir}")

    pipeline = Pipeline(cfg)

    source_df = load_source_documents()
    log.info(f"Loaded {len(source_df)} source documents")

    json_path = DATA_DIR / "documents_per_conflict_type.json"
    with open(json_path) as f:
        docs_per_type: dict[str, list] = json.load(f)
    log.info(f"Loaded document lists from {json_path}")

    top_k = cfg.pipeline.get("top_k_per_subject", 5)
    log.info(
        f"top_k_per_subject={top_k} → up to C({top_k},2)={top_k*(top_k-1)//2} "
        "pairs per patient per type"
    )

    total_processed = 0
    total_success = 0

    for short_name, row_ids in docs_per_type.items():
        conflict_type = PARQUET_TO_CONSTANT.get(short_name)
        if conflict_type is None:
            log.error(f"Unknown conflict type '{short_name}', skipping")
            continue

        pairs = build_pairs_for_type(short_name, row_ids, source_df, top_k=top_k)
        log.info(f"\n{conflict_type}: {len(row_ids)} docs → {len(pairs)} pairs")

        type_success = 0
        for document_pair in pairs:
            pair_id = f"{document_pair.doc1_id}_{document_pair.doc2_id}"
            log.info(f"  Processing {pair_id} (subject={document_pair.subject_id})")

            try:
                success, _ = pipeline.process_document_pair(
                    document_pair, conflict_types=[conflict_type]
                )
            except PropositionAgentError as e:
                log.warning(f"  SKIPPED {pair_id} — PropositionAgent failed: {e}")
                continue

            total_processed += 1
            if success:
                total_success += 1
                type_success += 1
                pipeline.dataset_manager.save_to_json()

            log.info(f"  {'VALID' if success else 'INVALID'} — {type_success} valid so far")

        log.info(f"{conflict_type}: {type_success}/{len(pairs)} pairs successful")

    log.info(f"\nDone. {total_success}/{total_processed} total pairs successful. Results saved.")


if __name__ == "__main__":
    main()
