"""
classify_documents.py — Classify MIMIC-III documents by conflict type
and save per-type document lists.

OVERVIEW
--------
This script classifies clinical notes from the MIMIC-III dataset into one or more of four
conflict types using embedding similarity (sentence-transformers). Each document is compared
against a text description of each conflict type; if the cosine similarity exceeds a
per-type threshold, the document is labelled with that type (multi-label).

COMMAND LINE
-----------
python classify_documents.py --data ../data/mimic-iii-verifact-bhc.parquet --output-dir ../data/

CONFLICT TYPES
--------------
  temporality      — Allergies, constitutional characteristics, immutable facts (blood type, etc.)
  clinical_history — Surgical files, transplant records, implanted devices
  biomarker        — Lab values, vital signs, physiological measurements
  pre_post_care    — Admission/discharge documents, care transition points

OUTPUTS  (all written under data/)
-------
  documents_per_conflict_type.json          — {conflict_type: [row_id, …]} for every matching doc
  conflict_type_documents/<type>_documents.csv
                                    — per-type CSV with row_id, subject_id, category, score
  top2_per_conflict_type.parquet            — top-2 highest-scoring documents per conflict type
  document_conflict_classification.parquet  — full classification table with all scores and flags

USAGE
-----
  python classify_documents.py [--data PATH] [--top-n N] [--output-dir DIR]

  --data        Path to the MIMIC-III parquet file.
                Default: ../data/mimic-iii-verifact-bhc.parquet
  --top-n       Number of top-scoring documents to save per conflict type.
                Default: 2
  --output-dir  Root directory for all outputs.
                Default: ../data

REQUIREMENTS
------------
  pip install sentence-transformers pandas numpy pyarrow

NOTES
-----
  - Thresholds were tuned to capture documents that *mention* the topic in at least one sentence.
  - Only the first 1000 characters of each document are embedded for speed.
  - Classification is multi-label: a document can belong to multiple conflict types.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Conflict type definitions
# ---------------------------------------------------------------------------

CONFLICT_DESCRIPTIONS: dict[str, str] = {
    "temporality": """Long-term follow-up documents for allergies or constitutional patient
        characteristics.
        Immutable characteristics like blood type, genetic markers, birth date.
        Allergy documentation, chronic irreversible conditions like Type 1 diabetes.
        Constitutional metrics like adult height.
        Patient characteristics that should remain constant.""",
    "clinical_history": """Surgical files, transplant records, pre and post operative documents.
        Surgical history, organ removal, appendectomy, cholecystectomy.
        Transplant status, immunosuppression, implanted medical devices, pacemaker.
        Anatomical alterations from prior procedures.""",
    "biomarker": """Laboratory values, vital signs, physiological measurements.
        Hemoglobin, glucose, troponin, creatinine, WBC counts.
        Post-operative biomarker trajectories, organ function markers.
        Immunological markers, procedural marker changes.""",
    "pre_post_care": """Admission and discharge documents, care transition points.
        Chief complaint, admission diagnosis, discharge summary.
        Medication reconciliation, functional status changes.
        Diagnosis changes between admission and discharge.""",
}

# Per-type cosine-similarity thresholds.
# Tuned so that a document "mentioning" the topic in at least one sentence is captured.
PER_TYPE_THRESHOLDS: dict[str, float] = {
    "temporality": 0.09,
    "clinical_history": 0.16,
    "biomarker": 0.21,
    "pre_post_care": 0.23,
}

CONFLICT_TYPES = list(CONFLICT_DESCRIPTIONS.keys())


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------


def embed_descriptions(model: SentenceTransformer) -> dict[str, np.ndarray]:
    """Pre-compute embeddings for all conflict type descriptions.

    Args:
        model: A loaded SentenceTransformer model.

    Returns:
        Mapping from conflict type name to its description embedding vector.
    """
    return {ct: model.encode(desc) for ct, desc in CONFLICT_DESCRIPTIONS.items()}


def classify_document(
    text: str,
    model: SentenceTransformer,
    desc_embeddings: dict[str, np.ndarray],
    thresholds: dict[str, float] = PER_TYPE_THRESHOLDS,
    max_chars: int = 1000,
) -> dict:
    """Classify a single document into one or more conflict types.

    Embeds the first `max_chars` characters of the document and computes cosine
    similarity against each conflict type description. A document is labelled with
    a conflict type if its similarity score meets or exceeds that type's threshold.

    Args:
        text:            Raw document text.
        model:           Loaded SentenceTransformer model.
        desc_embeddings: Pre-computed description embeddings (from embed_descriptions).
        thresholds:      Per-type minimum similarity scores.
        max_chars:       Number of leading characters to embed (truncated for speed).

    Returns:
        dict with keys:
          matching_types (list[str])  — conflict types above threshold
          scores         (dict)       — cosine similarity score per conflict type
          best_match     (str)        — conflict type with the highest similarity
    """
    doc_emb = model.encode(text[:max_chars])

    scores: dict[str, float] = {}
    for ct, desc_emb in desc_embeddings.items():
        cos_sim = np.dot(doc_emb, desc_emb) / (np.linalg.norm(doc_emb) * np.linalg.norm(desc_emb))
        scores[ct] = float(cos_sim)

    matching_types = [ct for ct, score in scores.items() if score >= thresholds[ct]]
    best_match = max(scores, key=scores.get)

    return {"matching_types": matching_types, "scores": scores, "best_match": best_match}


def classify_all_documents(
    df: pd.DataFrame,
    model: SentenceTransformer,
    desc_embeddings: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Classify every row in *df* and return a results DataFrame.

    Args:
        df:              Source dataframe with columns: text, category, row_id, subject_id.
        model:           Loaded SentenceTransformer model.
        desc_embeddings: Pre-computed description embeddings.

    Returns:
        DataFrame with columns: row_id, subject_id, category, matching_types, scores,
        best_match, is_<type> (bool), num_types (int), score_<type> (float).
    """
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        classification = classify_document(row["text"], model, desc_embeddings)
        classification["category"] = row["category"]
        classification["row_id"] = row["row_id"]
        classification["subject_id"] = row["subject_id"]
        results.append(classification)

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{total} documents...")

    results_df = pd.DataFrame(results)

    # Binary indicator columns
    for ct in CONFLICT_TYPES:
        results_df[f"is_{ct}"] = results_df["matching_types"].apply(lambda x: ct in x)

    # Score columns (flattened from the scores dict)
    for ct in CONFLICT_TYPES:
        results_df[f"score_{ct}"] = results_df["scores"].apply(lambda x: x[ct])

    results_df["num_types"] = results_df["matching_types"].apply(len)

    return results_df


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_document_lists(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Save per-conflict-type document lists to JSON and CSV.

    Produces:
      <output_dir>/documents_per_conflict_type.json
        A single JSON file mapping each conflict type to the list of matching row_ids.
      <output_dir>/conflict_type_documents/<type>_documents.csv
        One CSV per conflict type with row_id, subject_id, category, score columns,
        sorted by similarity score descending.

    Args:
        results_df:  Classification results from classify_all_documents.
        output_dir:  Root output directory (created if it does not exist).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON: {conflict_type: [row_id, ...]}
    docs_per_type: dict[str, list] = {}
    for ct in CONFLICT_TYPES:
        docs_per_type[ct] = results_df[results_df[f"is_{ct}"]]["row_id"].tolist()

    json_path = output_dir / "documents_per_conflict_type.json"
    with open(json_path, "w") as f:
        json.dump(docs_per_type, f, indent=2)
    print(f"Saved document lists → {json_path}")
    for ct, ids in docs_per_type.items():
        print(f"  {ct}: {len(ids)} documents")

    # Per-type CSVs
    csv_dir = output_dir / "conflict_type_documents"
    csv_dir.mkdir(exist_ok=True)

    for ct in CONFLICT_TYPES:
        mask = results_df[f"is_{ct}"]
        ct_docs = results_df[mask][["row_id", "subject_id", "category"]].copy()
        ct_docs["score"] = results_df[mask][f"score_{ct}"]
        ct_docs = ct_docs.sort_values("score", ascending=False)

        csv_path = csv_dir / f"{ct}_documents.csv"
        ct_docs.to_csv(csv_path, index=False)
        print(f"Saved CSV → {csv_path}  ({len(ct_docs)} docs)")


def save_top_n_documents(
    results_df: pd.DataFrame,
    source_df: pd.DataFrame,
    top_n: int,
    output_dir: Path,
) -> None:
    """Save the top-N highest-scoring documents per conflict type to a parquet file.

    Args:
        results_df:  Classification results DataFrame.
        source_df:   Original source DataFrame containing the 'text' column.
        top_n:       Number of top documents to collect per conflict type.
        output_dir:  Output directory.
    """
    rows = []
    for ct in CONFLICT_TYPES:
        top_idx = results_df[f"score_{ct}"].nlargest(top_n).index
        for rank, idx in enumerate(top_idx, 1):
            row = results_df.loc[idx]
            rows.append(
                {
                    "conflict_type": ct,
                    "rank": rank,
                    "score": row[f"score_{ct}"],
                    "row_id": row["row_id"],
                    "subject_id": row["subject_id"],
                    "category": row["category"],
                    "text": source_df.loc[idx, "text"],
                }
            )

    top_df = pd.DataFrame(rows)
    out_path = output_dir / "top2_per_conflict_type.parquet"
    top_df.to_parquet(out_path, index=False)
    print(f"Saved top-{top_n} documents per type → {out_path}  ({len(top_df)} rows)")


def save_full_classification(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Save the full classification table (all documents, all scores) to parquet.

    Columns: row_id, subject_id, category, best_match, num_types,
             is_<type> (bool × 4), score_<type> (float × 4).

    Args:
        results_df:  Classification results DataFrame.
        output_dir:  Output directory.
    """
    export_cols = (
        ["row_id", "subject_id", "category", "best_match", "num_types"]
        + [f"is_{ct}" for ct in CONFLICT_TYPES]
        + [f"score_{ct}" for ct in CONFLICT_TYPES]
    )
    out_path = output_dir / "document_conflict_classification.parquet"
    results_df[export_cols].to_parquet(out_path, index=False)
    print(f"Saved full classification → {out_path}  ({len(results_df)} rows)")


def print_summary(results_df: pd.DataFrame, source_df: pd.DataFrame) -> None:
    """Print a human-readable summary of the classification results.

    Args:
        results_df:  Classification results DataFrame.
        source_df:   Original source DataFrame (used for total document count).
    """
    total = len(source_df)
    print("\n" + "=" * 70)
    print("MULTI-LABEL CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"\nTotal documents classified: {total}")

    print("\nDocuments per conflict type (can overlap):")
    for ct in CONFLICT_TYPES:
        count = int(results_df[f"is_{ct}"].sum())
        thresh = PER_TYPE_THRESHOLDS[ct]
        print(f"  {ct:20}: {count:5} docs ({count / total * 100:5.1f}%)  [threshold={thresh}]")

    print("\nNumber of conflict types per document:")
    for n, count in results_df["num_types"].value_counts().sort_index().items():
        print(f"  {n} type(s): {count:5} docs ({count / total * 100:.1f}%)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify MIMIC-III clinical documents by conflict type using embedding similarity "
            "and save per-type document lists."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../data/mimic-iii-verifact-bhc.parquet"),
        help="Path to the MIMIC-III parquet file. (default: mimic-iii-verifact-bhc.parquet)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=2,
        help="Number of top-scoring documents to save per conflict type. (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data"),
        help="Root directory for all output files. (default: ../data)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data} ...")
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} documents from {df.subject_id.nunique()} patients")

    print("\nLoading embedding model (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding conflict type descriptions ...")
    desc_embeddings = embed_descriptions(model)

    print(f"\nClassifying {len(df)} documents ...")
    results_df = classify_all_documents(df, model, desc_embeddings)

    print_summary(results_df, df)

    print("\nSaving outputs ...")
    save_document_lists(results_df, args.output_dir)
    save_top_n_documents(results_df, df, args.top_n, args.output_dir)
    save_full_classification(results_df, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
