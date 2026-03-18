"""
eda_classification.py — Exploratory data analysis for document_conflict_classification.parquet

USAGE
-----
  python eda_classification.py [--data PATH] [--output-dir DIR] [--processed PATH]

  --data        Path to the classification parquet file.
                Default: ../data/document_conflict_classification.parquet
  --output-dir  Directory to save plots. Default: ../data/eda
  --processed   Path to a processed pipeline JSON file (e.g. processed/908a039_26092025.json).
                When provided, also generates moderator_score_dist.png (paper fig:score_dist).

EXAMPLE
-----------
python eda_classification.py --data ../data/document_conflict_classification.parquet \
                             --output-dir ../data/eda
python eda_classification.py --processed processed/908a039_26092025.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CONFLICT_TYPES = ["temporality", "clinical_history", "biomarker", "pre_post_care"]
SCORE_COLS = [f"score_{ct}" for ct in CONFLICT_TYPES]
IS_COLS = [f"is_{ct}" for ct in CONFLICT_TYPES]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def savefig(fig: plt.Figure, path: Path, name: str) -> None:
    out = path / name
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"  Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_label_coverage(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: % of documents labelled per conflict type."""
    total = len(df)
    counts = {ct: df[f"is_{ct}"].sum() for ct in CONFLICT_TYPES}

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.keys(), [v / total * 100 for v in counts.values()], color="steelblue")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_title("Documents labelled per conflict type")
    ax.set_ylabel("% of documents")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Conflict type")
    savefig(fig, output_dir, "label_coverage.png")


def plot_num_types_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: distribution of number of conflict types per document."""
    counts = df["num_types"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.bar_label(bars, padding=3)
    ax.set_title("Number of conflict types per document")
    ax.set_xlabel("Number of conflict types")
    ax.set_ylabel("Count")
    savefig(fig, output_dir, "num_types_distribution.png")


def plot_best_match_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart: best-match conflict type distribution."""
    counts = df["best_match"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color="steelblue")
    ax.bar_label(bars, padding=3)
    ax.set_title("Best-match conflict type distribution")
    ax.set_xlabel("Conflict type")
    ax.set_ylabel("Count")
    savefig(fig, output_dir, "best_match_distribution.png")


def plot_score_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Violin + boxplot: similarity score distribution per conflict type."""
    fig, axes = plt.subplots(1, len(CONFLICT_TYPES), figsize=(14, 5), sharey=False)

    for ax, ct in zip(axes, CONFLICT_TYPES):
        col = f"score_{ct}"
        sns.violinplot(y=df[col], ax=ax, color="steelblue", inner="box", cut=0)
        ax.set_title(ct)
        ax.set_xlabel("")
        ax.set_ylabel("Cosine similarity" if ax == axes[0] else "")

    fig.suptitle("Score distributions per conflict type", y=1.02)
    fig.tight_layout()
    savefig(fig, output_dir, "score_distributions.png")


def plot_score_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: mean similarity score per (category × conflict type)."""
    pivot = df.groupby("category")[SCORE_COLS].mean()
    pivot.columns = CONFLICT_TYPES

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Mean similarity score by note category")
    ax.set_xlabel("Conflict type")
    ax.set_ylabel("Note category")
    savefig(fig, output_dir, "score_heatmap_by_category.png")


def plot_label_heatmap_by_category(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: % of documents labelled per (category × conflict type)."""
    pivot = df.groupby("category")[IS_COLS].mean() * 100
    pivot.columns = CONFLICT_TYPES

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
        vmin=0,
        vmax=100,
    )
    ax.set_title("% of documents labelled per note category")
    ax.set_xlabel("Conflict type")
    ax.set_ylabel("Note category")
    savefig(fig, output_dir, "label_heatmap_by_category.png")


def plot_score_correlations(df: pd.DataFrame, output_dir: Path) -> None:
    """Correlation matrix of similarity scores across conflict types."""
    corr = df[SCORE_COLS].corr()
    corr.index = corr.columns = CONFLICT_TYPES

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        linewidths=0.5,
        mask=mask,
    )
    ax.set_title("Score correlation across conflict types")
    savefig(fig, output_dir, "score_correlations.png")


def plot_moderator_scores(processed_path: Path, output_dir: Path) -> None:
    """Grouped violin: Moderator score dimensions per conflict type (paper fig:score_dist).

    Reads a processed pipeline JSON file and plots, for accepted instances only
    (moderator_score >= 4), the distribution of clinical_plausibility_score,
    temporal_appropriateness_score, clinical_significance_score, and moderator_score
    as four sub-plots — one per score dimension — with conflict type on the x-axis.

    Args:
        processed_path: Path to a processed pipeline JSON file.
        output_dir:     Directory to write moderator_score_dist.png.
    """
    with open(processed_path) as f:
        raw = json.load(f)

    score_dims = {
        "Clinical plausibility": "clinical_plausibility_score",
        "Temporal appropriateness": "temporal_appropriateness_score",
        "Clinical significance": "clinical_significance_score",
        "Overall (Moderator)": "moderator_score",
    }

    rows = []
    for item in raw:
        d = item["data"]
        if d.get("moderator_score") is None:
            continue
        rows.append(
            {
                "conflict_type": d.get("conflict_type", "unknown"),
                **{label: d.get(col) for label, col in score_dims.items()},
            }
        )
    df = pd.DataFrame(rows)

    accepted = df[df["Overall (Moderator)"] >= 4].copy()
    conflict_types = sorted(accepted["conflict_type"].unique())

    fig, axes = plt.subplots(1, len(score_dims), figsize=(16, 5), sharey=True)

    for ax, (dim_label, _) in zip(axes, score_dims.items()):
        sns.violinplot(
            data=accepted,
            x="conflict_type",
            y=dim_label,
            ax=ax,
            palette="Blues",
            inner="box",
            cut=0,
            order=conflict_types,
        )
        ax.set_title(dim_label, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Score (1–5)" if ax == axes[0] else "")
        ax.set_ylim(1, 5.3)
        ax.axhline(4, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    n_accepted = len(accepted)
    n_total = len(df)
    fig.suptitle(
        f"Moderator score distributions per conflict type  "
        f"(accepted instances only, n={n_accepted}/{n_total})",
        y=1.02,
    )
    fig.tight_layout()
    savefig(fig, output_dir, "moderator_score_dist.png")


def plot_co_occurrence(df: pd.DataFrame, output_dir: Path) -> None:
    """Symmetric co-occurrence heatmap: how often two types are both labelled."""
    n = len(CONFLICT_TYPES)
    matrix = np.zeros((n, n), dtype=int)

    for i, ct_i in enumerate(CONFLICT_TYPES):
        for j, ct_j in enumerate(CONFLICT_TYPES):
            matrix[i, j] = (df[f"is_{ct_i}"] & df[f"is_{ct_j}"]).sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=CONFLICT_TYPES,
        yticklabels=CONFLICT_TYPES,
        linewidths=0.5,
    )
    ax.set_title("Label co-occurrence (document count)")
    savefig(fig, output_dir, "label_co_occurrence.png")


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame) -> None:
    total = len(df)
    n_subjects = df["subject_id"].nunique()

    print("\n" + "=" * 70)
    print("EDA SUMMARY — document_conflict_classification.parquet")
    print("=" * 70)
    print(f"\nDocuments : {total:,}")
    print(f"Subjects  : {n_subjects:,}")
    print(f"Columns   : {list(df.columns)}\n")

    print("Label coverage:")
    for ct in CONFLICT_TYPES:
        n = df[f"is_{ct}"].sum()
        print(f"  {ct:20}: {n:5,}  ({n / total * 100:.1f}%)")

    print("\nBest-match distribution:")
    for ct, cnt in df["best_match"].value_counts().items():
        print(f"  {ct:20}: {cnt:5,}  ({cnt / total * 100:.1f}%)")

    print("\nNum conflict types per document:")
    for k, cnt in df["num_types"].value_counts().sort_index().items():
        print(f"  {k} type(s): {cnt:5,}  ({cnt / total * 100:.1f}%)")

    print("\nScore statistics:")
    print(
        df[SCORE_COLS]
        .describe()
        .rename(columns={f"score_{ct}": ct for ct in CONFLICT_TYPES})
        .to_string()
    )

    print("\nDocuments per note category:")
    print(df["category"].value_counts().to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA for document_conflict_classification.parquet")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("../data/document_conflict_classification.parquet"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("../data/eda"))
    parser.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="Processed pipeline JSON (e.g. processed/908a039_26092025.json). "
        "When provided, generates moderator_score_dist.png (paper fig:score_dist).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.data} ...")
    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    print_summary(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to {args.output_dir} ...")

    plot_label_coverage(df, args.output_dir)
    plot_num_types_distribution(df, args.output_dir)
    plot_best_match_distribution(df, args.output_dir)
    plot_score_distributions(df, args.output_dir)
    plot_score_heatmap(df, args.output_dir)
    plot_label_heatmap_by_category(df, args.output_dir)
    plot_score_correlations(df, args.output_dir)
    plot_co_occurrence(df, args.output_dir)

    if args.processed is not None:
        print(f"\nGenerating moderator score plot from {args.processed} ...")
        plot_moderator_scores(args.processed, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
