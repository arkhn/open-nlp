"""
benchmark.py — Evaluate SOTA models on the Conflict Corpus for three tasks.

TASKS
-----
  1. Binary classification  — given (doc_1, doc_2), predict CONFLICT or NO_CONFLICT.
     Metric: macro-F1, precision, recall.

  2. Span detection         — given a conflicting pair, predict the exact text spans
     that are in conflict in each document.
     Metric: character-level span F1 (precision, recall, F1) averaged over all gold spans.

  3. Multiclass classification — given a conflicting pair, predict the conflict type
     among the four clinically defined categories.
     Metric: macro-F1, per-type F1.

DATASET CONSTRUCTION
--------------------
  Loaded from the processed JSON files in data/processed/ (Label Studio format).
  Only items with best_conflict=True are used:
    Positive examples : (doc_1,  doc_2)  — synthetically modified conflict pair
    Negative examples : (orig_doc_1, orig_doc_2) — original unmodified pair (no conflict)
  This gives a perfectly balanced binary dataset.

MODELS
------
  All models use the OpenAI-compatible chat API. Configure base_url and api_key_env
  per model in configs/benchmark.yaml. Models without a compatible endpoint
  (e.g. Gemini) can be routed via a proxy.

USAGE
-----
  # Evaluate one model (override from CLI):
  python -m llm_baselines.benchmark model.name=openai/o3-mini

  # Evaluate all models defined in config:
  python -m llm_baselines.benchmark

  # Change dataset path:
  python -m llm_baselines.benchmark data.processed_dir=processed/

OUTPUT
------
  outputs/<date>/<time>/
    results_summary.json   — per-model binary + span F1 table
    <model>_predictions.json — raw per-sample predictions
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
from llm_baselines.tasks import binary, multiclass, span
from llm_baselines.utils import make_client
from omegaconf import DictConfig

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(processed_dir: str) -> tuple[list[dict], list[dict]]:
    """Load binary classification and span detection examples from processed JSON files.

    Scans all .json files in processed_dir. For each item with best_conflict=True:
      - Positive example: (doc_1, doc_2) with gold conflict spans from annotations
      - Negative example: (orig_doc_1, orig_doc_2) with no spans

    Args:
        processed_dir: Path to directory containing processed Label Studio JSON files.

    Returns:
        Tuple of (binary_examples, span_examples).
          binary_examples: list of dicts with keys:
            doc1, doc2, ts1, ts2, label (1=conflict, 0=no_conflict), conflict_type
          span_examples: list of dicts with keys:
            doc1, doc2, ts1, ts2, gold_spans_doc1, gold_spans_doc2, conflict_type
            where gold_spans_doc* are lists of (start, end, text) tuples.
    """
    processed_path = REPO_ROOT / processed_dir
    if processed_path.is_file():
        json_files = [processed_path]
    else:
        json_files = sorted(processed_path.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {processed_path}")

    binary_examples: list[dict] = []
    span_examples: list[dict] = []

    for json_file in json_files:
        with open(json_file) as f:
            items = json.load(f)

        for item in items:
            data = item["data"]
            if str(data.get("best_conflict")) != "True":
                continue

            ts1 = str(data.get("timestamp_1", "Unknown"))
            ts2 = str(data.get("timestamp_2", "Unknown"))
            conflict_type = data.get("conflict_type", "unknown")

            # Positive example: modified conflict pair
            binary_examples.append(
                {
                    "doc1": data["doc_1"],
                    "doc2": data["doc_2"],
                    "ts1": ts1,
                    "ts2": ts2,
                    "label": 1,
                    "conflict_type": conflict_type,
                    "source_file": json_file.name,
                }
            )

            # Negative example: original unmodified pair
            binary_examples.append(
                {
                    "doc1": data["orig_doc_1"],
                    "doc2": data["orig_doc_2"],
                    "ts1": ts1,
                    "ts2": ts2,
                    "label": 0,
                    "conflict_type": conflict_type,
                    "source_file": json_file.name,
                }
            )

            # Span detection: extract gold spans per document
            gold_spans_doc1, gold_spans_doc2 = [], []
            for result in item.get("annotations", [{}])[0].get("result", []):
                val = result.get("value", {})
                ann_span = (val.get("start", 0), val.get("end", 0), val.get("text", ""))
                if result.get("to_name") == "doc_1":
                    gold_spans_doc1.append(ann_span)
                elif result.get("to_name") == "doc_2":
                    gold_spans_doc2.append(ann_span)

            span_examples.append(
                {
                    "doc1": data["doc_1"],
                    "doc2": data["doc_2"],
                    "ts1": ts1,
                    "ts2": ts2,
                    "gold_spans_doc1": gold_spans_doc1,
                    "gold_spans_doc2": gold_spans_doc2,
                    "conflict_type": conflict_type,
                    "source_file": json_file.name,
                }
            )

    log.info(
        f"Loaded {len(binary_examples)} binary examples "
        f"({len(binary_examples)//2} positive + {len(binary_examples)//2} negative) "
        f"and {len(span_examples)} span examples from {len(json_files)} file(s)"
    )
    return binary_examples, span_examples


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary_table(results: dict[str, dict]) -> None:
    """Print a formatted summary table of all model results.

    Args:
        results: {model_name: {binary: {...}, span: {...}, multiclass: {...}}} dict.
    """
    header = (
        f"{'Model':<40} {'Bin-P':>6} {'Bin-R':>6} {'Bin-F1':>7} " f"{'Span-F1':>8} {'MC-F1':>7}"
    )
    print("\n" + "=" * len(header))
    print("BENCHMARK RESULTS — Conflict Corpus")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model, res in results.items():
        b = res.get("binary", {})
        s = res.get("span", {})
        m = res.get("multiclass", {})
        short = model.split("/")[-1][:38]
        print(
            f"{short:<40} "
            f"{b.get('precision', 0):>6.3f} "
            f"{b.get('recall', 0):>6.3f} "
            f"{b.get('f1', 0):>7.3f} "
            f"{s.get('span_f1', 0):>8.3f} "
            f"{m.get('macro_f1', 0):>7.3f}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Evaluate all configured models on the Conflict Corpus."""

    log.info("Loading dataset...")
    binary_examples, span_examples = load_dataset(cfg.data.processed_dir)

    # Subsample if configured (useful for quick smoke tests)
    if cfg.data.get("max_examples"):
        n = cfg.data.max_examples
        binary_examples = binary_examples[: n * 2]  # keep balanced
        span_examples = span_examples[:n]
        log.info(
            f"Subsampled to {len(binary_examples)} binary / {len(span_examples)} span examples"
        )

    all_results: dict[str, dict] = {}
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    for model_cfg in cfg.models:
        model_name = model_cfg.name
        log.info(f"\n{'='*60}\nEvaluating: {model_name}\n{'='*60}")

        client = make_client(model_cfg)
        model_results: dict[str, dict] = {}
        model_predictions: dict[str, list] = {}

        # Task 1: Binary classification
        log.info(f"Task 1: Binary classification ({len(binary_examples)} examples)")
        binary_metrics, binary_preds = binary.run_evaluation(binary_examples, client, model_name)
        model_results["binary"] = binary_metrics
        model_predictions["binary"] = binary_preds
        log.info(
            f"  Binary F1={binary_metrics['f1']:.3f}  "
            f"P={binary_metrics['precision']:.3f}  R={binary_metrics['recall']:.3f}"
        )

        # Task 2: Span detection
        log.info(f"Task 2: Span detection ({len(span_examples)} examples)")
        span_metrics, span_preds = span.run_evaluation(span_examples, client, model_name)
        model_results["span"] = span_metrics
        model_predictions["span"] = span_preds
        log.info(f"  Span F1={span_metrics['span_f1']:.3f}")

        # Task 3: Multiclass conflict type classification
        log.info(f"Task 3: Multiclass classification ({len(span_examples)} examples)")
        mc_metrics, mc_preds = multiclass.run_evaluation(span_examples, client, model_name)
        model_results["multiclass"] = mc_metrics
        model_predictions["multiclass"] = mc_preds
        log.info(f"  Multiclass macro-F1={mc_metrics['macro_f1']:.3f}")

        all_results[model_name] = model_results

        # Save per-model predictions
        safe_name = model_name.replace("/", "_")
        pred_path = output_dir / f"{safe_name}_predictions.json"
        with open(pred_path, "w") as f:
            json.dump(model_predictions, f, indent=2)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_binary_examples": len(binary_examples),
        "num_span_examples": len(span_examples),
        "results": all_results,
    }
    summary_path = output_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary_table(all_results)
    log.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
