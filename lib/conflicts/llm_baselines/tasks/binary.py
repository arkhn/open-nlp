"""Task 1: Binary conflict classification.

Given (doc_1, doc_2), predict CONFLICT or NO_CONFLICT.
Metric: macro-F1, precision, recall.
"""

import logging
import re

import openai
from llm_baselines.utils import call_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical NLP expert specialised in detecting contradictions
in electronic health records (EHRs). You will be given two clinical documents from the same patient.
Your task is to determine whether the two documents contain a factual conflict.

A conflict exists when two statements in the documents are mutually inconsistent for the same
patient (e.g., contradictory diagnoses, conflicting lab values, incompatible surgical history).

Respond with exactly one word: CONFLICT or NO_CONFLICT."""

USER_PROMPT = """Document 1 (timestamp: {ts1}):
{doc1}

---

Document 2 (timestamp: {ts2}):
{doc2}

---

Do these two documents contain a factual conflict? Answer CONFLICT or NO_CONFLICT."""


def predict(response: str) -> int:
    """Parse binary prediction from model response.

    Returns:
        1 if CONFLICT detected, 0 otherwise.
    """
    text = response.upper()
    if "NO_CONFLICT" in text or "NO CONFLICT" in text:
        return 0
    if "CONFLICT" in text:
        return 1
    if re.search(r"\bYES\b", text):
        return 1
    if re.search(r"\bNO\b", text):
        return 0
    return 0  # Default to no conflict if unparseable


def compute_metrics(labels: list[int], preds: list[int]) -> dict:
    """Compute precision, recall, F1 for binary conflict detection.

    Args:
        labels: Ground truth labels (1=conflict, 0=no_conflict).
        preds:  Predicted labels.

    Returns:
        Dict with precision, recall, f1, accuracy, support_positive.
    """
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    accuracy = accuracy_score(labels, preds) if labels else 0.0
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(accuracy), 4),
        "support_positive": sum(1 for lbl in labels if lbl == 1),
    }


def run_evaluation(
    examples: list[dict],
    client: openai.OpenAI,
    model_name: str,
) -> tuple[dict, list[dict]]:
    """Run binary classification evaluation for one model.

    Args:
        examples:   Binary classification examples from load_dataset.
        client:     OpenAI-compatible client.
        model_name: Model identifier.

    Returns:
        Tuple of (metrics dict, predictions list).
    """
    labels, preds, predictions = [], [], []

    for i, ex in enumerate(examples):
        prompt = USER_PROMPT.format(
            doc1=ex["doc1"],
            doc2=ex["doc2"],
            ts1=ex["ts1"],
            ts2=ex["ts2"],
        )
        response, success = call_model(client, model_name, SYSTEM_PROMPT, prompt, max_tokens=16)
        pred = predict(response) if success else 0

        labels.append(ex["label"])
        preds.append(pred)
        predictions.append(
            {
                "index": i,
                "label": ex["label"],
                "predicted": pred,
                "response": response,
                "success": success,
                "conflict_type": ex["conflict_type"],
            }
        )

        if (i + 1) % 10 == 0:
            log.info(f"  Binary [{model_name}]: {i+1}/{len(examples)} done")

    metrics = compute_metrics(labels, preds)

    # Per conflict type breakdown (positives only)
    per_type: dict[str, dict] = {}
    for ex, pred in zip(examples, preds):
        if ex["label"] == 0:
            continue
        ct = ex["conflict_type"]
        per_type.setdefault(ct, {"labels": [], "preds": []})
        per_type[ct]["labels"].append(ex["label"])
        per_type[ct]["preds"].append(pred)

    metrics["per_conflict_type"] = {
        ct: compute_metrics(v["labels"], v["preds"]) for ct, v in per_type.items()
    }

    return metrics, predictions
