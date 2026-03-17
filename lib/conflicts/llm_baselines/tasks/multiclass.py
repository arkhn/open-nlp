"""Task 3: Multiclass conflict type classification.

Given a conflicting pair, predict the conflict type among four clinically defined categories.
Metric: macro-F1, per-type F1.
"""

import logging

import openai
from conflicts.core.constants import SPECIALIZED_CONFLICT_TYPES as CONFLICT_TYPES
from llm_baselines.utils import call_model
from sklearn.metrics import classification_report

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical NLP expert specialised in detecting contradictions
in electronic health records (EHRs). You will be given two clinical documents from the same patient
that contain a factual conflict. Your task is to classify the conflict into exactly one of the
following four types:

  temporality                  — Conflicts involving patient characteristics that should remain
                                 immutable over time (e.g. blood type, genetic markers, allergies,
                                 adult height).

  clinical_history_antecedents — Inconsistencies between current documentation and permanent
                                 historical events (e.g. surgical history omission, conflicting
                                 transplant status, undocumented implanted devices).

  biological_marker_monitoring — Biomarker values or trajectories incompatible with documented
                                 procedures or physiological constraints (e.g. normal organ
                                 function after organ removal, unchanged troponin after stenting).

  pre_post_care_evolution      — Contradictions between admission and discharge documentation
                                 (e.g. diagnosis changes without clinical justification,
                                 medication reconciliation failures).

Respond with exactly one of the four type names above, with no additional text."""

USER_PROMPT = """Document 1 (timestamp: {ts1}):
{doc1}

---

Document 2 (timestamp: {ts2}):
{doc2}

---

What is the conflict type? Answer with one of: temporality, clinical_history_antecedents,
biological_marker_monitoring, pre_post_care_evolution."""

_ALIASES = {
    "temporality": ["temporalit", "immutable", "allerg", "blood type"],
    "clinical_history_antecedents": ["clinical_history", "surgical", "transplant", "implant"],
    "biological_marker_monitoring": ["biological_marker", "biomarker", "lab value", "troponin"],
    "pre_post_care_evolution": ["pre_post", "admission", "discharge", "reconciliation"],
}


def predict(response: str) -> str:
    """Parse conflict type prediction from model response.

    Returns:
        One of the four CONFLICT_TYPES strings, or "unknown" if unparseable.
    """
    text = response.lower().strip()
    for ct in CONFLICT_TYPES:
        if ct in text:
            return ct
    for ct, keywords in _ALIASES.items():
        if any(kw in text for kw in keywords):
            return ct
    return "unknown"


def compute_metrics(labels: list[str], preds: list[str]) -> dict:
    """Compute per-type and macro-averaged F1 for multiclass classification.

    Args:
        labels: Ground truth conflict type strings.
        preds:  Predicted conflict type strings.

    Returns:
        Dict with macro_f1, accuracy, per_type F1/precision/recall/support.
    """
    report = classification_report(
        labels, preds, labels=CONFLICT_TYPES, output_dict=True, zero_division=0
    )
    macro = report.get("macro avg", {})
    per_type = {
        ct: {
            "precision": round(report.get(ct, {}).get("precision", 0.0), 4),
            "recall": round(report.get(ct, {}).get("recall", 0.0), 4),
            "f1": round(report.get(ct, {}).get("f1-score", 0.0), 4),
            "support": int(report.get(ct, {}).get("support", 0)),
        }
        for ct in CONFLICT_TYPES
    }
    return {
        "macro_f1": round(macro.get("f1-score", 0.0), 4),
        "accuracy": round(report.get("accuracy", 0.0), 4),
        "per_type": per_type,
    }


def run_evaluation(
    examples: list[dict],
    client: openai.OpenAI,
    model_name: str,
) -> tuple[dict, list[dict]]:
    """Run multiclass conflict type classification for one model.

    Only uses positive (conflict) examples since negative examples have no conflict type.

    Args:
        examples:   Span examples from load_dataset (which carry conflict_type labels).
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
        response, success = call_model(client, model_name, SYSTEM_PROMPT, prompt, max_tokens=32)
        pred = predict(response) if success else "unknown"

        labels.append(ex["conflict_type"])
        preds.append(pred)
        predictions.append(
            {
                "index": i,
                "gold_type": ex["conflict_type"],
                "predicted_type": pred,
                "response": response,
                "success": success,
                "correct": ex["conflict_type"] == pred,
            }
        )

        if (i + 1) % 10 == 0:
            log.info(f"  Multiclass [{model_name}]: {i+1}/{len(examples)} done")

    return compute_metrics(labels, preds), predictions
