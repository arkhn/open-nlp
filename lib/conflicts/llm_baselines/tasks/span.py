"""Task 2: Span detection.

Given a conflicting pair, predict the exact text spans in conflict in each document.
Metric: character-level span F1 averaged over all gold spans.
"""

import json
import logging
import re

import openai
from llm_baselines.utils import call_model

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical NLP expert specialised in detecting contradictions
in electronic health records (EHRs). You will be given two clinical documents from the same patient
that are known to contain a factual conflict.

Your task is to identify the exact text spans in each document that are in conflict.

Respond in JSON format:
{
  "span_doc1": "<exact verbatim text from Document 1 that is part of the conflict>",
  "span_doc2": "<exact verbatim text from Document 2 that is part of the conflict>"
}

If a conflict only appears in one document, set the other span to an empty string."""

USER_PROMPT = """Document 1 (timestamp: {ts1}):
{doc1}

---

Document 2 (timestamp: {ts2}):
{doc2}

---

These documents contain a factual conflict. Identify the conflicting spans."""


def char_f1(gold_start: int, gold_end: int, pred_text: str, doc: str) -> float:
    """Compute character-level F1 between a gold span and predicted text.

    Finds all occurrences of pred_text in doc and takes the best overlap
    with the gold span.

    Args:
        gold_start: Start character index of gold span in doc.
        gold_end:   End character index of gold span in doc.
        pred_text:  Predicted span text (exact string).
        doc:        Full document text.

    Returns:
        F1 score in [0, 1].
    """
    if not pred_text:
        return 0.0

    gold_len = gold_end - gold_start
    if gold_len <= 0:
        return 0.0

    best_f1 = 0.0
    pred_len = len(pred_text)
    start = 0
    while True:
        idx = doc.find(pred_text, start)
        if idx == -1:
            break
        overlap = max(0, min(gold_end, idx + pred_len) - max(gold_start, idx))
        if overlap > 0:
            prec = overlap / pred_len
            rec = overlap / gold_len
            f1 = 2 * prec * rec / (prec + rec)
            best_f1 = max(best_f1, f1)
        start = idx + 1

    return best_f1


def parse_response(response: str) -> tuple[str, str]:
    """Parse span predictions from model JSON response.

    Args:
        response: Raw model response, expected to contain JSON with
                  span_doc1 and span_doc2 fields.

    Returns:
        Tuple of (span_doc1_text, span_doc2_text).
    """
    try:
        match = re.search(r"\{[^}]+\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return str(data.get("span_doc1", "")), str(data.get("span_doc2", ""))
    except (json.JSONDecodeError, ValueError):
        pass
    return "", ""


def run_evaluation(
    examples: list[dict],
    client: openai.OpenAI,
    model_name: str,
) -> tuple[dict, list[dict]]:
    """Run span detection evaluation for one model.

    Args:
        examples:   Span detection examples from load_dataset.
        client:     OpenAI-compatible client.
        model_name: Model identifier.

    Returns:
        Tuple of (metrics dict, predictions list).
    """
    all_f1s: list[float] = []
    predictions: list[dict] = []

    for i, ex in enumerate(examples):
        prompt = USER_PROMPT.format(
            doc1=ex["doc1"],
            doc2=ex["doc2"],
            ts1=ex["ts1"],
            ts2=ex["ts2"],
        )
        response, success = call_model(client, model_name, SYSTEM_PROMPT, prompt, max_tokens=256)

        span_f1s: list[float] = []
        pred_span1, pred_span2 = "", ""
        if success:
            pred_span1, pred_span2 = parse_response(response)
            for gold_start, gold_end, _ in ex["gold_spans_doc1"]:
                span_f1s.append(char_f1(gold_start, gold_end, pred_span1, ex["doc1"]))
            for gold_start, gold_end, _ in ex["gold_spans_doc2"]:
                span_f1s.append(char_f1(gold_start, gold_end, pred_span2, ex["doc2"]))

        avg_f1 = sum(span_f1s) / len(span_f1s) if span_f1s else 0.0
        all_f1s.append(avg_f1)
        predictions.append(
            {
                "index": i,
                "pred_span_doc1": pred_span1,
                "pred_span_doc2": pred_span2,
                "gold_spans_doc1": ex["gold_spans_doc1"],
                "gold_spans_doc2": ex["gold_spans_doc2"],
                "span_f1": round(avg_f1, 4),
                "response": response,
                "success": success,
                "conflict_type": ex["conflict_type"],
            }
        )

        if (i + 1) % 10 == 0:
            log.info(f"  Span [{model_name}]: {i+1}/{len(examples)} done")

    avg_span_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
    per_type: dict[str, list[float]] = {}
    for ex_f1, ex in zip(all_f1s, examples):
        per_type.setdefault(ex["conflict_type"], []).append(ex_f1)

    return {
        "span_f1": round(avg_span_f1, 4),
        "per_conflict_type": {ct: round(sum(v) / len(v), 4) for ct, v in per_type.items()},
    }, predictions
