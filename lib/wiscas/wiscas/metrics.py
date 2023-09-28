import json
import os
import warnings
from typing import Any, Optional

import datasets
import seqeval.metrics
import spacy
import typer
import wandb
from spacy.tokens import Doc, Span
from spacy.training.example import Example
from wandb.wandb_run import Run


def sample_to_doc(nlp: spacy.Language, sample: dict):
    """Transform sample into a spacy doc."""
    doc: Doc = nlp.make_doc(sample["text"])
    ents: list[Span] = []
    for entity in sample["entities"]:
        ent: Optional[Span] = doc.char_span(
            entity["start_char"],
            entity["end_char"],
            label=entity["label"],
            alignment_mode="expand",
        )
        if ent is not None:
            ents.append(ent)

    try:
        doc.set_ents(ents)
    # Fallback heuristic to deal with cases of overlapping entities
    except ValueError:  # pragma: no cover
        warnings.warn(
            "Unable to set all entities at once, falling back to setting entities one by one."
        )
        for ent in ents:
            doc.set_ents([ent], default="unmodified")

    return doc


def get_classification_report(examples: list[Example]) -> dict[str, Any]:
    """Compute the scores for the provided examples for the NER task."""
    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    for example in examples:
        if len(example.predicted) != len(example.reference):
            raise ValueError("Different tokenization between reference & prediction not supported.")
        y_true.append(
            [
                f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ != "" else "O"
                for token in example.reference
            ]
        )
        y_pred.append(
            [
                f"{token.ent_iob_}-{token.ent_type_}" if token.ent_type_ != "" else "O"
                for token in example.predicted
            ]
        )

    classification_report = seqeval.metrics.classification_report(
        y_true, y_pred, output_dict=True, scheme="IOB2", zero_division=0
    )
    classification_report["accuracy"] = seqeval.metrics.accuracy_score(y_true, y_pred)
    return classification_report


def log_wandb(classification_report: dict, run: Run) -> None:
    """Log classification report to wandb."""
    plots = {}
    for metric_name in ["precision", "recall", "f1-score", "support"]:
        data = []
        for label in classification_report:
            if label == "accuracy":
                continue
            data.append((label, classification_report[label][metric_name]))
        table = wandb.Table(data=data, columns=["label", "value"])  # type: ignore
        plots[metric_name] = wandb.plot.bar(  # type: ignore
            table, "label", "value", title=metric_name
        )

    run.log(plots)
    run.log({"accuracy": classification_report["accuracy"]})


def eval(test_dataset: datasets.Dataset, weak_dataset: datasets.Dataset):
    """Evaluate the weakly annotated dataset against the gold dataset."""
    nlp = spacy.blank("fr")
    spacy_examples = []
    for weak_sample, gold_sample in zip(weak_dataset, test_dataset):
        assert weak_sample["text"] == gold_sample["text"]
        doc_predicted = sample_to_doc(nlp=nlp, sample=weak_sample)
        doc_reference = sample_to_doc(nlp=nlp, sample=gold_sample)
        spacy_examples.append(Example(doc_predicted, doc_reference))

    metrics = get_classification_report(examples=spacy_examples)

    return metrics


def main(
    weak_dataset_path: str,
):
    weak_dataset = datasets.Dataset.from_parquet(os.path.join(weak_dataset_path, "dataset.parquet"))
    with open(os.path.join(weak_dataset_path, "config.json"), "r") as f:
        weak_dataset_config = json.load(f)

    gold_dataset: datasets.Dataset = datasets.Dataset.from_parquet(
        weak_dataset_config["params"]["parquet_dataset_path"]
    )
    max_samples = weak_dataset_config["params"]["n_test_examples"]
    if max_samples:
        gold_dataset = gold_dataset.select(range(max_samples))

    metrics = eval(test_dataset=gold_dataset, weak_dataset=weak_dataset)

    run = wandb.init(  # type: ignore
        project="wiscas",
        group="eval",
        job_type="analytics",
        config={"weak_dataset_config": weak_dataset_config},
    )
    log_wandb(classification_report=metrics, run=run)

    run.finish()


if __name__ == "__main__":
    typer.run(main)
