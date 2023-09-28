import os
import sys
import time

import hydra
import wandb

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from commons.data.load_and_preprocess_dataset import load_and_preprocess_dataset  # noqa: E402
from commons.metrics.exact_match_ratio import exact_match_ratio  # noqa: E402
from commons.metrics.hamming_score import hamming_score  # noqa: E402
from commons.submission.to_logits import to_logits  # noqa: E402
from commons.submission.to_txt import to_txt  # noqa: E402
from omegaconf import DictConfig, omegaconf  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


def preprocess_function(e, tokenizer):
    """Preprocess the data.

    Args:
        e: Data entry.
        tokenizer: Tokenizer object.

    Returns:
        Dictionary: Dictionary with the preprocessed data.
    """

    CLS = "<s>"
    SEP = "</s>"
    EOS = "</s>"

    text = (
        CLS
        + " "
        + e["question"]
        + f" {SEP} "
        + f" {SEP} ".join([e[f"answer_{letter}"] for letter in ["a", "b", "c", "d", "e"]])
        + " "
        + EOS
    )

    res = tokenizer(text, truncation=True, max_length=512, padding="max_length")

    labels = [0.0] * 5

    for answer_id in e["correct_answers"]:
        labels[answer_id] = 1.0

    res["labels"] = labels

    return res


def preprocess_and_set_format(dataset, tokenizer):
    """Preprocess the dataset and set the format.

    Args:
        dataset: Dataset object.
        tokenizer: Tokenizer object.

    Returns:
        Dataset object.
    """
    dataset = dataset.map(preprocess_function, batched=False, fn_kwargs={"tokenizer": tokenizer})
    return dataset


def compute_metrics(p: EvalPrediction) -> dict:
    """Compute metrics for the task.

    Args:
        p: EvalPrediction object.

    Returns:
        Dictionary with the metrics.
    """

    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

    y_pred = to_logits(preds, 0.5)
    y_true = p.label_ids

    return {
        "hamming_score": hamming_score(y_pred, y_true),
        "exact_match": exact_match_ratio(y_pred, y_true),
    }


@hydra.main(config_name="default", config_path="configs", version_base="1.2")
def main(config: DictConfig):
    """Main function to train the model.

    Args:
        config: Hydra configuration object.

    """

    run = wandb.init(
        entity=config.logging.entity,
        project=config.logging.project,
        tags=config.logging.tags,
        settings=wandb.Settings(start_method="thread"),
    )
    container = omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    run.config.update(container)
    dataset_train, dataset_val, dataset_test, dataset_sub = load_and_preprocess_dataset()
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name_or_path, use_fast=True)

    dataset_train = preprocess_and_set_format(dataset_train, tokenizer)
    dataset_val = preprocess_and_set_format(dataset_val, tokenizer)
    dataset_test = preprocess_and_set_format(dataset_test, tokenizer)
    dataset_sub = preprocess_and_set_format(dataset_sub, tokenizer)

    labels_list = dataset_train.features["correct_answers"].feature.names
    id2label = {idx: label for idx, label in enumerate(labels_list)}
    num_labels = len(labels_list)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name_or_path,
        problem_type="multi_label_classification",
        num_labels=num_labels,
    )

    args = TrainingArguments(
        config.logging.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        num_train_epochs=config.training.epochs,
        weight_decay=config.training.weight_decay,
        push_to_hub=config.training.push_to_hub,
        eval_steps=config.training.eval_steps,
        logging_steps=config.training.logging_steps,
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model="exact_match",
        save_total_limit=3,
        report_to=["wandb"],
        lr_scheduler_type="constant",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate(eval_dataset=dataset_test, metric_key_prefix="test")

    predictions, _, _ = trainer.predict(dataset_sub)

    # generate submission file on deft test set
    run_name = run.name
    run_id = run.id
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    submission_file_path = f"submissions/submission-baseline_{run_name}_{run_id}_{timestamp}.txt"

    # convert predictions to txt file
    to_txt(to_logits(predictions, threshold=0.5), dataset_sub, submission_file_path, id2label)

    # log submission file to wandb
    artifact = wandb.Artifact("submission-file", type="submission")
    artifact.add_file(submission_file_path)
    run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
