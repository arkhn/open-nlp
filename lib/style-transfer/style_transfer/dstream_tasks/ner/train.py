import os
from collections import Counter
from pathlib import Path

import evaluate
import hydra
import nltk
import numpy as np
import wandb
from datasets import concatenate_datasets, load_dataset
from nltk.corpus import stopwords
from omegaconf import DictConfig, omegaconf
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    set_seed,
)
from transformers.integrations import WandbCallback

nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))
seqeval = evaluate.load("seqeval")


ABS_PATH = Path(__file__).parent.parent.parent.parent.absolute()
DATA_PATH = ABS_PATH / "hf_datasets/mimic_iii_ner/data"


@hydra.main(
    version_base="1.3",
    config_path=str(ABS_PATH / "configs" / "ds_stream"),
    config_name="ner_train.yaml",
)
def main(cfg: DictConfig):
    if cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
    wandb.init()
    set_seed(cfg.seed)

    # load necessary files
    datasets = (
        ["0.06-0", "0.06-1-ofzh3aqu", "0.06-2-ofzh3aqu"]
        if cfg.dataset.name == "combined"
        else [cfg.dataset.name]
    )
    train_ds = concatenate_datasets(
        [
            load_dataset("parquet", data_files=str(DATA_PATH / f"ner-{dataset}-train.parquet"))[
                "train"
            ]
            for dataset in datasets
        ]
    )

    test_ds = load_dataset(
        "parquet",
        data_files=str(DATA_PATH / "ner-gold-test.parquet"),
    )["train"]

    # load tokenizer, metrics, training args
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    current_max_f1 = [0]

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            pred = [all_labels[p] for p, l in zip(prediction, label) if l != -100]
            lab = [all_labels[l] for p, l in zip(prediction, label) if l != -100]
            true_predictions.append(pred)
            true_labels.append(lab)

        results = seqeval.compute(
            predictions=true_predictions,
            references=true_labels,
            scheme="IOB2",
        )

        current_max_f1[0] = max(current_max_f1[0], results["overall_f1"])

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "max_f1": current_max_f1[0],
            "classification_report": classification_report(
                true_labels, true_predictions, output_dict=True, mode="strict", scheme=IOB2
            ),
        }

    training_args = hydra.utils.instantiate(cfg.training_args)

    # preprocess labels
    def preprocess_labels(example):
        return example

    train_ds = train_ds.map(preprocess_labels)

    # collect all labels
    all_labels = sorted(list(set([token for tokens in test_ds["entities"] for token in tokens])))
    class2id = {class_: id for id, class_ in enumerate(all_labels)}
    id2class = {id: class_ for class_, id in class2id.items()}

    # select topk labels and apply tokenizer
    percentile = np.percentile(train_ds["score"], cfg.dataset.percentile)

    if cfg.dataset.random_sampling:
        train_ds = (
            train_ds.train_test_split(test_size=(1 - (cfg.dataset.percentile / 100)))["test"]
            if cfg.dataset.percentile > 0
            else train_ds
        )
    else:
        train_ds = train_ds.filter(lambda x: x["score"] >= percentile)

    # EDA: Labels distribution
    def count_labels(dataset):
        return [
            [label, sum(label in entities for entities in dataset["entities"])]
            for label in all_labels
        ]

    test_labels = count_labels(test_ds)
    train_labels = count_labels(train_ds)

    for name, labels in [("test", test_labels), ("train", train_labels)]:
        table = wandb.Table(data=labels, columns=["entities", "value"])
        wandb.log(
            {
                f"{name} labels distribution": wandb.plot.bar(
                    table, "entities", "value", title=f"{name.capitalize()} labels distribution"
                )
            }
        )

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model,
        num_labels=len(id2class),
        id2label=id2class,
        label2id=class2id,
    )

    class CustomWandbCallback(WandbCallback):
        def setup(self, args, state, model, **kwargs):
            super().setup(args, state, model, **kwargs)
            if state.is_world_process_zero:
                config_dict = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                )
                args.__dict__.update(config_dict)
                self._wandb.config.update(args, allow_val_change=True)

    train_ds.remove_columns(["score"])
    test_ds.remove_columns(["score"])

    # rename columns
    train_ds = train_ds.rename_column("words", "tokens")
    test_ds = test_ds.rename_column("words", "tokens")
    train_ds = train_ds.rename_column("entities", "ner_tags")
    test_ds = test_ds.rename_column("entities", "ner_tags")

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    train_ds = train_ds.map(
        lambda x: {
            "ner_tags": [
                class2id[tag.split(" ")[0]] if tag.split(" ")[0] in class2id else class2id["O"]
                for tag in x["ner_tags"]
            ]
        }
    )
    test_ds = test_ds.map(
        lambda x: {"ner_tags": [class2id[tag.split(" ")[0]] for tag in x["ner_tags"]]}
    )

    train_ds = train_ds.map(tokenize_and_align_labels, batched=True)
    test_ds = test_ds.map(tokenize_and_align_labels, batched=True)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[CustomWandbCallback],
    )

    # EDA: Labels distribution
    def eda_count_labels(dataset):
        return [
            (
                label,
                sum(
                    Counter(id2class[label] if label != -100 else "O" for label in labels)[label]
                    for labels in dataset["labels"]
                ),
            )
            for label in all_labels
        ]

    test_labels = eda_count_labels(test_ds)
    train_labels = eda_count_labels(train_ds)

    def log_distribution(name, data):
        table = wandb.Table(data=data, columns=["entities", "value"])
        wandb.log(
            {
                f"{name} labels distribution": wandb.plot.bar(
                    table, "entities", "value", title=f"{name} labels distribution"
                )
            }
        )

    log_distribution("test", test_labels)
    log_distribution("train", train_labels)

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
