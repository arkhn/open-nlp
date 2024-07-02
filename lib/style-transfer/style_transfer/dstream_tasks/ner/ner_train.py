import os
from collections import Counter

import evaluate
import hydra
import nltk
import numpy as np
import wandb
from datasets import load_dataset
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

os.environ["WANDB_PROJECT"] = "ner-style-transfer"
os.environ["WANDB_OFFLINE"] = "true"
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))
seqeval = evaluate.load("seqeval")


@hydra.main(version_base="1.3", config_path="../configs", config_name="ner_train.yaml")
def main(cfg: DictConfig):
    if cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
    wandb.init()
    set_seed(cfg.seed)

    # load necessary files
    train_ds = load_dataset(
        "parquet",
        data_files=f"hf_datasets/mimic_iii_ner/data/ner-{cfg.dataset.name}-train.parquet",
    )["train"]

    test_ds = load_dataset(
        "parquet",
        data_files="hf_datasets/mimic_iii_ner/data/ner-gold-test.parquet",
    )["train"]

    # load tokenizer, metrics, training args
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    current_max_f1 = [0]

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

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

    if cfg.dataset.random_sampling:
        train_ds = train_ds.sort("score")

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
                dict_conf = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                )
                for k_param, v_params in dict_conf.items():
                    setattr(args, k_param, v_params)
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

    tokens_size_cumulated = np.array([0])

    def add_ner_tags(example):
        tokens_size_cumulated[0] += len(
            [token for token in example["labels"] if token != -100 and token != class2id["O"]]
        )
        example["cumulative_tokens"] = tokens_size_cumulated[0]
        return example

    train_ds = train_ds.map(add_ner_tags)
    train_ds = train_ds.filter(lambda x: x["cumulative_tokens"] < cfg.dataset.max_tokens)
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

    # EDA
    # labels distribution
    test_labels = [
        [
            labels_set,
            sum(
                [
                    Counter([id2class[label] if label != -100 else "O" for label in labels])[
                        labels_set
                    ]
                    for labels in test_ds["labels"]
                ]
            ),
        ]
        for labels_set in all_labels
    ]

    train_labels = [
        [
            labels_set,
            sum(
                [
                    Counter(id2class[label] if label != -100 else "O" for label in labels)[
                        labels_set
                    ]
                    for labels in train_ds["labels"]
                ]
            ),
        ]
        for labels_set in all_labels
    ]
    table = wandb.Table(data=test_labels, columns=["entities", "value"])
    wandb.log(
        {
            "test labels distributions": wandb.plot.bar(
                table, "entities", "value", title="test labels distribution"
            )
        }
    )

    table = wandb.Table(data=train_labels, columns=["entities", "value"])
    wandb.log(
        {
            "train labels distributions": wandb.plot.bar(
                table, "entities", "value", title="train labels distribution"
            )
        }
    )
    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
