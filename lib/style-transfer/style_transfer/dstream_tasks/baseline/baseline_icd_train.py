import os
from collections import Counter

import datasets
import evaluate
import hydra
import nltk
import numpy as np
import wandb
from datasets import load_dataset
from nltk.corpus import stopwords
from omegaconf import DictConfig, omegaconf
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    set_seed,
)
from transformers.integrations import WandbCallback

os.environ["WANDB_PROJECT"] = "icd-style-transfer"
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))


@hydra.main(version_base="1.3", config_path="../configs", config_name="baseline_icd_train.yaml")
def main(cfg: DictConfig):
    if cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
    wandb.init()
    set_seed(cfg.seed)
    # load necessary files

    train_ds = load_dataset("csv", data_files="hf_datasets/mimic_iii_icd/data/baseline_ds.csv")
    test_ds = load_dataset("csv", data_files="hf_datasets/mimic_iii_icd/data/test_ds.csv")
    icd9_descriptions: dict = {
        line.split("\t")[0]: line.split("\t")[1][:-1]
        for line in open("hf_datasets/mimic_iii_icd/data/ICD9_descriptions").readlines()
    }

    # load tokenizer, metrics, training args
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = sigmoid(predictions)
        predictions = (predictions > 0.5).astype(int).reshape(-1)
        current_metrics = metrics.compute(
            predictions=predictions, references=labels.astype(int).reshape(-1)
        )
        return current_metrics

    training_args = hydra.utils.instantiate(cfg.training_args)

    def preprocess(example):
        example["LABELS"] = [
            label.strip() if not cfg.dataset.precision else label.strip().split(".")[0]
            for label in example["LABELS"].split(";")
        ]
        example["LABELS"] = [
            icd9_descriptions[label]
            for label in example["LABELS"]
            if label in icd9_descriptions.keys()
        ]
        example["prompts"] = example["prompts"].split("Keywords: ")[1].removesuffix("[/INST]\n")
        return example

    train_ds = train_ds.filter(lambda x: x["LABELS"] is not None).map(preprocess)
    test_ds = test_ds.map(preprocess)

    # collect all labels
    all_labels = [label for labels in test_ds["train"]["LABELS"] for label in labels]
    all_labels = (
        Counter(all_labels).most_common(cfg.dataset.topk) if cfg.dataset.topk != -1 else all_labels
    )
    all_labels = [label for label, _ in all_labels]
    all_labels = list(set(all_labels))
    class2id = {class_: id for id, class_ in enumerate(all_labels)}
    id2class = {id: class_ for class_, id in class2id.items()}

    # select topk labels and apply tokenizer
    def select_topk(example):
        example["LABELS"] = [label for label in example["LABELS"] if label in all_labels]
        return example

    def tokenize(example):
        labels = [0.0 for _ in range(len(all_labels))]
        for label in example["LABELS"]:
            label_id = class2id[label]
            labels[label_id] = 1.0

        input_ids = tokenizer(example["prompts"], truncation=True)["input_ids"]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    train_ds = (
        train_ds.map(select_topk)
        .filter(lambda x: x["LABELS"] != [])
        .map(tokenize)
        .filter(lambda x: x["input_ids"] != [])
    )
    test_ds = test_ds.map(select_topk).filter(lambda x: x["LABELS"] != []).map(tokenize)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # EDA
    # labels distribution
    test_labels = [sum(binary) for binary in zip(*test_ds["train"]["labels"])]
    test_labels = [(id2class[idx_l], l) for idx_l, l in enumerate(test_labels)]
    train_labels = [sum(binary) for binary in zip(*train_ds["train"]["labels"])]
    train_labels = [(id2class[idx_l], l) for idx_l, l in enumerate(train_labels)]
    table = wandb.Table(data=test_labels, columns=["ICD", "value"])
    wandb.log(
        {
            "test labels distributions": wandb.plot.bar(
                table, "ICD", "value", title="test labels distribution"
            )
        }
    )

    table = wandb.Table(data=train_labels, columns=["ICD", "value"])
    wandb.log(
        {
            "train labels distributions": wandb.plot.bar(
                table, "ICD", "value", title="train labels distribution"
            )
        }
    )

    # size
    train_data = [
        [
            len(train_ds),
            len(
                [label for labels in train_ds["train"]["labels"] for label in labels if label == 1]
            ),
        ]
    ]
    table = wandb.Table(data=train_data, columns=["documents", "labels"])
    wandb.log(
        {
            "train size": wandb.plot.scatter(
                table,
                "documents",
                "labels",
                title="train size",
            )
        }
    )

    test_data = [
        [
            len(test_ds["train"]),
            len([label for labels in test_ds["train"]["labels"] for label in labels if label == 1]),
        ]
    ]
    table = wandb.Table(data=test_data, columns=["labels", "documents"])
    wandb.log(
        {
            "test size": wandb.plot.scatter(
                table,
                "labels",
                "documents",
                title="test size",
            )
        }
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model,
        num_labels=len(all_labels),
        id2label=id2class,
        label2id=class2id,
        problem_type="multi_label_classification",
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds["train"],
        eval_dataset=datasets.Dataset.from_pandas(test_ds["train"].to_pandas()),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[CustomWandbCallback],
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
