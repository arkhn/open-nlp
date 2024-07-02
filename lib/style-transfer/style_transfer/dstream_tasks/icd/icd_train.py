import os
from collections import Counter

import datasets
import evaluate
import hydra
import nltk
import numpy as np
import pandas as pd
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
    train_ds = load_dataset("csv", data_files="hf_datasets/mimic_iii_icd/data/train_ds.csv")
    test_ds = load_dataset("csv", data_files="hf_datasets/mimic_iii_icd/data/test_ds.csv")
    icd9_descriptions = {
        line.split("\t")[0]: line.split("\t")[1][:-1]
        for line in open("hf_datasets/mimic_iii_icd/data/ICD9_descriptions").readlines()
    }
    df = pd.read_csv("hf_datasets/mimic_iii_icd/data/train_ds.csv")
    score_cols = list(df.filter(regex="eval_sem.*").columns)
    gen_cols = list(df.filter(regex="generation.*").columns)
    del df

    # load tokenizer, metrics, training args
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    current_max_f1 = [0]

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = sigmoid(predictions)
        predictions = (predictions > 0.5).astype(int).reshape(-1)
        current_metrics = metrics.compute(
            predictions=predictions, references=labels.astype(int).reshape(-1)
        )
        current_max_f1[0] = max(current_max_f1[0], current_metrics["f1"])
        current_metrics["max_f1"] = current_max_f1[0]
        return current_metrics

    training_args = hydra.utils.instantiate(cfg.training_args)

    # preprocess labels
    def preprocess_labels(example):
        example["LABELS"] = [
            label.strip() if not cfg.dataset.precision else label.strip().split(".")[0]
            for label in example["LABELS"].split(";")
        ]
        example["LABELS"] = [
            icd9_descriptions[label]
            for label in example["LABELS"]
            if label in icd9_descriptions.keys()
        ]
        return example

    train_ds = (
        (
            train_ds.filter(lambda x: x["LABELS"] is not None)
            .filter(lambda example: example["dataset_ids"] == cfg.dataset.name)
            .map(preprocess_labels)
        )
        if cfg.dataset.name != "all"
        else train_ds.filter(lambda x: x["LABELS"] is not None)
        .filter(lambda x: x["dataset_ids"] != "gold" and "0.04" not in x["dataset_ids"])
        .map(preprocess_labels)
    )
    test_ds = test_ds.map(preprocess_labels)

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

    percentile = np.percentile(train_ds["train"][score_cols[0]], cfg.dataset.percentile)

    def tokenize_and_filter_train(example):
        labels = [0.0 for _ in range(len(all_labels))]
        for label in example["LABELS"]:
            label_id = class2id[label]
            labels[label_id] = 1.0

        texts = []
        score = []
        if cfg.dataset.name != "gold":
            for score_col, gen_col in zip(score_cols, gen_cols):
                score.append(
                    1 if example[score_col] >= percentile or cfg.dataset.random_sampling else 0
                )
                texts.append(example[gen_col])
        else:
            score.append(1)
            texts.append(example["ground_texts"])
        input_ids = [tokenizer(text, truncation=True)["input_ids"] for text in texts]
        return {
            "input_ids": input_ids,
            "labels": [labels] * len(texts),
            "scores": score,
        }

    def tokenize_and_filter_test(example):
        labels = [0.0 for _ in range(len(all_labels))]
        for label in example["LABELS"]:
            label_id = class2id[label]
            labels[label_id] = 1.0
        text = tokenizer(example["ground_texts"], truncation=True)
        return {"input_ids": text["input_ids"], "labels": labels}

    train_ds = (
        train_ds.map(select_topk)
        .filter(lambda x: x["LABELS"] != [])
        .map(tokenize_and_filter_train)
        .filter(lambda x: x["input_ids"] != [])
        .select_columns(["input_ids", "labels", "scores"])
    )
    df = {
        "input_ids": [],
        "labels": [],
        "scores": [],
    }
    for example in train_ds["train"]:
        for input_id, label, score in zip(
            example["input_ids"], example["labels"], example["scores"]
        ):
            df["input_ids"].append(input_id)
            df["labels"].append(label)
            df["scores"].append(score)
    train_ds = (
        datasets.Dataset.from_pandas(pd.DataFrame(df))
        .filter(lambda x: x["scores"] == 1)
        .select_columns(["input_ids", "labels"])
    )
    test_ds = (
        test_ds.map(select_topk)
        .filter(lambda x: x["LABELS"] != [])
        .map(tokenize_and_filter_test)
        .select_columns(["input_ids", "labels"])
    )

    if cfg.dataset.random_sampling:
        train_ds = train_ds.train_test_split(test_size=(1 - (cfg.dataset.percentile / 100)))["test"]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # EDA
    # labels distribution
    test_labels = [sum(binary) for binary in zip(*test_ds["train"]["labels"])]
    test_labels = [(id2class[idx_l], l) for idx_l, l in enumerate(test_labels)]
    train_labels = [sum(binary) for binary in zip(*train_ds["labels"])]
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
            len([label for labels in train_ds["labels"] for label in labels if label == 1]),
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
        train_dataset=train_ds,
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
