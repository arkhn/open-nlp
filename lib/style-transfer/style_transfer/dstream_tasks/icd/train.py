import os
from collections import Counter
from pathlib import Path

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

nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

ABS_PATH = Path(__file__).parent.parent.parent.parent.absolute()
DATA_PATH = ABS_PATH / "hf_datasets/mimic_iii_icd/data"


@hydra.main(
    version_base="1.3",
    config_path=str(ABS_PATH / "configs" / "ds_stream"),
    config_name="20.yaml",
)
def main(cfg: DictConfig):
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    wandb.init()
    set_seed(cfg.seed)
    # load necessary files
    train_ds = load_dataset("csv", data_files=str(DATA_PATH / "train_ds.csv"))
    test_ds = load_dataset("csv", data_files=str(DATA_PATH / "test_ds.csv"))
    icd9_descriptions = {
        line.split("\t")[0]: line.split("\t")[1][:-1]
        for line in open(str(DATA_PATH / "ICD9_descriptions")).readlines()
    }
    score_cols = list(train_ds["train"].to_pandas().filter(regex="eval_sem.*").columns)
    gen_cols = list(train_ds["train"].to_pandas().filter(regex="generation.*").columns)

    # load tokenizer, metrics, training args
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = sigmoid(predictions)
        predictions = predictions > cfg.threshold
        labels = labels.astype(float)
        current_metrics = metrics.compute(
            predictions=predictions.astype(float).reshape(-1), references=labels.reshape(-1)
        )
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

    def special_sampling(example):
        if cfg.dataset.name == "combined":
            return example["dataset_ids"] in ["0.06-0", "0.06-1-ofzh3aqu", "0.06-2-ofzh3aqu"]
        elif cfg.dataset.name == "combined-4":
            return example["dataset_ids"] in ["0.04-0", "0.04-1-mru97w7c", "0.04-2-mru97w7c"]
        elif cfg.dataset.name == "supsampling":
            return example["dataset_ids"] == "gold"

    train_ds = (
        train_ds.filter(lambda x: x["LABELS"] is not None)
        .filter(
            lambda example: (
                example["dataset_ids"] == cfg.dataset.name
                if cfg.dataset.name != "combined"
                and cfg.dataset.name != "supsampling"
                and cfg.dataset.name != "combined-4"
                else special_sampling(example)
            )
        )
        .map(preprocess_labels)
    )
    train_ds["train"] = datasets.concatenate_datasets(
        [train_ds["train"]] * 4 if cfg.dataset.name == "supsampling" else [train_ds["train"]]
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
        if cfg.dataset.name != "gold" and cfg.dataset.name != "supsampling":
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
    df: dict = {
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
