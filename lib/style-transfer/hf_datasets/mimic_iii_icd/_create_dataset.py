import json

import pandas as pd
import typer
import wandb
from tqdm import tqdm


def load_wandb_ds(ds_name):
    api = wandb.Api()
    test_dataset = api.artifact(ds_name)
    test_dataset.logged_by()
    json_file = json.load(test_dataset.files()[0].download(replace=True))
    json_config = json.loads(test_dataset.logged_by().json_config)
    stf_ratio = json_config["sft_ratio"]["value"]
    dpo_gen = json_config["dpo_gen"]["value"]
    checkpoint = (
        json_config["sem_model"]["value"]["checkpoint"].split("run-")[1].split("-")[0]
        if json_config["sem_model"]["value"]["checkpoint"]
        else None
    )
    return (
        pd.DataFrame(data=json_file["data"], columns=json_file["columns"]),
        stf_ratio,
        dpo_gen,
        checkpoint,
    )


def main(datasets: list[str]):
    hadm_ids_train = pd.read_csv("hadm_ids/gen.csv")
    hadm_ids_test = pd.read_csv("hadm_ids/test.csv")
    caml_train = pd.read_csv("raw/caml_train_full.csv")
    caml_test = pd.read_csv("raw/caml_test_full.csv")
    caml = pd.concat([caml_train, caml_test])
    caml = caml.rename(columns={"HADM_ID": "HADM_IDS"})
    train_datasets = []
    baseline_ds: pd.DataFrame

    for idx_dataset, dataset in enumerate(tqdm(datasets)):
        train_ds, sft_ratio, dpo_gen, checkpoint = load_wandb_ds(dataset)
        merged_ds = train_ds.merge(hadm_ids_train, on="ground_texts", how="inner")
        merged_ds = merged_ds.merge(caml, on="HADM_IDS", how="inner")
        merged_ds["dataset_ids"] = (
            f"{sft_ratio}-{dpo_gen}-{checkpoint}" if checkpoint else f"{sft_ratio}-{dpo_gen}"
        )
        if idx_dataset == 0:
            baseline_ds = merged_ds[
                [
                    "prompts",
                    "LABELS",
                    "dataset_ids",
                ]
            ]
        merged_ds = merged_ds[
            [
                "ground_texts",
                "generation_0",
                "generation_1",
                "generation_2",
                "generation_3",
                "eval_sem_scores_0",
                "eval_sem_scores_1",
                "eval_sem_scores_2",
                "eval_sem_scores_3",
                "HADM_IDS",
                "SUBJECT_ID",
                "LABELS",
                "dataset_ids",
            ]
        ]

        if idx_dataset == 0:
            gold_ds = merged_ds.copy()
            gold_ds["dataset_ids"] = "gold"
            train_datasets.append(gold_ds)

        merged_ds = merged_ds.drop("ground_texts", axis=1)
        train_datasets.append(merged_ds)
    test_ds, _, _, _ = load_wandb_ds(datasets[0].replace("-gen", "-test"))
    test_ds = test_ds.merge(hadm_ids_test, on="ground_texts", how="inner")
    test_ds = test_ds.merge(caml, on="HADM_IDS", how="inner")
    train_ds = pd.concat(train_datasets)
    print("splits: ", train_ds["dataset_ids"].unique())
    baseline_ds.to_csv("data/baseline_ds.csv")
    train_ds.to_csv("data/train_ds.csv")
    test_ds.to_csv("data/test_ds.csv")


if __name__ == "__main__":
    typer.run(main)
