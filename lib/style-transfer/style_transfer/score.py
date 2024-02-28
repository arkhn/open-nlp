import json
import logging
import os

import datasets
import hydra
import pandas as pd
import torch
import wandb
from fastchat.conversation import get_conv_template
from omegaconf import omegaconf
from sentence_transformers import InputExample, SentenceTransformer, util
from style_transfer.utils import EVAL_PROMPT

os.environ["WANDB_LOG_MODEL"] = "checkpoint"


@hydra.main(version_base="1.3", config_path="../configs", config_name="score.yaml")
def main(cfg):
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg,
    )
    with wandb.init(
        project="score-style-transfer",
        name=f"sft-ratio-{cfg.sft_ratio}_gen-ratio-{cfg.gen_ratio}"
        f"{'' if cfg.dpo_gen == 0 else f'_dpo{cfg.dpo_gen}'}",
        config=wandb.config,
    ) as run:
        gen_dataset = run.use_artifact(cfg.gen_dataset)
        json_file = json.load(gen_dataset.files()[0].download(replace=True))
        df = pd.DataFrame(data=json_file["data"], columns=json_file["columns"])
        gen_dataset = datasets.Dataset.from_pandas(df)

        test_dataset = run.use_artifact(cfg.test_dataset)
        json_file = json.load(test_dataset.files()[0].download(replace=True))
        df = pd.DataFrame(data=json_file["data"], columns=json_file["columns"])
        test_dataset = datasets.Dataset.from_pandas(df)

        gold_test_dataset = run.use_artifact(cfg.gold_test_dataset)
        json_file = json.load(gold_test_dataset.files()[0].download(replace=True))
        df = pd.DataFrame(data=json_file["data"], columns=json_file["columns"])
        gold_test_dataset = datasets.Dataset.from_pandas(df)

        logging.info("Loading the Semantic Model 🐈‍")
        sem_model = SentenceTransformer(cfg.sem_model.name)

        def sem_score(cfg, dataset, sem_model):
            score_dict = {}
            ground_enc = sem_model.encode(
                dataset["ground_texts"],
                batch_size=cfg.batch_size,
            )
            for seq in range(cfg.num_generated_sequences):
                prediction_enc = sem_model.encode(
                    dataset[f"generation_{seq}"],
                    batch_size=cfg.batch_size,
                )
                scores = [
                    util.cos_sim(ground_enc, pred_enc)[0][0].item()
                    for ground_enc, pred_enc in zip(ground_enc, prediction_enc)
                ]
                score_dict.setdefault(f"eval_sem_scores_{seq}", []).extend(scores)
            return score_dict

        def add_prompt(data_point):
            for seq in range(cfg.num_generated_sequences):
                data_point[f"eval_prompt_{seq}"] = str.format(
                    EVAL_PROMPT,
                    data_point["prompts"],
                    data_point[f"generation_{seq}"],
                    data_point["ground_texts"],
                )
                conv = get_conv_template("llama-2")
                conv.set_system_message("You are a fair evaluator language model.")
                conv.append_message(conv.roles[0], data_point[f"eval_prompt_{seq}"])
                conv.append_message(conv.roles[1], None)
                data_point[f"eval_prompt_{seq}"] = conv.get_prompt()
            return data_point

        gen_dataset = gen_dataset.map(
            add_prompt,
            batched=False,
        )
        gen_dataset = gen_dataset.train_test_split(train_size=cfg.sem_model.train_size)
        train_gen_dataset = gen_dataset["train"]
        gen_dataset = gen_dataset["test"].to_dict()

        logging.info("Training the Semantic Model 🐈‍")
        train_examples = []
        train_examples.extend(
            [
                InputExample(texts=[pred, ground_text], label=0)
                for pred, ground_text in zip(
                    train_gen_dataset["generation_0"], train_gen_dataset["ground_texts"]
                )
            ]
        )

        if cfg.sem_model.use_ground_truth:
            train_examples.extend(
                [
                    InputExample(texts=[ground_text, ground_text], label=1)
                    for ground_text in train_gen_dataset["ground_texts"]
                ]
            )
        train_gen_dataloader = torch.utils.data.DataLoader(
            train_examples,
            batch_size=cfg.sem_model.batch_size,
        )

        train_loss = hydra.utils.instantiate(cfg.sem_model.loss, sem_model)()
        sem_model.fit(
            train_objectives=[(train_gen_dataloader, train_loss)],
            epochs=cfg.sem_model.epochs,
            warmup_steps=cfg.sem_model.warmup_steps,
        )
        if cfg.sem_model.is_logged:
            sem_model.save(cfg.sem_model.path)
            run.log_artifact(cfg.sem_model.path, type="model")
        logging.info("Semantic Model Trained 🎉")

        test_dataset = test_dataset.map(
            add_prompt,
            batched=False,
        ).to_dict()

        gold_test_dataset = gold_test_dataset.map(
            add_prompt,
            batched=False,
        ).to_dict()

        eval_cols = [f"eval_sem_scores_{i}" for i in range(cfg.num_generated_sequences)]

        # gen log
        gen_scores = sem_score(cfg, gen_dataset, sem_model)
        gen_dataset.update(gen_scores)
        wandb.log({"gen_score_dataset": wandb.Table(dataframe=pd.DataFrame(data=gen_dataset))})

        # gold test log
        gold_test_scores = sem_score(cfg, gold_test_dataset, sem_model)
        gold_test_dataset.update(gold_test_scores)
        gold_test_dataset = pd.DataFrame(data=gold_test_dataset)
        gold_test_dataset["max_score"] = gold_test_dataset[eval_cols].max(axis=1)
        gold_test_dataset["min_score"] = gold_test_dataset[eval_cols].min(axis=1)
        gold_test_dataset["mean_score"] = gold_test_dataset[eval_cols].mean(axis=1)
        gold_test_dataset["std_score"] = gold_test_dataset[eval_cols].std(axis=1)
        wandb.log({"gold_test/max/mean": gold_test_dataset["max_score"].mean()})
        wandb.log({"gold_test/min/mean": gold_test_dataset["min_score"].mean()})
        wandb.log({"gold_test/mean/mean": gold_test_dataset["mean_score"].mean()})
        wandb.log({"gold_test/std/mean": gold_test_dataset["std_score"].mean()})
        wandb.log({"gold_test_score_dataset": wandb.Table(dataframe=gold_test_dataset)})

        # test log
        test_scores = sem_score(cfg, test_dataset, sem_model)
        test_dataset.update(test_scores)
        test_dataset = pd.DataFrame(data=test_dataset)
        test_dataset["max_score"] = test_dataset[eval_cols].max(axis=1)
        test_dataset["min_score"] = test_dataset[eval_cols].min(axis=1)
        test_dataset["mean_score"] = test_dataset[eval_cols].mean(axis=1)
        test_dataset["std_score"] = test_dataset[eval_cols].std(axis=1)
        wandb.log({"test/max/mean": test_dataset["max_score"].mean()})
        wandb.log({"test/min/mean": test_dataset["min_score"].mean()})
        wandb.log({"test/mean/mean": test_dataset["mean_score"].mean()})
        wandb.log({"test/std/mean": test_dataset["std_score"].mean()})
        wandb.log({"test_score_dataset": wandb.Table(dataframe=test_dataset)})


if __name__ == "__main__":
    main()
