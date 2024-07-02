import logging
import os

import hydra
import pandas as pd
import torch
import wandb
from omegaconf import omegaconf
from sentence_transformers import InputExample, util

os.environ["WANDB_LOG_MODEL"] = "checkpoint"


@hydra.main(version_base="1.3", config_path="../configs", config_name="default.yaml")
def score(cfg, eval_model, step, gen_dataset, test_dataset):
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg,
    )
    with wandb.init(
        project="score-style-transfer",
        name=f"sft-ratio-{cfg.sft_ratio}_gen-ratio-{cfg.gen_ratio}"
        f"{'' if cfg.dpo_gen == 0 else f'_dpo{cfg.dpo_gen}'}",
        config=wandb.config,
    ) as run:
        logging.info("Loading the Semantic Model 🐈‍")
        if cfg.sem_model.checkpoint:
            model_artifact = run.use_artifact(cfg.sem_model.checkpoint)
            model_dir = model_artifact.download()
            eval_model = eval_model.load(model_dir)

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
        if cfg.sem_model.is_trainable:
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

            train_loss = hydra.utils.instantiate(cfg.sem_model.loss, eval_model)()
            eval_model.fit(
                train_objectives=[(train_gen_dataloader, train_loss)],
                epochs=cfg.sem_model.epochs,
                warmup_steps=cfg.sem_model.warmup_steps,
            )

        if cfg.sem_model.is_logged:
            eval_model.save(cfg.sem_model.path)
            run.log_artifact(cfg.sem_model.path, type="model")
        logging.info("Semantic Model Trained 🎉")
        eval_cols = [f"eval_sem_scores_{i}" for i in range(cfg.num_generated_sequences)]

        # gen log
        gen_scores = sem_score(cfg, gen_dataset, eval_model)
        gen_dataset.update(gen_scores)
        wandb.log({"gen_score_dataset": wandb.Table(dataframe=pd.DataFrame(data=gen_dataset))})

        # gold test log
        test_scores = sem_score(cfg, test_dataset, eval_model)
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

        # test log
        test_scores = sem_score(cfg, test_dataset, eval_model)
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

        return eval_model, test_dataset


if __name__ == "__main__":
    score()
