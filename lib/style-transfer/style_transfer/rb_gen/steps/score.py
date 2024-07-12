import logging
import os

import hydra
import pandas as pd
import torch
import wandb
from omegaconf import omegaconf
from sentence_transformers import InputExample, util

os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def train_eval_model(cfg, eval_model, gen_dataset):
    logging.info("🎲 Training Semantic Model ...")
    logging.info("🪚 Splitting Dataset for training...")
    gen_dataset = gen_dataset.train_test_split(train_size=cfg.score.sem_model.train_size)
    train_gen_dataset = gen_dataset["train"]
    gen_dataset = gen_dataset["test"].to_dict()
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

    train_loss = hydra.utils.instantiate(cfg.sem_model.loss, eval_model)()
    eval_model.fit(
        train_objectives=[(train_gen_dataloader, train_loss)],
        epochs=cfg.sem_model.epochs,
        warmup_steps=cfg.sem_model.warmup_steps,
    )
    logging.info("🎉 Semantic Model Trained !")
    return gen_dataset


def dataset_scoring(cfg, dataset, sem_model):
    logging.info("🔍 Scoring the dataset ...")
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
    logging.info("🎉 Dataset Scored !")
    return score_dict


def score(cfg, step, is_trainable, gen_dataset):
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg,
    )
    wandb.config.update({"step": step})
    with wandb.init(
        project="score-style-transfer",
        config=wandb.config,
    ) as run:
        logging.info("🐈 Loading the Semantic Model ...")
        eval_model = hydra.utils.instantiate(cfg.score.sem_model)
        gen_dataset = (
            train_eval_model(cfg, eval_model, gen_dataset) if is_trainable else gen_dataset
        )
        if cfg.sem_model.is_logged:
            eval_model.save(cfg.sem_model.path)
            run.log_artifact(cfg.sem_model.path, type="model")

        gen_scores = dataset_scoring(cfg, gen_dataset, eval_model)
        gen_dataset.update(gen_scores)
        wandb.log({"gen_score_dataset": wandb.Table(dataframe=pd.DataFrame(data=gen_dataset))})
        wandb.log({"gen/max/mean": gen_dataset["max_score"].mean()})
        wandb.log({"gen/min/mean": gen_dataset["min_score"].mean()})
        wandb.log({"gen/mean": gen_dataset["mean_score"].mean()})

        return gen_dataset
