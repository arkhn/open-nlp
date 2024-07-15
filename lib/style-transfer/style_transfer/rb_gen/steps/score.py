import logging
import os

import hydra
import pandas as pd
import torch
import wandb
from datasets import Dataset
from omegaconf import omegaconf
from sentence_transformers import InputExample, util

os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def train_eval_model(cfg, eval_model, gen_dataset):
    logging.info("🎲 Training Semantic Model ...")
    logging.info("🪚 Splitting Dataset for training...")
    gen_dataset = gen_dataset.train_test_split(train_size=cfg.score.train.train_size)
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
    if cfg.score.train.use_ground_truth:
        train_examples.extend(
            [
                InputExample(texts=[ground_text, ground_text], label=1)
                for ground_text in train_gen_dataset["ground_texts"]
            ]
        )
    train_gen_dataloader = torch.utils.data.DataLoader(
        train_examples,
        batch_size=cfg.score.batch_size,
    )

    train_loss = hydra.utils.instantiate(cfg.score.train.loss, eval_model)()
    eval_model.fit(
        train_objectives=[(train_gen_dataloader, train_loss)],
        epochs=cfg.score.train.epochs,
        warmup_steps=cfg.score.train.warmup_steps,
    )
    logging.info("🎉 Semantic Model Trained !")
    return gen_dataset


def dataset_scoring(cfg, dataset, evaluator):
    logging.info("🔍 Scoring the dataset ...")
    score_dict = {}
    ground_enc = evaluator.encode(
        dataset["ground_texts"],
        batch_size=cfg.score.batch_size,
    )
    for seq in range(cfg.model.num_generated_sequences):
        prediction_enc = evaluator.encode(
            dataset[f"generation_{seq}"],
            batch_size=cfg.score.batch_size,
        )
        scores = [
            util.cos_sim(ground_enc, pred_enc)[0][0].item()
            for ground_enc, pred_enc in zip(ground_enc, prediction_enc)
        ]
        score_dict.setdefault(f"evaluator_scores_{seq}", []).extend(scores)
    logging.info("🎉 Dataset Scored !")
    return score_dict


def score(cfg, step, is_trainable, dataset, checkpoint):
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg,
    )
    wandb.config.update({"step": step})
    with wandb.init(
        project="style-transfer_score",
        config=wandb.config,
    ):
        logging.info("🐈 Loading the Semantic Model ...")
        if step == 0:
            eval_model = hydra.utils.instantiate(cfg.score.model)
        else:
            eval_model = hydra.utils.instantiate(
                cfg.score.model,
                model_name_or_path=checkpoint,
            )

        gen_dict = train_eval_model(cfg, eval_model, dataset) if is_trainable else dataset.to_dict()
        eval_model.save(checkpoint)
        gen_dict_scores = dataset_scoring(cfg, gen_dict, eval_model)
        gen_dict.update(gen_dict_scores)
        gen_df = pd.DataFrame.from_dict(gen_dict)
        generated_sequences = [
            f"evaluator_scores_{seq}" for seq in range(cfg.model.num_generated_sequences)
        ]

        gen_df["max_score"] = gen_df[generated_sequences].max(axis=1)
        gen_df["min_score"] = gen_df[generated_sequences].min(axis=1)
        gen_df["mean_score"] = gen_df[generated_sequences].mean(axis=1)

        wandb.log({"gen_score_dataset": wandb.Table(dataframe=gen_df)})
        wandb.log({"gen/max/mean": gen_df["max_score"].mean()})
        wandb.log({"gen/min/mean": gen_df["min_score"].mean()})
        wandb.log({"gen/mean": gen_df["mean_score"].mean()})

    wandb.finish()
    del eval_model
    return Dataset.from_pandas(gen_df)
