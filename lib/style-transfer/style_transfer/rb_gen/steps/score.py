import logging
import os

import hydra
import pandas as pd
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig
from sentence_transformers import InputExample, SentenceTransformer, util
from style_transfer.rb_gen.utils.utils import CustomWandbCallback

os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def train_eval_model(
    cfg: DictConfig, eval_model: SentenceTransformer, gen_dataset: Dataset
) -> Dataset:
    """Train the evaluator model.

    Args:
        cfg: The configuration for the training.
        eval_model: The model to train.
        gen_dataset: The dataset to use for training.
    """
    logging.info("🎲 Training Semantic Model ...")
    logging.info("🪚 Splitting Dataset for training...")
    gen_dataset = gen_dataset.train_test_split(train_size=cfg.score.train.train_size)
    train_gen_dataset = gen_dataset["train"]
    gen_dataset = gen_dataset["test"]
    train_examples = []
    for num_seq in range(cfg.model.num_generated_sequences):
        train_examples.extend(
            [
                InputExample(texts=[pred, ground_text], label=0)
                for pred, ground_text in zip(
                    train_gen_dataset[f"generation_{num_seq}"], train_gen_dataset["ground_texts"]
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
        callback=[CustomWandbCallback],
    )
    logging.info("🎉 Semantic Model Trained !")
    return gen_dataset


def dataset_scoring(cfg: DictConfig, dataset: dict, evaluator: SentenceTransformer) -> dict:
    """Score the dataset. Using the evaluator model and cosine similarity.
    We score the dataset by calculating the cosine similarity between the ground truth a
    nd the generated text.
    We iterate over the number of generated sequences and calculate the cosine similarity
    for each sequence.

    Args:
        cfg: The configuration for the scoring.
        dataset: The dataset to score.
        evaluator: The model to use for scoring.

    Returns:
        The scored dataset.
    """

    logging.info("🔍 Scoring the dataset ...")
    score_dict: dict = {}
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


def score(cfg, step: int, is_trainable: bool, dataset: Dataset, checkpoint: str) -> Dataset:
    """Score the dataset and log the results.

    Args:
        cfg: The configuration for the scoring.
        step: The current step.
        is_trainable: Whether the model is trainable.
        dataset: The dataset to score.
        checkpoint: The checkpoint path to save the model.

    Returns:
        The scored dataset.
    """
    wandb.config.update({"state": f"score/{step}"}, allow_val_change=True)
    logging.info("🐈 Loading the Semantic Model ...")
    if step == 0:
        eval_model = hydra.utils.instantiate(cfg.score.model)
    else:
        eval_model = hydra.utils.instantiate(
            cfg.score.model,
            model_name_or_path=checkpoint,
        )

    gen_dict = (train_eval_model(cfg, eval_model, dataset) if is_trainable else dataset).to_dict()
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

    wandb.log({f"{wandb.config['state']}/dataset/score": wandb.Table(dataframe=gen_df)})
    wandb.log({f"{wandb.config['state']}/max/mean": gen_df["max_score"].mean()})
    wandb.log({f"{wandb.config['state']}/min/mean": gen_df["min_score"].mean()})
    wandb.log({f"{wandb.config['state']}/mean": gen_df["mean_score"].mean()})

    del eval_model
    return Dataset.from_pandas(gen_df)
