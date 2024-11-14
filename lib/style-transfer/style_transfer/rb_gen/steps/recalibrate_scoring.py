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

    train_ground_dict = process_ground_predictions(train_gen_dataset, eval_model)
    train_examples.extend(
        create_input_examples(
            train_ground_dict["text_1"],
            train_ground_dict["text_2"],
            train_ground_dict["ground_scores"],
        )
    )

    train_score_dict = process_generation_scores(train_gen_dataset, eval_model)
    for seq in range(4):
        train_examples.extend(
            create_input_examples(
                train_gen_dataset[f"generation_{seq}"],
                train_gen_dataset["ground_texts"],
                train_score_dict[f"evaluator_scores_{seq}"],
                offset=-0.5,
            )
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


def create_input_examples(texts1, texts2, scores, offset=0.5):
    return [
        InputExample(texts=[t1, t2], label=score + offset)
        for t1, t2, score in zip(texts1, texts2, scores)
    ]


def process_ground_predictions(dataset, model):
    split_point = len(dataset["ground_texts"]) // 2
    split1 = dataset["ground_texts"][:split_point]
    split2 = dataset["ground_texts"][split_point:]
    scores = encode_and_score_texts(model, split1, split2)
    return {"ground_scores": scores, "text_1": split1, "text_2": split2}


def encode_and_score_texts(model, texts1, texts2, batch_size=8):
    enc1 = model.encode(texts1, batch_size=batch_size)
    enc2 = model.encode(texts2, batch_size=batch_size)
    return [util.cos_sim(e1, e2)[0][0].item() for e1, e2 in zip(enc1, enc2)]


def process_generation_scores(dataset, model):
    ground_enc = model.encode(dataset["ground_texts"], batch_size=8)
    score_dict = {}
    for seq in range(4):
        prediction_enc = model.encode(dataset[f"generation_{seq}"], batch_size=8)
        scores = [
            util.cos_sim(g_enc, p_enc)[0][0].item()
            for g_enc, p_enc in zip(ground_enc, prediction_enc)
        ]
        score_dict[f"evaluator_scores_{seq}"] = scores
    return score_dict


def score_gen_dataset(cfg: DictConfig, dataset: dict, eval_model: SentenceTransformer) -> dict:
    """Score the dataset. Using the evaluator model and cosine similarity.
    We score the dataset by calculating the cosine similarity between the ground truth a
    nd the generated text.
    We iterate over the number of generated sequences and calculate the cosine similarity
    for each sequence.

    Args:
        cfg: The configuration for the scoring.
        dataset: The dataset to score.
        eval_model: The model to use for scoring.

    Returns:
        The scored dataset.
    """

    logging.info("🔍 Scoring the dataset ...")
    score_dict: dict = {}
    ground_encoding = eval_model.encode(
        dataset["ground_texts"],
        batch_size=cfg.score.batch_size,
    )
    for seq in range(cfg.model.num_generated_sequences):
        prediction_enc = eval_model.encode(
            dataset[f"generation_{seq}"],
            batch_size=cfg.score.batch_size,
        )
        scores = [
            util.cos_sim(ground_enc, pred_enc)[0][0].item()
            for ground_enc, pred_enc in zip(ground_encoding, prediction_enc)
        ]
        score_dict.setdefault(f"evaluator_scores_{seq}", []).extend(scores)
    logging.info("🎉 Dataset Scored !")
    return score_dict


def recalibrate_scoring(
    cfg, step: int, is_trainable: bool, dataset: Dataset, checkpoint: str
) -> Dataset:
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

    gen_dataset = (
        train_eval_model(cfg, eval_model, dataset) if is_trainable else dataset
    ).to_dict()
    eval_model.save(checkpoint)
    gen_dict_scores = score_gen_dataset(cfg, gen_dataset, eval_model)
    gen_dataset.update(gen_dict_scores)
    gen_df = pd.DataFrame.from_dict(gen_dataset)
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
