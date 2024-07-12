# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time

import datasets
import hydra
import numpy as np
import pandas as pd
from omegaconf import omegaconf
from tqdm import tqdm
from transformers.integrations import WandbCallback
from trl import DPOTrainer

tqdm.pandas()
os.environ["WANDB_PROJECT"] = "style-transfer-dpo"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def add_preferences(data_point):
    df_point = pd.DataFrame({k: [v] for k, v in dict(data_point).items()})
    filtered_columns = pd.DataFrame(df_point).filter(regex="^eval_sem_scores")
    max_labels = filtered_columns.max().idxmax()[-1]
    best_generation = df_point[f"generation_{max_labels}"].values[0]
    best_score = filtered_columns.max().max()
    min_labels = filtered_columns.min().idxmin()[-1]
    worst_generation = df_point[f"generation_{min_labels}"].values[0]
    worst_score = filtered_columns.min().min()
    data_point["chosen"] = best_generation
    data_point["rejected"] = worst_generation
    data_point["chosen_score"] = best_score
    data_point["rejected_score"] = worst_score
    data_point["deviation_score"] = best_score - worst_score
    return data_point


def dpo_train(cfg, model, tokenizer, dataset):
    dataset = dataset.map(
        add_preferences,
        batched=False,
    )

    percentile = np.percentile(dataset["chosen_score"], cfg.percentile)
    dataset = dataset.filter(lambda x: x["chosen_score"] > percentile)
    dataset = dataset.select_columns(["prompts", "chosen", "rejected"])
    dataset = dataset.rename_column("prompts", "prompt")

    args = hydra.utils.instantiate(cfg.training_args)
    dpo_trainer = DPOTrainer(
        args=args,
        ref_model=None,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        padding_value=tokenizer.eos_token_id,
        beta=cfg.beta,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
    )

    dpo_trainer.train()
    path = "models/dpo/"
    model.save_pretrained(path)
    return path
