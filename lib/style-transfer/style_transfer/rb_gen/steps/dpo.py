import glob

import hydra
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from style_transfer.rb_gen.utils.utils import CustomWandbCallback
from transformers import PreTrainedTokenizerBase
from trl import DPOTrainer


def add_preferences(data_point: dict) -> dict:
    """Add preferences to the data point.
    The preferences are the best and worst generations and their scores.
    Previously added during the evaluation step.
    We also add the deviation score which is the difference between the best and worst scores.
    Args:
        data_point: The data point to add preferences to.
    Returns:
        The data point with preferences added.
    """
    df_point = pd.DataFrame({k: [v] for k, v in dict(data_point).items()})
    filtered_columns = pd.DataFrame(df_point).filter(regex="^evaluator_scores")
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


def dpo_train(
    cfg, step, model_path: str, tokenizer: PreTrainedTokenizerBase, dataset: Dataset
) -> str:
    """Train the model using the reinforcement learning algorithm DPO.
    We fix the percentile of the best candidate to keep for training.

    Args:
        cfg: The configuration for the training.
        step: The current step.
        model_path: The path to the model.
        tokenizer: The tokenizer.
        dataset: The dataset to train on.
    """
    wandb.config.update({"state": f"dpo/{step}"}, allow_val_change=True)
    dataset = dataset.map(
        add_preferences,
        batched=False,
    )

    percentile = np.percentile(dataset["chosen_score"], cfg.dpo.percentile)
    dataset = dataset.filter(lambda x: x["chosen_score"] > percentile)
    dataset = dataset.select_columns(["prompts", "chosen", "rejected"])
    dataset = dataset.rename_column("prompts", "prompt")

    cfg.dpo.training_args.output_dir = f"models/dpo/{step}"
    args = hydra.utils.instantiate(cfg.dpo.training_args)
    args.padding_value = tokenizer.eos_token_id
    model = AutoPeftModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path)
    model.enable_input_require_grads()
    dpo_trainer = DPOTrainer(
        args=args,
        ref_model=None,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        callbacks=[CustomWandbCallback],
    )
    if glob.glob(f"{args.output_dir}/*"):
        dpo_trainer.train(resume_from_checkpoint=True)
    else:
        dpo_trainer.train()

    dpo_path = args.output_dir
    model.save_pretrained(dpo_path)
    return dpo_path
