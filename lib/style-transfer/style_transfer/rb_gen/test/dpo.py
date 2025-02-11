import hydra
import numpy as np
import pandas as pd
import peft
from datasets import Dataset
from omegaconf import ListConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer


def add_preferences(data_point: dict) -> dict:
    """Add preferences to the data point.
    The preferences are the bes and worst generations and their scores.
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


@hydra.main(version_base="1.3", config_path="./", config_name="dpo.yaml")
def dpo_train(cfg) -> str:
    """Train the model using the reinforcement learning algorithm DPO.
    We fix the percentile of the best candidate to keep for training.

    Args:
        cfg: The configuration for the training.
        step: The current step.
        model_path: The path to the model.
        tokenizer: The tokenizer.
        dataset: The dataset to train on.
    """
    dataset = Dataset.from_pandas(pd.read_csv("./score.csv"))
    dataset = dataset.map(
        add_preferences,
        batched=False,
    )

    percentile = np.percentile(dataset["chosen_score"], cfg.percentile)
    dataset = dataset.filter(lambda x: x["chosen_score"] > percentile)
    dataset = dataset.select_columns(["prompts", "chosen", "rejected"])
    dataset = dataset.rename_column("prompts", "prompt")

    args = hydra.utils.instantiate(cfg.training_args)
    peft_config = hydra.utils.instantiate(cfg.model.peft_config)
    peft_config.target_modules = (
        list(peft_config.target_modules)
        if isinstance(peft_config.target_modules, ListConfig)
        else peft_config.target_modules
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
    )
    model = peft.get_peft_model(
        model,
        peft_config,
    )

    model.add_adapter(peft_config=peft_config, adapter_name="reference")
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.enable_input_require_grads()
    dpo_trainer = DPOTrainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        model_adapter_name="default",
        ref_adapter_name="reference",
    )
    dpo_trainer.train()

    dpo_path = args.output_dir
    dpo_trainer.save_model(dpo_path)
    del model
    return dpo_path


if __name__ == "__main__":
    dpo_train()
