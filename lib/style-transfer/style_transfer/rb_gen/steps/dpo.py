import hydra
import numpy as np
import pandas as pd
import wandb
from peft import AutoPeftModelForCausalLM
from trl import DPOTrainer


def add_preferences(data_point):
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


def dpo_train(cfg, model_path, tokenizer, dataset):
    dataset = dataset.map(
        add_preferences,
        batched=False,
    )

    percentile = np.percentile(dataset["chosen_score"], cfg.dpo.percentile)
    dataset = dataset.filter(lambda x: x["chosen_score"] > percentile)
    dataset = dataset.select_columns(["prompts", "chosen", "rejected"])
    dataset = dataset.rename_column("prompts", "prompt")

    args = hydra.utils.instantiate(cfg.dpo.training_args)
    args.padding_value = tokenizer.eos_token_id
    model = AutoPeftModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path)
    model.enable_input_require_grads()
    wandb.init(project="style-transfer_dpo")
    dpo_trainer = DPOTrainer(
        args=args,
        ref_model=None,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )
    dpo_trainer.train()
    dpo_path = "models/dpo/"
    model.save_pretrained(dpo_path)
    wandb.finish()
    return dpo_path
