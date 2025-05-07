import hydra
import peft
import torch
import wandb
from datasets import Dataset
from omegaconf import ListConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg):
    """Train the model using the reinforcement learning algorithm DPO.
    We fix the percentile of the best candidate to keep for training.

    Args:
        cfg: The configuration for the training.
    """

    wandb_config = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )
    wandb.init(
        project="synth-kg",
        tags=cfg.tags,
        config=wandb_config,
        job_type="training",
        group=f"{cfg.group_id}",
    )
    model_config = hydra.utils.instantiate(cfg.model_config)
    dpo_config = hydra.utils.instantiate(cfg.dpo_config)
    peft_config = hydra.utils.instantiate(cfg.peft_config)
    dpo_config.group_by_length = False

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    peft_config.target_modules = (
        list(peft_config.target_modules)
        if isinstance(peft_config.target_modules, ListConfig)
        else peft_config.target_modules
    )
    model_kwargs = dict(
        torch_dtype=torch_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    merge_adapters(model, cfg.adapters_paths)
    model = peft.get_peft_model(
        model,
        peft_config,
    )

    model.add_adapter(peft_config=peft_config, adapter_name="reference")
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    dpo_config.padding_value = tokenizer.eos_token_id

    dataset = Dataset.from_parquet(cfg.dataset).to_pandas().head(cfg.dataset_size)
    dataset["prompt"] = dataset["instruction"]
    dataset = Dataset.from_pandas(dataset)
    dataset.select_columns(["prompt", "chosen", "rejected"])

    dpo_trainer = DPOTrainer(
        args=dpo_config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
    )
    dpo_trainer.train()
    dpo_path = f"lora/dpo-{cfg.iteration}/{wandb.run.id}"
    dpo_trainer.save_model(dpo_path)


def merge_adapters(model, adapter_paths):
    for adapter_path in adapter_paths:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    return model


if __name__ == "__main__":
    main()
