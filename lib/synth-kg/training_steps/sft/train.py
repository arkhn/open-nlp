import logging

import hydra
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig, ListConfig, OmegaConf
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, get_peft_config

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig):
    """Train the model with supervised fine-tuning.

    This function loads a pre-trained model, applies supervised fine-tuning using the provided
    dataset, and evaluates the model on a test dataset.

    Args:
        cfg (DictConfig): The configuration for the training, containing hyperparameters
        and settings.

    Note:
        This function uses the SFTTrainer from the TRL library for supervised fine-tuning.
        It also integrates with Weights & Biases (wandb) for experiment tracking.
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
        group=cfg.group_id,
    )
    model_config = hydra.utils.instantiate(cfg.model_config)
    sft_config = hydra.utils.instantiate(cfg.sft_config)
    cfg.sft_config.output_dir = f"lora/sft/{wandb.run.id}"
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        use_cache=False if sft_config.gradient_checkpointing else True,
    )
    model_config.lora_target_modules = (
        list(model_config.lora_target_modules)
        if isinstance(model_config.lora_target_modules, ListConfig)
        else model_config.lora_target_modules
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    dataset = Dataset.from_parquet(cfg.dataset)
    dataset = dataset.map(lambda x: {"text": x["instruction"] + x["response"]})
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(cfg.sft_config.output_dir)


if __name__ == "__main__":
    main()
