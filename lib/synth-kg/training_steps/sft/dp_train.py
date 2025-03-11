import logging

import dp_transformers
import hydra
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig, ListConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        This function uses the classic Trainer from Hugging Face for supervised fine-tuning.
        It also integrates with Weights & Biases (wandb) for experiment tracking.
    """
    wandb.init(project="synth-kg", tags=cfg.tags)

    model_config = hydra.utils.instantiate(cfg.model_config)
    cfg.training_arguments.output_dir = f"lora/dp-sft/{wandb.run.id}"

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    peft_config = hydra.utils.instantiate(cfg.peft_config)
    peft_config.target_modules = (
        list(peft_config.target_modules)
        if isinstance(peft_config.target_modules, ListConfig)
        else peft_config.target_modules
    )
    model = get_peft_model(model=model, peft_config=peft_config)
    dataset = Dataset.from_parquet(cfg.dataset)
    dataset = dataset.map(lambda x: {"text": x["instruction"] + x["response"]})

    training_args = hydra.utils.instantiate(cfg.training_arguments)

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        ),
        batched=True,
        num_proc=8,
        desc="tokenizing dataset",
    )
    print("dataset len:", len(tokenized_dataset))
    tokenized_dataset = tokenized_dataset.select_columns(["input_ids", "attention_mask"])
    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)
    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=training_args,
        model=model,
        train_dataset=tokenized_dataset,
        privacy_args=hydra.utils.instantiate(cfg.private_arguments),
        data_collator=data_collator,
    )
    trainer.args.gradient_checkpointing = False
    trainer.train()
    model.save_pretrained(cfg.training_arguments.output_dir)


if __name__ == "__main__":
    main()
