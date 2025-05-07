import gc

import hydra
import pandas as pd
import peft
import torch
import wandb
from datasets import Dataset
from omegaconf import ListConfig, OmegaConf
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOTrainer


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg):
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
    kto_config = hydra.utils.instantiate(cfg.kto_config)
    peft_config = hydra.utils.instantiate(cfg.peft_config)
    kto_config.group_by_length = False

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
    print("Merge Adapters: {cfg.adapters_paths}")
    merge_adapters(model, cfg.adapters_paths)
    model = peft.get_peft_model(
        model,
        peft_config,
    )

    model.add_adapter(peft_config=peft_config, adapter_name="reference")
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    kto_config.padding_value = tokenizer.eos_token_id

    dataset = Dataset.from_parquet(cfg.dataset)

    dataset = dataset.to_pandas()
    best_examples = dataset.head(cfg.dataset_size / 2).copy()
    worst_examples = dataset.tail(cfg.dataset_size / 2).copy()
    dataset = pd.concat([best_examples, worst_examples])
    dataset = Dataset.from_pandas(dataset)

    # Train model
    kto_trainer = CustomKTOTrainer(
        args=kto_config,
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    kto_trainer.train()

    # Save and cleanup
    kto_path = f"lora/kto-{cfg.iteration}/{wandb.run.id}"
    kto_trainer.save_model(kto_path)


class CustomKTOTrainer(KTOTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """Execute a training step with memory management.

        Args:
            model: The model to train
            inputs: Training inputs
            num_items_in_batch: Number of items in batch

        Returns:
            Training loss
        """
        loss = super().training_step(model, inputs, num_items_in_batch)
        torch.cuda.empty_cache()
        gc.collect()
        return loss


def merge_adapters(model, adapter_paths):
    for adapter_path in adapter_paths:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    return model


if __name__ == "__main__":
    main()
