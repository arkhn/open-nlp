import os

import hydra
import torch
from accelerate import Accelerator
from style_transfer.utils import PROMPT, build_dataset
from transformers import AutoModelForCausalLM, set_seed
from trl import SFTTrainer
from utils import split_dataset

os.environ["WANDB_PROJECT"] = "sft-style-transfer"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


@hydra.main(version_base="1.3", config_path="../configs", config_name="sft.yaml")
def main(cfg):
    set_seed(cfg.seed)
    dataset = build_dataset(
        dataset_name=cfg.dataset,
        model_name=cfg.model,
        max_sampler_length=cfg.max_seq_length,
    )

    def add_prompt(data_point):
        data_point["text"] = (
            str.format(
                PROMPT,
                data_point["keywords"],
            )
            + "\n"
            + data_point["text"]
        )
        return data_point

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
        quantization_config=hydra.utils.instantiate(cfg.bnb_config),
    )
    dataset = dataset.map(
        add_prompt,
        batched=False,
    )
    train_dataset, gen_, test_dataset = split_dataset(dataset, cfg.sft_ratio, cfg.gen_ratio)

    args = hydra.utils.instantiate(cfg.training_args)

    peft_config = hydra.utils.instantiate(cfg.lora)
    peft_config.target_modules = list(peft_config.target_modules)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        peft_config=peft_config,
        packing=True,
    )
    trainer.train()


if __name__ == "__main__":
    main()
