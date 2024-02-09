import os

import hydra
import torch
from accelerate import Accelerator
from omegaconf import omegaconf
from style_transfer.utils import PROMPT, build_dataset
from transformers import AutoModelForCausalLM, set_seed
from transformers.integrations import WandbCallback
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

    dataset = dataset.map(
        add_prompt,
        batched=False,
    )
    train_dataset, gen_dataset, test_dataset = split_dataset(dataset, cfg.sft_ratio, cfg.gen_ratio)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=cfg.model,
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
        quantization_config=hydra.utils.instantiate(cfg.bnb_config),
    )

    args = hydra.utils.instantiate(cfg.training_args)

    peft_config = hydra.utils.instantiate(cfg.lora)
    peft_config.target_modules = list(peft_config.target_modules)

    class CustomWandbCallback(WandbCallback):
        def setup(self, args, state, model, **kwargs):
            super().setup(args, state, model, **kwargs)
            if state.is_world_process_zero:
                dict_conf = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                )
                for k_param, v_params in dict_conf.items():
                    setattr(args, k_param, v_params)
                self._wandb.config.update(args, allow_val_change=True)
                self._wandb.config["sft_dataset_size"] = len(train_dataset)
                self._wandb.config["gen_dataset_size"] = len(gen_dataset)
                self._wandb.config["test_dataset_size"] = len(test_dataset)

    args.run_name = f"sft-ratio-{cfg.sft_ratio}_gen-ratio-{cfg.gen_ratio}"
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        peft_config=peft_config,
        packing=True,
        callbacks=[CustomWandbCallback],
    )
    trainer.train()


if __name__ == "__main__":
    main()
