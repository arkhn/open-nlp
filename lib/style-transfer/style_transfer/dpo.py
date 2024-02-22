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

import datasets
import hydra
import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from omegaconf import omegaconf
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import WandbCallback
from trl import DPOTrainer, set_seed

tqdm.pandas()
os.environ["WANDB_PROJECT"] = "dpo-style-transfer"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_START_METHOD"] = "thread"


@hydra.main(version_base="1.3", config_path="../configs", config_name="dpo.yaml")
def dpo(cfg):
    api = wandb.Api()
    dataset = api.artifact(cfg.dataset)
    dataset = dataset.files()[0].download(replace=True)
    model_artifact = api.artifact(cfg.checkpoint)
    model_dir = model_artifact.download()
    Accelerator().wait_for_everyone()
    set_seed(cfg.seed)
    lora_config = hydra.utils.instantiate(cfg.lora)
    bnb_config = hydra.utils.instantiate(cfg.bnb_config)
    lora_config.target_modules = list(lora_config.target_modules)
    json_file = json.load(dataset)
    df = pd.DataFrame(data=json_file["data"], columns=json_file["columns"])
    dataset = datasets.Dataset.from_pandas(df)
    model = AutoPeftModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = model.merge_and_unload()
    model.save_pretrained("models/merged/")
    Accelerator().wait_for_everyone()
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="models/merged/",
        torch_dtype=torch.bfloat16,
        device_map={"": Accelerator().local_process_index},
        quantization_config=bnb_config,
    )
    model.config.use_cache = True
    model.enable_input_require_grads()
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    ]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    dataset = dataset.map(
        add_preferences,
        batched=False,
    )

    dataset = dataset.select_columns(["prompts", "chosen", "rejected"])
    dataset = dataset.rename_column("prompts", "prompt")

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

    dpo_trainer = DPOTrainer(
        args=hydra.utils.instantiate(cfg.training_args),
        ref_model=None,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        padding_value=tokenizer.eos_token_id,
        beta=cfg.beta,
        peft_config=lora_config,
        callbacks=[CustomWandbCallback],
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
    )

    dpo_trainer.train()


if __name__ == "__main__":
    dpo()
