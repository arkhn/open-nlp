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
import logging
import os
from functools import partial
from typing import Dict, Union

import hydra
import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from torch import nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from trl import DPOTrainer, set_seed
from trl.trainer.utils import DPODataCollatorWithPadding
from utils import EVAL_PROMPT, build_dataset, extract_score

tqdm.pandas()
os.environ["WANDB_PROJECT"] = "dpo-style-transfer"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        args,
        model,
        ref_model,
        tokenizer,
        train_dataset,
        beta,
        evaluator,
        generation_config,
        bnb_config,
        peft_config,
        num_generated_sequences,
        padding_value,
        seed,
    ):
        super().__init__(
            args=args,
            ref_model=ref_model,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            beta=beta,
            peft_config=peft_config,
            padding_value=padding_value,
        )
        # load evaluator model
        evaluator_tokenizer = AutoTokenizer.from_pretrained(evaluator, padding_side="left")
        evaluator_tokenizer.pad_token = evaluator_tokenizer.eos_token

        evaluator_model = AutoModelForCausalLM.from_pretrained(
            evaluator,
            torch_dtype=torch.bfloat16,
            device_map={"": Accelerator().local_process_index},
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            quantization_config=bnb_config,
        )

        for param in evaluator_model.parameters():
            param.requires_grad = False

        self.data_collator: CustomDPOCollator = CustomDPOCollator(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            is_encoder_decoder=self.data_collator.is_encoder_decoder,
            evaluator=evaluator_model,
            evaluator_tokenizer=evaluator_tokenizer,
            generation_config=generation_config,
            tokenize_row_fn=partial(super().tokenize_row, model=self.model),
            dpo_trainer=self,
            num_generated_sequences=num_generated_sequences,
            seed=seed,
        )

    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        return feature


class CustomDPOCollator(DPODataCollatorWithPadding):
    def __init__(
        self,
        tokenizer,
        pad_token_id,
        label_pad_token_id,
        is_encoder_decoder,
        evaluator,
        evaluator_tokenizer,
        generation_config,
        model,
        tokenize_row_fn,
        dpo_trainer: CustomDPOTrainer,
        num_generated_sequences,
        seed,
    ):
        super().__init__(
            pad_token_id,
            label_pad_token_id,
            is_encoder_decoder,
        )
        self.num_generated_sequences = num_generated_sequences
        self.dpo_trainer = dpo_trainer
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.evaluator_tokenizer = evaluator_tokenizer
        self.generation_config = generation_config
        self.model = model
        self.tokenize_row_fn = tokenize_row_fn
        self.seed = seed

    def __call__(self, features):
        logging.info("Generation  step ... 🌱")
        batch = self.auto_evaluate(features)
        batch = [self.tokenize_row_fn(data_point) for data_point in batch]
        logging.info("Generation step ... done 🌈")
        return super().__call__(features=batch)

    def auto_evaluate(self, batch):
        """Generate reports with model and evaluate with evaluator model

        Args:
            batch: list of data points

        Returns:
            list: list of data points with generated reports
        """
        dpo_batch_dict = {}
        dpo_batch_dict["prompt"] = [data_point["query"] for data_point in batch]

        formatted_queries = [data_point["formatted_query"] for data_point in batch]
        input_ids = self.tokenizer(
            formatted_queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.dpo_trainer.accelerator.device)

        reports_candidates = []
        for seed in range(self.num_generated_sequences):
            with torch.no_grad():
                set_seed(seed)
                reports = self.model.generate(
                    **input_ids,
                    max_new_tokens=256,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                reports = [
                    report.split("[/INST] ")[1].removeprefix("HPI:").strip()
                    for report in self.tokenizer.batch_decode(reports, skip_special_tokens=True)
                ]
                reports_candidates.append(reports)
        set_seed(self.seed)
        reports_candidates = list(map(list, zip(*reports_candidates)))
        evaluator_prompts = [
            [
                str.format(
                    EVAL_PROMPT,
                    prompt,
                    report,
                    ground_text,
                )
                for report in reports
            ]
            for reports, prompt, ground_text in zip(
                reports_candidates,
                dpo_batch_dict["prompt"],
                [data_point["ground_texts"] for data_point in batch],
            )
        ]

        evaluator_prompts = [
            e_prompt for evaluator_prompt in evaluator_prompts for e_prompt in evaluator_prompt
        ]
        batch_gen_inputs = self.evaluator_tokenizer(
            evaluator_prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.dpo_trainer.accelerator.device)
        with torch.no_grad():
            evaluator_output = self.evaluator.generate(
                **batch_gen_inputs,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                max_new_tokens=512,
                repetition_penalty=1.03,
                pad_token_id=self.evaluator_tokenizer.eos_token_id,
            )
        feedbacks = [
            feedback.split("###Feedback:")[1].strip()
            for feedback in self.evaluator_tokenizer.batch_decode(
                evaluator_output, skip_special_tokens=True
            )
        ]
        scores = [extract_score(feedback) for feedback in feedbacks]
        scores = [
            scores[i : i + self.num_generated_sequences]
            for i in range(0, len(scores), self.num_generated_sequences)
        ]

        # list of sorted list of reports candidates by score
        sorted_reports_candidates = [
            [(x, s) for s, x in sorted(zip(score, reports))]
            for score, reports in zip(scores, reports_candidates)
        ]

        # for each batch, we take the best and worst report
        for batch_idx, reports_candidates in enumerate(sorted_reports_candidates):
            dpo_batch_dict.setdefault("chosen", []).append(reports_candidates[-1][0])
            dpo_batch_dict.setdefault("rejected", []).append(reports_candidates[0][0])

        dpo_batch_dict = [
            {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            for prompt, chosen, rejected in zip(
                dpo_batch_dict["prompt"], dpo_batch_dict["chosen"], dpo_batch_dict["rejected"]
            )
        ]
        table = wandb.Table(
            dataframe=pd.DataFrame(
                {
                    "prompt": [data_point["prompt"] for data_point in dpo_batch_dict],
                    "chosen": [data_point["chosen"] for data_point in dpo_batch_dict],
                    "rejected": [data_point["rejected"] for data_point in dpo_batch_dict],
                    "ground_texts": [data_point["ground_texts"] for data_point in batch],
                }
            )
        )
        self.dpo_trainer.log({"train/batch_sample_predictions": table})
        return dpo_batch_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="dpo.yaml")
def main(cfg):
    set_seed(cfg.seed)
    lora_config = hydra.utils.instantiate(cfg.lora)
    bnb_config = hydra.utils.instantiate(cfg.bnb)
    lora_config.target_modules = list(lora_config.target_modules)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        device_map={"": Accelerator().local_process_index},
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = True
    model._ddp_params_and_buffers_to_ignore = [
        name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    ]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    generation_config = hydra.utils.instantiate(cfg.generation_config)

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(
        dataset_name=cfg.dataset,
        model_name=cfg.model.name,
        max_sampler_length=cfg.max_sampler_length,
    )

    dpo_trainer = CustomDPOTrainer(
        args=hydra.utils.instantiate(cfg.training_args),
        ref_model=None,
        model=model,
        evaluator="kaist-ai/Prometheus-13b-v1.0",
        generation_config=generation_config,
        tokenizer=tokenizer,
        train_dataset=dataset,
        padding_value=tokenizer.eos_token_id,
        beta=cfg.beta,
        bnb_config=bnb_config,
        peft_config=lora_config,
        num_generated_sequences=cfg.num_generated_sequences,
        seed=cfg.seed,
    )

    dpo_trainer.train()


if __name__ == "__main__":
    main()
