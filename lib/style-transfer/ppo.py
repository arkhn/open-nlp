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
import re

import datasets
import hydra
import pandas as pd
import rootutils
import torch
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler

tqdm.pandas()
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPTJ model to generate less toxic contents
# by using allenai/real-toxicity-prompts dataset. We use PPO
#  (proximal policy optimization) to optimize the model.
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


@hydra.main(version_base="1.3", config_path="./", config_name="ppo.yaml")
def main(cfg):
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    config = cfg.hparams.ppo_config

    max_gen_sampler = LengthSampler(
        cfg.hparams.max_gen_sampler.min,
        cfg.hparams.max_gen_sampler.max,
    )

    # Below is an example function to build the dataset. In our case, we use the IMDB dataset
    # from the `datasets` library. One should customize this function to train the model on
    # its own dataset.
    def build_dataset(
        config,
        dataset_name=cfg.dataset,
    ):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one should
        customize this function to train the model on its own dataset.

        Args:
            dataset_name (`str`):
                The name of the dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        ds = load_dataset(dataset_name, split="train")

        def tokenize(sample):
            instruction = """
            <|prompter|>construct an original 'History of Present Illness' (HPI) section for
            a discharge summary.
            Your response should capture the essence of a patient's health journey
            and recent medical experiences,
            while strictly using all the provided keywords conserving the order.
            You must adopt a medical telegraphic style, abbreviated, characterized
            by concise and direct language.
            Current Keywords: {}</s><|assistant|>
            """
            continuation = sample["text"]
            sampler = max_gen_sampler()
            sample["ground_ids"] = tokenizer.encode(continuation)[sampler:]
            ground_text = tokenizer.decode(sample["ground_ids"], skip_special_tokens=True)
            keywords = ",".join(
                [keyword for keyword in sample["keywords"].split(",") if keyword in ground_text]
            )
            prompt = str.format(
                instruction,
                keywords,
            )
            sample["input_ids"] = tokenizer.encode(prompt)
            sample["query"] = tokenizer.decode(sample["input_ids"])
            sample["keywords"] = keywords
            sample["max_gen_len"] = sampler
            return sample

        ds_dict = {"keywords": [], "text": []}
        for keywords, text in zip(ds["keywords"], ds["text"]):
            for kw, t in zip(keywords, text):
                ds_dict["keywords"].append(kw)
                ds_dict["text"].append(t)
        ds = datasets.Dataset.from_dict(ds_dict)
        ds = ds.map(tokenize, batched=False)
        ds = ds.filter(lambda x: len(x["keywords"]) > 5)
        ds.set_format(type="torch")

        ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

        return ds

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(
        config,
    )

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # set seed before initializing value head for deterministic eval
    set_seed(config.seed)
    # Now let's build the model, the reference model, and the tokenizer. We first load the model
    # in bfloat16 to save memory using `transformers`.
    # And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
    lora_config = cfg.hparams.lora

    bnb_config = cfg.bnb_config

    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        peft_config=lora_config,
        use_flash_attention_2=cfg.hparams.model.use_flash_attention_2,
        bnb_config=bnb_config,
    )
    model.gradient_checkpointing_enable()
    # We create a reference model by sharing 20 layers
    ref_model = create_reference_model(model, num_shared_layers=cfg.hparams.num_shared_layers)

    # We make sure to use `Adam` optimizer on the model parameters that require gradients.
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # We then build the reward pipeline, we will use the toxicity model to compute the reward.
    # We first load the toxicity model and tokenizer.
    # We load the toxicity model in fp16 to save memory.
    evaluator_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.bfloat16
    )
    evaluator_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        evaluator_model, use_flash_attention_2=True, bnb_config=bnb_config
    ).to(ppo_trainer.accelerator.device)
    for param in evaluator_model.parameters():
        param.requires_grad = False

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        queries_ids = batch["input_ids"]
        # Get response from the policy model
        response_tensors = []
        rewards = []
        for query, ground_id, gen_len in zip(
            queries_ids, batch["ground_ids"], batch["max_gen_len"]
        ):
            with torch.no_grad():
                response = ppo_trainer.generate(
                    query,
                    max_new_tokens=gen_len,
                    generation_config=cfg.hparams.generation_config,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = response.squeeze()[-gen_len:]
                response_tensors.append(response)
                # Compute scores
                eval_gen_len = LengthSampler(6, 8)()
                # create prompts
                style_accuracy = str.format(
                    """Here is sentence S1: {} and sentence S2: {}.
                    How different is sentence S2 compared to S1 on a continuous scale from 0
                    (completely different styles) to 100 (completely identical styles)? Result =""",
                    tokenizer.decode(ground_id),
                    tokenizer.decode(response),
                )

                content_preservation = str.format(
                    """Here is sentence S1: {} and sentence S2: {}.
                    How much does S2 preserve the content of S1 on a continuous scale from 0
                    (completely different topic) to 100 (identical topic)? Result =""",
                    tokenizer.decode(ground_id),
                    tokenizer.decode(response),
                )

                fluency = str.format(
                    """<|prompter|>How fluent is this sentence S1: "{}"
                    on a continuous scale from 1 to 100
                    where 0 (lowest fluent) and 100 (highest fluent)? Result =</s><|assistant|>""",
                    tokenizer.decode(response),
                )
                batch_gen_inputs = tokenizer(
                    [
                        style_accuracy,
                        content_preservation,
                        fluency,
                    ],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(ppo_trainer.accelerator.device)

                # generate scores
                scores = evaluator_model.generate(**batch_gen_inputs, max_new_tokens=eval_gen_len)[
                    :, -eval_gen_len:
                ]

                def extract_score(score_ids):
                    pattern = r"(?:[\d]+|[\d]+\.[\d]+)"
                    findall = re.findall(
                        pattern,
                        tokenizer.decode(score_ids.squeeze()),
                    )
                    return eval(findall[0]) / 100 if len(findall) == 1 else 0

                # eval to create tensor
                aggregate_score = sum([extract_score(score_ids) for score_ids in scores])
                reward = torch.FloatTensor([float(aggregate_score) / 3])
                rewards.append(reward)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        batch_table = wandb.Table(
            dataframe=pd.DataFrame(
                {
                    "keywords": batch["keywords"],
                    "query": batch["query"],
                    "response": batch["response"],
                    "ground_truth": [
                        tokenizer.decode(ground_id) for ground_id in batch["ground_ids"]
                    ],
                    "rewards": [reward.squeeze() for reward in rewards],
                }
            )
        )
        ppo_trainer.accelerator.log({"train/batch_sample_predictions": batch_table})

        # Run PPO step
        stats = ppo_trainer.step(queries_ids, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        torch.cuda.empty_cache()
        # Save model every 100 epochs
        if epoch % 100 == 0:
            if ppo_trainer.accelerator.is_main_process:
                ppo_trainer.save_pretrained(cfg.hparams.model.save_path)


if __name__ == "__main__":
    main()
