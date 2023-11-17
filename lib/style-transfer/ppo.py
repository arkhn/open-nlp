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
from dataclasses import dataclass, field
from typing import Optional

import datasets
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from torch.optim import Adam
from trl import (
    AutoModelForCausalLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    create_reference_model,
    set_seed,
)
from trl.core import LengthSampler

tqdm.pandas()

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


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `project_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(
        default="amazon/MistralLite", metadata={"help": "the model name"}
    )
    log_with: Optional[str] = field(
        default="wandb", metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./amazon/MistralLite-clinical-notes",
        metadata={"help": "the path to save the model"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=100,
    tracker_project_name="style-transfer-ppo",
    init_kl_coef=2,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    use_score_scaling=True,
    use_score_norm=True,
    score_clip=0.5,
    remove_unused_columns=False,
)

max_gen_sampler = LengthSampler(80, 256)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    config,
    dataset_name="bio-datasets/mimic_style_transfer",
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
        <|prompter|>construct an original 'History of Present Illness' (HPI) section for a discharge summary. 
        Your response should capture the essence of a patient's health journey and recent medical experiences, 
        while strictly using all the provided keywords conserving the order. 
        You must adopt a medical telegraphic style, abbreviated, characterized by concise and direct language.
        Current Keywords: {}</s><|assistant|>
        """
        continuation = sample["text"]
        sample["ground_ids"] = tokenizer.encode(continuation)[max_gen_sampler() :]
        ground_text = tokenizer.decode(sample["ground_ids"], skip_special_tokens=True)
        keywords = ",".join([keyword for keyword in sample["keywords"].split(",")
                             if keyword in ground_text])
        prompt = str.format(
            instruction,
            keywords,
        )
        sample["input_ids"] = tokenizer.encode(prompt)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        sample["keywords"] = keywords
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
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model,
    peft_config=lora_config,
    use_flash_attention_2=True,
    bnb_config=bnb_config,
)
model.gradient_checkpointing_enable()
# We create a reference model by sharing 20 layers
ref_model = create_reference_model(model, num_shared_layers=20)

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


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 80
output_max_length = 120
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model_save_path = script_args.model_save_path
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    queries_ids = batch["input_ids"]
    # Get response from the policy model
    response_tensors = []
    rewards = []
    # TODO batch generate here
    for query, ground_id in zip(queries_ids, batch["ground_ids"]):
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len

        with torch.no_grad():
            response = ppo_trainer.generate(
                query,
                **generation_kwargs,
            )
            response = response.squeeze()[-gen_len:]
            response_tensors.append(response)
            # Compute scores
            eval_gen_len = LengthSampler(6, 10)()

            # create prompts
            style_accuracy = str.format(
                """Here is sentence S1: {} and sentence S2: {}. 
                How different is sentence S2 compared to S1 on a continuous scale from 0
                (completely different styles) to 100 (completely identical styles)? Result =""",
                tokenizer.decode(ground_id),
                tokenizer.decode(response),
            )

            content_preservation = str.format(
                """Here is S1: {} and sentence S2: {}.
                How much does S2 preserve the content of S1 on a continuous scale from 0
                (completely different topic) to 100 (identical topic)? Result =""",
                tokenizer.decode(ground_id),
                tokenizer.decode(response),
            )

            fluency = str.format(
                """<|prompter|>How fluent is this sentence S1: "{}" on a continuous scale from 1 to 100
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
                "rewards": rewards,
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
            ppo_trainer.save_pretrained(model_save_path)
