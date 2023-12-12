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
import re

import datasets
import hydra
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, create_reference_model, set_seed

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


@hydra.main(version_base="1.3", config_path="./", config_name="ppo.yaml")
def main(cfg):
    ppo_config = hydra.utils.instantiate(cfg.ppo_config)

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
            <s>[INST]As a doctor, you must write an original 'History of Present Illness' (HPI) section for a discharge summary. 
            Your response should capture the essence of a patient's health journey and recent medical experiences, 
            while strictly using all the provided keywords conserving the order. 
            You must adopt a medical telegraphic style, abbreviated, characterized by concise and direct language.
            Keywords: {}[/INST]"""
            continuation = sample["text"]
            inputs = tokenizer.encode(continuation)
            sample["ground_ids"] = (
                inputs
                if len(continuation) > cfg.max_sampler_length
                else inputs[: cfg.max_sampler_length]
            )
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
            sample["max_gen_len"] = len(sample["ground_ids"])
            return sample

        ds_dict = {"keywords": [], "text": []}
        for keywords, text in zip(ds["keywords"], ds["text"]):
            for kw, t in zip(keywords, text):
                ds_dict["keywords"].append(kw)
                ds_dict["text"].append(t)
        ds = datasets.Dataset.from_dict(ds_dict)
        ds = ds.map(tokenize, batched=False)
        ds = ds.filter(lambda x: len(x["keywords"].split(",")) > 1)
        ds.set_format(type="torch")

        ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]
        return ds

    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(
        ppo_config,
    )

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # set seed before initializing value head for deterministic eval
    set_seed(cfg.seed)
    # Now let's build the model, the reference model, and the tokenizer. We first load the model
    # in bfloat16 to save memory using `transformers`.
    # And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
    lora_config = hydra.utils.instantiate(cfg.lora)
    bnb_config = hydra.utils.instantiate(cfg.bnb)
    lora_config.target_modules = list(lora_config.target_modules)
    model = AutoModelForCausalLM.from_pretrained(ppo_config.model_name, torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model,
        peft_config=lora_config,
        use_flash_attention_2=cfg.model.use_flash_attention_2,
        bnb_config=bnb_config,
    )
    model.gradient_checkpointing_enable()
    # We create a reference model by sharing 20 layers
    ref_model = create_reference_model(model, num_shared_layers=cfg.model.num_shared_layers)

    # We make sure to use `Adam` optimizer on the model parameters that require gradients.
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ppo_config.learning_rate,
    )

    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(
        ppo_config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )
    ppo_trainer.accelerator.get_tracker("wandb").config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    if ppo_trainer.accelerator.is_main_process:
        model.push_to_hub(
            f"bio-datasets/{cfg.ppo_config.model_name.split('/')[-1]}-"
            f"{ppo_trainer.accelerator.get_tracker('wandb').run.name}-epoch-0"
        )
    # We then build the reward pipeline, we will use the toxicity model to compute the reward.
    # We first load the toxicity model and tokenizer.
    # We load the toxicity model in fp16 to save memory.
    evaluator_tokenizer = AutoTokenizer.from_pretrained(
        "kaist-ai/Prometheus-13b-v1.0", padding_side="left"
    )
    evaluator_tokenizer.pad_token = evaluator_tokenizer.eos_token
    evaluator_model = AutoModelForCausalLM.from_pretrained(
        "kaist-ai/prometheus-13b-v1.0", torch_dtype=torch.bfloat16
    ).to(ppo_trainer.accelerator.device)
    for param in evaluator_model.parameters():
        param.requires_grad = False

    generation_config = hydra.utils.instantiate(cfg.generation_config)
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        queries_ids = batch["input_ids"]
        # Get response from the policy model
        response_tensors = []
        rewards = []
        style_rewards = []
        content_preservation_rewards = []
        fluency_rewards = []
        style_feedbacks = []
        content_preservation_feedbacks = []
        fluency_feedbacks = []
        for query_ids, ground_id, gen_len in zip(
            queries_ids, batch["ground_ids"], batch["max_gen_len"]
        ):
            with torch.no_grad():
                response = ppo_trainer.generate(
                    query_ids,
                    max_new_tokens=gen_len,
                    min_new_tokens=40,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = response.squeeze()[query_ids.shape[-1] :]
                response_tensors.append(response)
                # create prompts
                tokenizer_decode = tokenizer.decode(response)
                decode = tokenizer_decode.replace("</s>", "").replace("<s>", "")
                content_preservation = (
                    str.format(
                        """###Task Description:
                   An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
                   1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
                   2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
                   3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
                   4. Please do not generate any other opening, closing, and explanations.

                   ###Response to evaluate:
                   {}

                   ###Reference Answer (Score 5):
                   {}

                   ###Score Rubrics:
                   [Does the response has preserve the content as the reference answer?]
                   Score 1: The content of the two excerpts is completely different, with no overlap in topic, details, or keywords.
                   Score 2: The excerpts show very minimal content preservation, with slight overlaps in topic, a few details, or keywords, but the order and presence of keywords are largely different.
                   Score 3: The excerpts have a moderate level of content preservation, sharing several elements such as key topics, details, or keywords, but the order of keywords might be slightly different, leading to noticeable differences in content.
                   Score 4: The excerpts are very similar in content, preserving most of the key topics and details, including the presence and order of most keywords, with only minor deviations.
                   Score 5: The excerpts have identical content, perfectly mirroring each other in terms of topic, key details, and the presence and order of all keywords.
                   ###Feedback:""",
                        decode,
                        tokenizer.decode(ground_id),
                    )
                    .replace("\n", " ")
                    .replace("\t", " ")
                    .strip()
                )
                style_accuracy = (
                    str.format(
                        """###Task Description:
                   An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
                   1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
                   2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
                   3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
                   4. Please do not generate any other opening, closing, and explanations.

                   ###Response to evaluate:
                   {}

                   ###Reference Answer (Score 5):
                   {}

                   ###Score Rubrics:
                   [Does the response exhibit conciseness, simplicity, directness, abbreviation, telegraphic style, and punctuation similar to the reference answer?]
                   Score 1: The response is verbose, complex, indirect, elaborated, lacks a telegraphic style, and has significantly different punctuation.
                   Score 2: The response shows minimal alignment with the reference in terms of conciseness, simplicity, directness, abbreviation, telegraphic style, or punctuation.
                   Score 3: The response has a moderate level of similarity, sharing several aspects of conciseness, simplicity, directness, abbreviation, telegraphic style, and punctuation, but also has noticeable differences.
                   Score 4: The response is very similar to the reference, sharing most aspects of conciseness, simplicity, directness, abbreviation, telegraphic style, and punctuation, with only minor differences.
                   Score 5: The response is identical to the reference in terms of conciseness, simplicity, directness, abbreviation, telegraphic style, and punctuation.
                   ###Feedback:""",
                        decode,
                        tokenizer.decode(ground_id),
                    )
                    .replace("\n", " ")
                    .replace("\t", " ")
                    .strip()
                )

                fluency = (
                    str.format(
                        """###Task Description:
                   An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
                   1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
                   2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
                   3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
                   4. Please do not generate any other opening, closing, and explanations.

                   ###Response to evaluate:
                   {}

                   ###Reference Answer (Score 5):
                   {}

                   ###Score Rubrics:
                   [Does the response accurately and effectively assess the fluency of the sentence?]
                   Score 1: The assessment is entirely irrelevant or off-topic, with no consideration of fluency factors like grammar, vocabulary, coherence, or readability.
                   Score 2: The assessment shows minimal relevance, with slight consideration of some fluency factors but major inaccuracies or omissions.
                   Score 3: The assessment is moderately accurate, considering several fluency factors like grammar, vocabulary, and readability, but with some inaccuracies or inconsistencies.
                   Score 4: The assessment is very accurate, thoroughly considering fluency factors with minor inaccuracies or omissions.
                   Score 5: The assessment is perfect, accurately considering all fluency factors like grammar, vocabulary, coherence, and readability, and assigns an accurate fluency score.                   ###Feedback:""",
                        decode,
                        tokenizer.decode(ground_id),
                    )
                    .replace("\n", " ")
                    .replace("\t", " ")
                    .strip()
                )
                batch_gen_inputs = evaluator_tokenizer(
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
                scores = evaluator_model.generate(
                    **batch_gen_inputs,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    max_new_tokens=512,
                    repetition_penalty=1.03,
                )[:, -512:]

                def extract_score(score_ids):
                    feedback = evaluator_tokenizer.decode(score_ids.squeeze())
                    pattern = r"(?:[\d]+|[\d]+\.[\d]+)"
                    if "[RESULT]" in feedback:
                        findall = re.findall(
                            pattern,
                            feedback.split("[RESULT]")[1],
                        )
                        return (float(eval(findall[0]) - 1)) / 4 if len(findall) == 1 else 0

                    else:
                        logging.warning(f"NO SCORE:\n {feedback}")
                        return 0

                # eval to create tensor
                aggregate_score = sum([extract_score(score_ids) for score_ids in scores]) / 3
                reward = torch.FloatTensor([float(aggregate_score)])
                rewards.append(reward)

                # get logs
                style_rewards.append(extract_score(scores[0]))
                content_preservation_rewards.append(extract_score(scores[1]))
                fluency_rewards.append(extract_score(scores[2]))

                style_feedbacks.append(evaluator_tokenizer.decode(scores[0].squeeze()))
                content_preservation_feedbacks.append(
                    evaluator_tokenizer.decode(scores[1].squeeze())
                )
                fluency_feedbacks.append(evaluator_tokenizer.decode(scores[2].squeeze()))

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
                    "style_rewards": style_rewards,
                    "content_preservation_rewards": content_preservation_rewards,
                    "fluency_rewards": fluency_rewards,
                    "style_feedbacks": style_feedbacks,
                    "content_preservation_feedbacks": content_preservation_feedbacks,
                    "fluency_feedbacks": fluency_feedbacks,
                }
            )
        )
        ppo_trainer.accelerator.log({"train/batch_sample_predictions": batch_table})

        # Run PPO step
        stats = ppo_trainer.step(queries_ids, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        torch.cuda.empty_cache()
        # Save model every 20 epochs
        if epoch % 20 == 0:
            if ppo_trainer.accelerator.is_main_process:
                model.push_to_hub(
                    f"bio-datasets/{cfg.ppo_config.model_name.split('/')[-1]}-"
                    f"{ppo_trainer.accelerator.get_tracker('wandb').run.name}-epoch-{epoch}"
                )


if __name__ == "__main__":
    main()
