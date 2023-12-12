"""
This script was a submission for the Kaggle LLM Science Exam competition.
It consists in finetuning Mistral on the training dataset and then using the model to generate 3
answers for the test dataset.
Result:
Private score: 0.636344
Public score: 0.628173
Execution time: 27 minutes
"""

# Requirements:
# !pip in stall accelerate
# !pip install bitsandbytes
# !pip install peft
# !pip install transformers
# !pip install trl

import re

import pandas as pd
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig, PeftModel, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

seed = 42
transformers.set_seed(seed)

prompt_template = """### Question: {}
A: {}
B: {}
C: {}
D: {}
E: {}
Answer by only giving up the letter corresponding to the correct answer. Very important, your \
response must have this format: 'Letter 1'. Letter1 corresponding to the most correct answer.
### Answer:"""
response_template = "Answer:"
base_model_name = "/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1"
bits_and_bytes_config = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": False,
}
lora_config = {
    "r": 64,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
training_config = {
    "output_dir": "/kaggle/working/results",
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "optim": "paged_adamw_32bit",
    "save_steps": 10,
    "logging_steps": 5,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "fp16": False,
    "bf16": False,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "group_by_length": True,
    "lr_scheduler_type": "constant",
    "report_to": "none",
}
generation_config = {
    "max_new_tokens": 1,
    "do_sample": False,
    "num_beams": 3,
    "num_beam_groups": 3,
    "diversity_penalty": 4.0,
    "num_return_sequences": 3,
    "return_dict_in_generate": True,
    "output_scores": True,
    "pad_token_id": 50256,
}

df_train_dataset = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv")
df_test_dataset = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv")


def create_prompt(example, with_answer: bool):
    prompt = prompt_template.format(
        example["prompt"],
        example["A"],
        example["B"],
        example["C"],
        example["D"],
        example["E"],
    )
    if with_answer:
        prompt += " " + example["answer"]
    example["text"] = prompt
    return example


train_dataset = Dataset.from_pandas(df_train_dataset)
train_dataset = train_dataset.map(lambda example: create_prompt(example, with_answer=True))
test_dataset = Dataset.from_pandas(df_test_dataset)
test_dataset = test_dataset.map(lambda example: create_prompt(example, with_answer=False))

# Load model and tokenizer with bits and bytes

bnb_config = BitsAndBytesConfig(**bits_and_bytes_config)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
base_model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

# PEFT config

peft_params = LoraConfig(**lora_config)
base_model = prepare_model_for_kbit_training(base_model)

# Training

training_args = TrainingArguments(**training_config)
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    args=training_args,
    data_collator=DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    ),
    peft_config=peft_params,
)

trainer.train()

trainer.model.save_pretrained("/kaggle/working/peft_model")

config = PeftConfig.from_pretrained("/kaggle/working/peft_model")
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, "/kaggle/working/peft_model", is_trainable=False)
model.config.use_cache = True

submission_records = []

for example in tqdm(test_dataset, total=len(test_dataset)):
    model_inputs = tokenizer(example["text"], return_tensors="pt").to("cuda:0")
    outputs = model.generate(**model_inputs, **generation_config)
    predictions_scores = []
    for output, score in zip(outputs["sequences"], outputs["sequences_scores"]):
        response = tokenizer.decode(output, skip_special_tokens=True)
        predictions = re.findall(f"{response_template.lower()}  ?(a|b|c|d|e)", response.lower())
        if predictions:
            predictions_scores.append((predictions[0].upper(), float(score)))
        predictions_scores.sort(key=lambda item: item[1], reverse=True)
    submission_records.append(
        {"id": example["id"], "prediction": " ".join([item[0] for item in predictions_scores])}
    )

submission = pd.DataFrame.from_records(submission_records)

submission.to_csv("/kaggle/working/submission.csv", index=False)
