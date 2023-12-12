"""Vanilla Mistral used to generate 3 answers
Result on 200 samples: 0.5
"""

# Requirements:
# !pip install accelerate
# !pip install bitsandbytes
# !pip install wandb


import pandas as pd
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

dataset = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv")


model_name_or_path = "/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1"
bnb_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
}
generation_config = {"max_new_tokens": 1000, "do_sample": True}
run = wandb.init(  # type: ignore
    project="lg+sj_science_exam_llm",
    group="vanilla_LLM",
    config={
        "model_name_or_path": model_name_or_path,
        "generation_config": generation_config,
        "bnb_config": bnb_config,
    },
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=BitsAndBytesConfig(**bnb_config),
)


def generate_decoded_values_v1(row, model, tokenizer):
    prompt = row["prompt"]
    for column in ["A", "B", "C", "D", "E"]:
        prompt += f"\n{column}: {row[column]}"
    prompt += """
    Answer by only giving up to three letter corresponding to most probable correct answers, in \
    decreasing probability order.
    Very important, your response must have this format: 'Letter 1, Letter 2, Letter 3'. \
    Letter1 corresponding to the most correct answer.
    """

    questions = [{"role": "user", "content": prompt}]

    encodeds = tokenizer.apply_chat_template(questions, return_tensors="pt")
    model_inputs = encodeds
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0]  # Assuming you want a single decoded value for each row


# Assuming 'dataset' is your original dataset
decoded_values = dataset.apply(
    lambda row: generate_decoded_values_v1(row, model, tokenizer), axis=1
)

# Add the new column to your dataset
dataset["decoded_values_v1"] = decoded_values


def generate_decoded_values_v2(row, model, tokenizer):
    prompt = row["prompt"]
    for column in ["A", "B", "C", "D", "E"]:
        prompt += f"\n{column}: {row[column]}"
    prompt += """
    Answer by only giving up to three letter corresponding to most probable correct answers, in \
    decreasing probability order.
    Very important, your response must have this format and shouldn't contain anything else \
    in your response sentence: 'Letter 1, Letter 2, Letter 3'. Letter 1 corresponding to the \
    letter of the most correct answer between A, B, C, D, and E.
    """

    questions = [{"role": "user", "content": prompt}]

    encodeds = tokenizer.apply_chat_template(questions, return_tensors="pt")
    model_inputs = encodeds
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)

    return decoded[0]  # Assuming you want a single decoded value for each row


# Assuming 'dataset' is your original dataset
decoded_values = dataset.apply(
    lambda row: generate_decoded_values_v1(row, model, tokenizer), axis=1
)

# Add the new column to your dataset
dataset["decoded_values_v2"] = decoded_values


def score(act, pred):
    if len(pred) == 1:
        return 1 if act == pred[0] else 0
    if len(pred) == 2:
        return 1 if act == pred[0] else 2 / 3 if act == pred[1] else 0
    else:
        return 1 if act == pred[0] else 2 / 3 if act == pred[1] else 1 / 3 if act == pred[2] else 0


def get_output_letters(output_string: str) -> str:
    pattern = "[/INST]"
    pattern_position = output_string.find(pattern)
    if pattern_position != -1 and pattern_position + len(pattern) + 8 < len(output_string):
        result_string = output_string[
            pattern_position + len(pattern) : pattern_position + len(pattern) + 8
        ]
    else:
        result_string = output_string[pattern_position + len(pattern) + 1]
    result_string = result_string.replace(",", "").replace(" ", "")
    return result_string


dataset["output_letters"] = dataset.apply(
    lambda row: get_output_letters(row["decoded_values_v2"]), axis=1
)

output_string = dataset.iloc[0]["decoded_values_v1"]
pattern = "[/INST]"
pattern_position = output_string.find(pattern)
result_string = output_string[pattern_position + len(pattern) : pattern_position + len(pattern) + 8]
result_string = result_string.replace(",", "").replace(" ", "")


scores_v2 = dataset.apply(lambda row: score(row["answer"], row["output_letters"]), axis=1)

dataset["score_values_v2"] = scores_v2


mean_score_v2 = dataset["score_values_v2"].mean()

print(f"Mean Score (score_values_v2): {mean_score_v2}")

wandb.log({"score": mean_score_v2})
wandb.log({"result": wandb.Table(dataframe=dataset)})
