import logging
import re

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

PROMPT = """As a doctor, you must write an original \
'History of Present Illness' (HPI) section for a discharge summary.
Your response should capture the essence of a patient's health journey \
and recent medical experiences, \
while strictly using all the provided keywords conserving the order.
You must adopt a medical telegraphic style, abbreviated, characterized by concise and \
direct language.
Keywords: {}"""


def tokenize(sample, tokenizer, max_sampler_length):
    continuation = sample["text"]
    ground_ids = tokenizer.encode(continuation, add_special_tokens=False)
    ground_ids = (
        ground_ids if len(continuation) <= max_sampler_length else ground_ids[:max_sampler_length]
    )
    sample["ground_texts"] = tokenizer.decode(ground_ids)
    keywords = ",".join(
        [keyword for keyword in sample["keywords"].split(",") if keyword in sample["ground_texts"]]
    )
    prompt = str.format(
        PROMPT,
        keywords,
    )
    sample["input_ids"] = tokenizer.encode("[INST] " + prompt + " [/INST]")
    sample["formatted_query"] = "[INST] " + prompt + " [/INST]"
    sample["query"] = prompt
    sample["keywords"] = keywords
    sample["max_gen_len"] = len(ground_ids)
    return sample


def build_dataset(dataset_name, model_name, max_sampler_length):
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
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")

    ds_dict = {"keywords": [], "text": []}
    for keywords, text in zip(ds["keywords"], ds["text"]):
        for kw, t in zip(keywords, text):
            ds_dict["keywords"].append(kw)
            ds_dict["text"].append(t)
    ds = Dataset.from_dict(ds_dict)
    ds = ds.map(
        tokenize,
        batched=False,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_sampler_length": max_sampler_length,
        },
    )
    ds = ds.filter(lambda x: len(x["keywords"].split(",")) > 1)
    ds.set_format(type="torch")
    return ds


def extract_score(feedback):
    pattern = r"(?:[\d]+|[\d]+\.[\d]+)"
    if "[RESULT]" in feedback:
        findall = re.findall(
            pattern,
            feedback.split("[RESULT]")[1],
        )
        return float(eval(findall[0])) if len(findall) == 1 else 0

    else:
        logging.warning(f"NO SCORE:\n {feedback}")
        return 0


def split_dataset(dataset, sft_ratio, dpo_ratio):
    # Split the dataset into train, gen and test
    # first we split the dataset into train and test
    sft_dataset, test_dataset = dataset.train_test_split(
        train_size=dpo_ratio, shuffle=False
    ).values()
    # then we split the train dataset into train and gen
    sft_dataset, gen_dataset = sft_dataset.train_test_split(
        train_size=sft_ratio, shuffle=False
    ).values()
    return sft_dataset, gen_dataset, test_dataset


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
