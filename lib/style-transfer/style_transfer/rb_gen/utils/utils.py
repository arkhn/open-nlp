import datasets
from datasets import Dataset, load_dataset
from tokenizers.implementations import BaseTokenizer
from transformers import AutoTokenizer


def tokenize(sample: dict, tokenizer: BaseTokenizer, max_sampler_length: int, prompt: str) -> dict:
    """Tokenize the sample.

    Args:
        sample: The sample to tokenize.
        tokenizer: The tokenizer to use.
        max_sampler_length: The maximum length of the input sequence.
        prompt: The prompt to use.

    Returns:
        The tokenized sample.
    """
    continuation = sample["text"]
    ground_ids = tokenizer.encode(continuation, add_special_tokens=False)
    ground_ids = (
        ground_ids if len(continuation) <= max_sampler_length else ground_ids[:max_sampler_length]
    )
    sample["ground_texts"] = tokenizer.decode(ground_ids)
    keywords = ",".join(
        [keyword for keyword in sample["keywords"].split(",") if keyword in sample["ground_texts"]]
    )
    text_prompt = str.format(
        prompt,
        keywords,
    )
    sample["input_ids"] = tokenizer.encode("[INST] " + text_prompt + " [/INST]")
    sample["formatted_query"] = "[INST] " + text_prompt + " [/INST]"
    sample["query"] = text_prompt
    sample["keywords"] = keywords
    sample["max_gen_len"] = len(ground_ids)
    return sample


def build_dataset(
    dataset_name: str, model_name: str, max_sampler_length: int, prompt: str
) -> Dataset:
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name: The name of the dataset.
        model_name: The name of the model.
        max_sampler_length: The maximum length of the input sequence.
        prompt: The prompt to use.

    Returns:
        The dataset for training / testing.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")

    ds_dict: dict = {"keywords": [], "text": []}
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
            "prompt": prompt,
        },
    )
    ds = ds.filter(lambda x: len(x["keywords"].split(",")) > 1)
    ds.set_format(type="torch")
    return ds


def split_dataset(
    dataset: datasets.Dataset, sft_ratio: float, dpo_ratio: float
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """Split the dataset into train, gen and test.

    Args:
        dataset: The dataset to split.
        sft_ratio: The ratio of the dataset to use for SFT.
        dpo_ratio: The ratio of the dataset to use for DPO.

    Returns:
        The train, gen and test datasets
    """
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


def add_prompt(data_point: dict, prompt: str) -> dict:
    """Add prompt to the data point.

    Args:
        data_point: The data point to add prompt to.

    Returns:
        The data point with prompt added.
    """
    data_point["text"] = (
        str.format(
            prompt,
            data_point["keywords"],
        )
        + "\n"
        + data_point["text"]
    )
    return data_point
