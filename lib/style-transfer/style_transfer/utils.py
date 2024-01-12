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

EVAL_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, \
a reference answer that gets a score of 5, and a score rubric representing \
a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly \
based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. \
You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for \
criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.\n
###The instruction to Evaluate:
{}\n
###Response to evaluate:
{}\n
###Reference Answer (Score 5):
{}\n
###Score Rubrics:
[Does the response reproduce the exact same writing style as the reference answer?]
Score 1: The response significantly deviates from the medical telegraphic style, showing a lack \
of concise and direct language. It does not resemble the style of the reference answer, \
with lengthy sentences and unrelated details.
Score 2: The response shows an attempt to use the medical telegraphic style, \
but it is not consistent. There are moments of conciseness, \
yet the overall style is not as direct or brief as the reference answer.
Score 3: The response adopts a medical telegraphic style to a moderate extent. \
It is more concise and direct than lower scores but still contains elements \
that are not as efficiently presented as in the reference answer.
Score 4: The response closely follows the medical telegraphic style, \
with concise and direct language. It slightly differs from the reference answer in terms of \
efficiency and clarity but still maintains a high standard of the required style.
Score 5: The response perfectly aligns with the medical telegraphic style, characterized by \
concise and direct language. \
It mirrors the style of the reference answer, \
effectively conveying information in a brief and clear manner.\n
###Feedback:"""


def tokenize(sample, tokenizer, max_sampler_length):
    continuation = sample["text"]
    ground_ids = tokenizer.encode(continuation, add_special_tokens=False)
    ground_ids = (
        ground_ids if len(continuation) > max_sampler_length else ground_ids[:max_sampler_length]
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

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]
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
