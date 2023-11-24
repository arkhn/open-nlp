"""File to run the evaluation of an LLM on the task.

When used in Kaggle's notebooks, the following lines must be run beforehand, and the notebook 
must probably be reset as well:

```
# !pip install loguru
# !pip install datasets==2.14.6
# !pip install bitsandbytes
```

This script yields a score around 0.4741.
"""

from pathlib import Path  # noqa: F401
from typing import Optional

import transformers
from datasets import Dataset, DatasetDict, load_dataset
from kaggle_llm_science_exam._path import _ROOT  # noqa: F401
from kaggle_llm_science_exam.metric import score_batch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GenerationConfig

# To use locally, change the following line
DATASET_PATH: str | Path = "/kaggle/input/kaggle-llm-science-exam/train.csv"
# DATASET_PATH: str | Path = _ROOT / "kaggle_dataset" / "LLM Science Exam Train.csv"

# Global parameters
DATASET_MAX_LENGTH: Optional[int] = None  # If not None, truncate the dataset
PRETRAINED_MODEL_NAME_OR_PATH: str = "mistralai/Mistral-7B-v0.1"
NUM_BEAMS: int = 3  # Must be >= MAX_RETURN_SEQUENCES
NUM_RETURN_SEQUENCES = 3
MAX_NEW_TOKENS = 3
BATCH_SIZE = 1
N_BATCHES: Optional[int] = None  # If not None, number of batches to process

# Load and create a Dataset object
dataset_dict = load_dataset("csv", data_files=[str(DATASET_PATH)])
if not isinstance(dataset_dict, DatasetDict):
    raise TypeError("Expected a DatasetDict")
if not dataset_dict.keys() == {"train"}:
    raise ValueError("Expected only a train dataset")
dataset: Dataset = dataset_dict["train"]
logger.info("Dataset loaded")
for col in dataset.column_names:
    logger.info(f"{col}: {dataset[col][:3]}")

# Truncate the dataset
if DATASET_MAX_LENGTH is not None:
    dataset = dataset.select(range(DATASET_MAX_LENGTH))
    logger.info("Dataset truncated")

# Reformat the prompt and choices as a prompt
dataset = dataset.map(
    lambda row: {
        "question": f"""
        {row["prompt"]}\n
        A: {row["A"]}\n
        B: {row["B"]}\n
        C: {row["C"]}\n
        D: {row["D"]}\n
        E: {row["E"]}\n
        Answer= """,
    }
)
logger.info("Dataset reformatted")

# Create a dataloader
data_loader = DataLoader(
    dataset,  # type: ignore
    batch_size=BATCH_SIZE,
    shuffle=False,
)
logger.info("Data loader created")

# load model with huggingface
model = transformers.AutoModelForCausalLM.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    device_map="auto",
    offload_folder="offload",
    load_in_4bit=True,
)
logger.info("Model loaded")

tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    padding_side="left",
)
tokenizer.pad_token = tokenizer.eos_token
logger.info("Tokenizer loaded")

generation_config = GenerationConfig(
    max_length=100,
    num_beams=NUM_BEAMS,
    no_repeat_ngram_size=2,
    num_return_sequences=NUM_RETURN_SEQUENCES,
    early_stopping=True,
    max_new_tokens=MAX_NEW_TOKENS,
    pad_token_id=tokenizer.eos_token_id,
)
logger.info("Generation config loaded")

# iterate over dataloader using generate
questions_answers: list[list[str]] = []
scores: list[float] = []
for i, batch_data in enumerate(data_loader):
    if N_BATCHES is not None and i == N_BATCHES:
        break
    logger.info(f"Batch {i}")

    # Shape: BATCH_SIZE * input sequence length
    batch_tokens: dict[str, Tensor] = tokenizer(
        batch_data["question"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    # shape: (NUM_RETURN_SEQUENCES * BATCH_SIZE) * (input sequence_length + MAX_NEW_TOKENS)
    batch_output_ids: Tensor = model.generate(
        **batch_tokens,
        generation_config=generation_config,
    )

    # reformat the output by removing from the output the input sequence
    # shape: (NUM_RETURN_SEQUENCES * BATCH_SIZE) * MAX_NEW_TOKENS
    batch_output_ids = batch_output_ids[:, -MAX_NEW_TOKENS:]

    # reshape the output to have a sequence for each input
    # shape: BATCH_SIZE * NUM_RETURN_SEQUENCES * MAX_NEW_TOKENS
    batch_output_ids = batch_output_ids.view(BATCH_SIZE, NUM_RETURN_SEQUENCES, -1)

    # decode the output
    # shape: BATCH_SIZE * NUM_RETURN_SEQUENCES
    batch_generated_answers_lists: list[list[str]] = [
        tokenizer.batch_decode(batch_output_ids[i], skip_special_tokens=True)
        for i in range(BATCH_SIZE)
    ]
    logger.info(f"{batch_generated_answers_lists}")

    # compute the scores for each question
    batch_scores: list[float] = score_batch(
        generated_answers_lists=batch_generated_answers_lists,
        reference_answers=batch_data["answer"],
    )
    logger.info(f"{batch_scores}")

    questions_answers.extend(batch_generated_answers_lists)
    scores.extend(batch_scores)

global_score = sum(scores) / len(scores)
logger.info(f"Global score: {global_score}")
