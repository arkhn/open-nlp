"""File to run the evaluation of an LLM on the task.

When used in Kaggle's notebooks, the following lines must be run beforehand, and the notebook 
must probably be reset as well:

```
# !pip install loguru
# !pip install datasets==2.14.6
```

This script yields a very small score below 0.1.
"""

from pathlib import Path
from typing import Optional

import transformers
from datasets import Dataset, DatasetDict, load_dataset
from kaggle_llm_science_exam.metric import score_batch
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GenerationConfig

# To use locally, change the following line
DATASET_PATH: str | Path = "/kaggle/input/kaggle-llm-science-exam/train.csv"
# from kaggle_llm_science_exam._path import _ROOT
# DATASET_PATH: str | Path = _ROOT / "kaggle_dataset" / "LLM Science Exam Train.csv"

# Global parameters
DATASET_MAX_LENGTH: Optional[int] = None  # If not None, truncate the dataset
PRETRAINED_MODEL_NAME_OR_PATH: str = "gpt2-medium"
NUM_BEAMS: int = 6  # Must be >= MAX_RETURN_SEQUENCES
NUM_RETURN_SEQUENCES = 6
MAX_NEW_TOKENS = 3
BATCH_SIZE = 5
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
    batch_answers: list[list[str]] = [
        tokenizer.batch_decode(batch_output_ids[i], skip_special_tokens=True)
        for i in range(BATCH_SIZE)
    ]
    logger.info(f"{batch_answers}")

    # compute the score
    batch_scores: list[float] = score_batch(
        generated_answers_lists=batch_answers,
        reference_answers=batch_data["answer"],
    )
    logger.info(f"{batch_scores}")

    questions_answers.extend(batch_answers)
    scores.extend(batch_scores)

global_score = sum(scores) / len(scores)
logger.info(f"Global score: {global_score}")
