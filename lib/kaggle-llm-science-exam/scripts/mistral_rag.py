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

import string
from typing import Optional

import cudf
import pysbd
import torch
import transformers
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.metrics import pairwise_distances
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from transformers.utils import quantization_config

try:
    from pandas.api.types import is_datetime64tz_dtype
except ImportError:
    # pandas < 0.19.2
    pass

# Global parameters
DATASET_PATH = "/kaggle/input/kaggle-llm-science-exam/train.csv"
DATASET_MAX_LENGTH: Optional[int] = None  # If not None, truncate the dataset
WIKI_DS_PATH = "/kaggle/input/wikipedia-20230701"
WIKI_COLUMN = "title"
PRETRAINED_MODEL_NAME_OR_PATH: str = "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1"
NUM_BEAMS: int = 3  # Must be >= MAX_RETURN_SEQUENCES
NUM_RETURN_SEQUENCES = 3
MAX_NEW_TOKENS = 3
BATCH_SIZE = 1
N_BATCHES: Optional[int] = None  # If not None, number of batches to process

tfidf_vectorizer_config = dict(
    stop_words="english",
    analyzer="word",
    ngram_range=(1, 2),
    sublinear_tf=True,
)

pysbd_config = dict(
    language="en",
    clean=False,
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

generation_config = GenerationConfig(
    max_length=100,
    num_beams=NUM_BEAMS,
    no_repeat_ngram_size=2,
    num_return_sequences=NUM_RETURN_SEQUENCES,
    early_stopping=True,
    max_new_tokens=MAX_NEW_TOKENS,
    output_scores=True,
    return_dict_in_generate=True,
)


def prompting_function(row) -> dict:
    seg = pysbd.Segmenter(**pysbd_config)
    prompt = row["prompt"]
    similar_wiki_context = similar_contexts[row["id"]][:1500]
    wiki_sentences = seg.segment(similar_wiki_context)

    prompt_tfidf_vector = tfidf_vectorizer.transform(prompt)
    wiki_tfidf_matrix = tfidf_vectorizer.transform(wiki_sentences)

    similarity_scores = pairwise_distances(prompt_tfidf_vector, wiki_tfidf_matrix, metric="cosine")
    similarity_scores_tensor = torch.Tensor(similarity_scores)
    topk = torch.topk(similarity_scores_tensor, k=10, largest=False)

    indices = topk.indices
    wiki_similar_sentences = [wiki_sentences[i] for i in indices]

    return {
        "question": f"""
        {row["prompt"]}\n
        {' '.join(wiki_similar_sentences)}\n
        A: {row["A"]}\n
        B: {row["B"]}\n
        C: {row["C"]}\n
        D: {row["D"]}\n
        E: {row["E"]}\n
        Answer= """,
    }


# Augment Dataset with context
test = cudf.read_csv(DATASET_PATH)
if DATASET_MAX_LENGTH is not None:
    test = test[:DATASET_MAX_LENGTH]

# Extract titles
wiki_index = cudf.read_parquet(f"{WIKI_DS_PATH}/wiki_2023_index.parquet")
wiki_contexts = wiki_index[WIKI_COLUMN]


def calculate_similarity(row):
    # Extract the vector for the current title
    prompt_vector = tfidf_vectorizer.transform(row["prompt"])
    # Calculate cosine similarity
    similarity_scores = pairwise_distances(prompt_vector, tfidf_matrix, metric="cosine")
    idx = torch.argmin(torch.Tensor(similarity_scores))
    logger.debug(torch.min(torch.Tensor(similarity_scores)))
    # Returning the similarity scores as a list
    alpha_parquet = cudf.read_parquet(
        f"{WIKI_DS_PATH}/{wiki_index.loc[idx]['file'].to_pandas().values[0]}"
    )
    return (
        alpha_parquet[alpha_parquet["id"] == wiki_index.loc[idx]["id"].to_pandas().values[0]][
            "text"
        ]
        .to_pandas()
        .values[0]
    )


# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(**tfidf_vectorizer_config)

# Fit and transform the titles
tfidf_matrix = tfidf_vectorizer.fit_transform(wiki_contexts)
similar_contexts = []
for row_index in range(len(test)):
    similar_contexts.append(calculate_similarity(test.iloc[row_index]))

# Load and create a Dataset object
dataset_dict = load_dataset("csv", data_files=[str(DATASET_PATH)])
if not isinstance(dataset_dict, DatasetDict):
    raise TypeError("Expected a DatasetDict")

dataset: Dataset = dataset_dict["train"]
# Truncate the dataset
if DATASET_MAX_LENGTH is not None:
    dataset = dataset.select(range(DATASET_MAX_LENGTH))
    logger.info("Dataset truncated")
logger.info("Dataset loaded")
for col in dataset.column_names:
    logger.info(f"{col}: {dataset[col][:3]}")

# Reformat the prompt and choices as a prompt
dataset = dataset.map(prompting_function)
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
    quantization_config=quantization_config,
)
logger.info("Model loaded")

tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    padding_side="left",
)
tokenizer.pad_token = tokenizer.eos_token
generation_config.pad_token_id = tokenizer.eos_token_id

logger.info("Tokenizer loaded")
logger.info("Generation config loaded")

# iterate over dataloader using generate
questions_answers: list[list[str]] = []
for i, batch_data in enumerate(data_loader):
    if N_BATCHES is not None and i == N_BATCHES:
        break
    logger.info(f"Batch {i}")
    # Shape: BATCH_SIZE * input sequence length
    logger.debug(batch_data["question"])
    batch_tokens: dict[str, Tensor] = tokenizer(
        batch_data["question"],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    # shape: (NUM_RETURN_SEQUENCES * BATCH_SIZE) * (input sequence_length + MAX_NEW_TOKENS)
    batch_outputs: Tensor = model.generate(
        **batch_tokens,
        generation_config=generation_config,
    )
    batch_output_ids = batch_outputs["sequences"]
    batch_output_scores = batch_outputs["sequences_scores"]

    # reformat the output by removing from the output the input sequence
    # shape: (NUM_RETURN_SEQUENCES * BATCH_SIZE) * MAX_NEW_TOKENS
    batch_output_ids = batch_output_ids[:, -MAX_NEW_TOKENS:]
    # reshape the output to have a sequence for each input
    # shape: BATCH_SIZE * NUM_RETURN_SEQUENCES * MAX_NEW_TOKENS
    batch_output_ids = batch_output_ids.view(BATCH_SIZE, NUM_RETURN_SEQUENCES, -1)
    batch_output_scores = batch_output_scores.view(BATCH_SIZE, NUM_RETURN_SEQUENCES, -1)

    for sequence_scores in list(batch_output_scores):
        if list(sequence_scores) != sorted(list(sequence_scores), reverse=True):
            raise ValueError(
                f"Expected the scores to be sorted in descending order: {sequence_scores}"
            )

    # decode the output
    # shape: BATCH_SIZE * NUM_RETURN_SEQUENCES
    batch_answers: list[list[str]] = [
        [
            answer.strip().translate(str.maketrans("", "", string.punctuation)).upper()
            for answer in tokenizer.batch_decode(batch_output_ids[i], skip_special_tokens=True)
        ]
        for i in range(BATCH_SIZE)
    ]
    logger.info(f"{batch_answers}")
    questions_answers.extend(batch_answers)

scores: list[float] = []
for reference_answer, generated_answers in zip(dataset["answer"], questions_answers):
    # remove duplicates and invalid answers in generated answers
    new_generated_answers = []
    for answer in generated_answers:
        if answer in new_generated_answers:  # duplicates
            continue
        if answer not in {"A", "B", "C", "D", "E"}:  # invalid answer
            continue
        new_generated_answers.append(answer)
    # truncate & pad answers to have exactly 3 answers
    new_generated_answers = new_generated_answers[:3]
    while len(new_generated_answers) < 3:
        new_generated_answers.append("NA")
    score: float = 0
    for i, answer in enumerate(new_generated_answers):  # k is i+1
        if answer != reference_answer:
            continue
        precision_at_k = 1 / (i + 1)  # shortcut assuming that there is only one good answer
        score += precision_at_k
    scores.append(score)

global_score = sum(scores) / len(scores)
logger.info(f"Global score: {global_score}")
