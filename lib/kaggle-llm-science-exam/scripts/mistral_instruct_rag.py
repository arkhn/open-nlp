"""This script yields a score of 0.6699999999999998.

Before running the script, the following packages must be installed:
!pip install loguru
!pip install datasets==2.14.6
#!pip install pandas==1.5.3
!pip install bitsandbytes
!pip install --upgrade pandas scipy cudf cuml
!pip install pysbd
"""

import string
from typing import Optional

import cudf
import pysbd
import torch
import transformers
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.metrics import pairwise_distances
from datasets import Dataset, load_dataset
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig

try:
    from pandas.api.types import is_datetime64tz_dtype  # noqa: F401
except ImportError:
    # pandas < 0.19.2
    pass


# Global parameters
DATASET_PATH = "/kaggle/input/kaggle-llm-science-exam/train.csv"
WIKI_DS_PATH = "/kaggle/input/wikipedia-20230701"
WIKI_DS_INDEX_COLUMN = "title"
SIMILARITY_THRESHOLD: float = -1  # Skip any article with similarity below or equal (-1 to disable)
TOP_SENTENCES: int = 20
SKIP_SIMILARITY_COMPUTATION: bool = False  # If True, skip the similarity computation
DATASET_MAX_LENGTH: Optional[int] = None  # If not None, truncate the dataset
PRETRAINED_MODEL_NAME_OR_PATH: str = "/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1"
NUM_BEAMS: int = 6  # Must be >= MAX_RETURN_SEQUENCES
NUM_RETURN_SEQUENCES = 6
MAX_NEW_TOKENS = 3
BATCH_SIZE = 1  # A larger batch size will lead to OOM if not reducing the memory usage
N_BATCHES: Optional[int] = None  # If not None, number of batches to process

tfidf_vectorizer_config = dict(
    stop_words="english",
    analyzer="word",
    ngram_range=(1, 2),
    sublinear_tf=True,
)
data_loader_config = dict(
    batch_size=BATCH_SIZE,
    shuffle=False,
)
quantization_config = dict(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model_config = dict(
    pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
    device_map="auto",
    offload_folder="offload",
)
tokenizer_config = dict(
    pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
    padding_side="left",
)
generation_config_dict = dict(
    max_length=100,
    num_beams=NUM_BEAMS,
    no_repeat_ngram_size=2,
    num_return_sequences=NUM_RETURN_SEQUENCES,
    early_stopping=True,
    max_new_tokens=MAX_NEW_TOKENS,
)

dataset_dict = load_dataset("csv", data_files=[str(DATASET_PATH)])
dataset: Dataset = dataset_dict["train"]
if DATASET_MAX_LENGTH is not None:
    dataset = dataset.select(range(DATASET_MAX_LENGTH))

df_dataset = cudf.read_csv(DATASET_PATH)
df_wiki_index = cudf.read_parquet(f"{WIKI_DS_PATH}/wiki_2023_index.parquet")
wiki_index_context_column = df_wiki_index[WIKI_DS_INDEX_COLUMN]

tfidf_vectorizer = TfidfVectorizer(**tfidf_vectorizer_config)
articles_tfidf_matrix = tfidf_vectorizer.fit_transform(wiki_index_context_column)
seg = pysbd.Segmenter(language="en", clean=False)


def calculate_similarity(row):
    if SKIP_SIMILARITY_COMPUTATION:
        return ""

    prompt_vector = tfidf_vectorizer.transform(row["prompt"])

    articles_similarity_scores = pairwise_distances(
        prompt_vector,
        articles_tfidf_matrix,
        metric="cosine",
    )
    min_output = torch.min(torch.Tensor(articles_similarity_scores), dim=1)
    min_idx = min_output.indices[0]
    min_value = min_output.values[0]
    logger.debug(min_value)

    if min_value <= SIMILARITY_THRESHOLD:
        logger.debug("Similarity below threshold")
        return ""

    articles_file_name: str = df_wiki_index.loc[min_idx]["file"].to_pandas().values[0]
    articles_file_path: str = f"{WIKI_DS_PATH}/{articles_file_name}"
    df_articles = cudf.read_parquet(articles_file_path)
    wiki_article: str = (
        df_articles[df_articles["id"] == df_wiki_index.loc[min_idx]["id"].to_pandas().values[0]][
            "text"
        ]
        .to_pandas()
        .values[0]
    )

    wiki_sentences: list[str] = seg.segment(wiki_article)
    wiki_sentences_series: cudf.Series = cudf.Series(wiki_sentences)
    # transform takes as input a cudf or pandas Series
    sentences_tfidf_matrix = tfidf_vectorizer.transform(wiki_sentences_series)
    sentences_similarity_scores = pairwise_distances(
        prompt_vector,
        sentences_tfidf_matrix,
        metric="cosine",
    )
    topk_output = torch.topk(
        torch.Tensor(sentences_similarity_scores),
        k=min(TOP_SENTENCES, len(wiki_sentences)),
        largest=False,
        sorted=False,
    )
    logger.debug(topk_output.indices)
    similar_sentences_indices: list[int] = sorted(topk_output.indices[0].tolist())
    similar_text: str = " ".join(wiki_sentences[i] for i in similar_sentences_indices)
    logger.debug(similar_text)

    return similar_text


wiki_closest_contexts = []
for row_index in range(len(df_dataset)):
    wiki_closest_contexts.append(calculate_similarity(df_dataset.iloc[row_index]))

# Reformat the prompt and choices as a prompt
dataset = dataset.map(
    lambda row: {
        "question": f"""
        Context: {wiki_closest_contexts[row["id"]][:2000]}\n
        Question: {row["prompt"]}\n
        Possible answers are:\n
        A: {row["A"]}\n
        B: {row["B"]}\n
        C: {row["C"]}\n
        D: {row["D"]}\n
        E: {row["E"]}\n
        What is the right answer between A, B, C, D and E? Answer: """,
    }
)
data_loader = DataLoader(dataset=dataset, **data_loader_config)

model = transformers.AutoModelForCausalLM.from_pretrained(
    **model_config,
    quantization_config=BitsAndBytesConfig(**quantization_config),
)
tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
tokenizer.pad_token = tokenizer.eos_token
generation_config = GenerationConfig(
    **generation_config_dict,
    pad_token_id=tokenizer.eos_token_id,
)


# iterate over dataloader using generate
questions_answers: list[list[str]] = []
scores: list[float] = []
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
    logger.debug(f"{batch_generated_answers_lists}")

    # compute the scores for each question
    batch_scores: list[float] = []
    for generated_answers, reference_answer in zip(
        batch_generated_answers_lists,
        batch_data["answer"],
    ):
        # remove duplicates and invalid answers in generated answers
        new_generated_answers = []
        for answer in generated_answers:
            # preprocess the results
            answer = answer.strip()
            answer = answer.translate(str.maketrans("", "", string.punctuation))
            answer = answer.upper()

            if answer in new_generated_answers:  # duplicates
                continue
            if answer not in {"A", "B", "C", "D", "E"}:  # invalid answer
                continue
            new_generated_answers.append(answer)

        # truncate & pad answers to have exactly 3 answers
        new_generated_answers = new_generated_answers[:3]
        while len(new_generated_answers) < 3:
            new_generated_answers.append("NA")

        # compute the average precision at 3
        score: float = 0
        for i, answer in enumerate(new_generated_answers):  # k is i+1
            if answer != reference_answer:
                continue
            precision_at_k = 1 / (i + 1)  # shortcut assuming that there is only one good answer
            score += precision_at_k
        batch_scores.append(score)

    logger.debug(f"{batch_scores}")

    questions_answers.extend(batch_generated_answers_lists)
    scores.extend(batch_scores)

global_score = sum(scores) / len(scores)
logger.info(f"Global score: {global_score}")
