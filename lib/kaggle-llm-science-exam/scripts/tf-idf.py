"""File to compute dataset augmentation with articles from Wikipedia using TF-IDF.

This script can run on a CPU, as it's made mainly with sklearn, polars and datasets, however it
takes just a bit more than one hour to run on my MacOS CPU (Intel chip).
"""

from pathlib import Path
from time import time
from typing import Any, Optional

import polars as pl
import sklearn.metrics
from datasets import Dataset, DatasetDict, load_dataset
from datasets.formatting.formatting import LazyRow
from kaggle_llm_science_exam._path import _ROOT
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

wikipedia_dir_path: Path = _ROOT / "data" / "wikipedia"
wikipedia_index_file_name: Path = Path("wiki_2023_index.parquet")
wikipedia_index_limit: Optional[int] = None  # If not None, truncate the database
ds_dir_path: Path = _ROOT / "kaggle_dataset"
ds_file_name: Path = Path("LLM Science Exam Train.csv")
wikipedia_file_name_suffix: str = "_augmented"


t0: float = time()
wikipedia_index_file_path: Path = wikipedia_dir_path / wikipedia_index_file_name
df_wikipedia_index: pl.DataFrame = pl.read_parquet(
    wikipedia_index_file_path,
    n_rows=wikipedia_index_limit,
)
t1: float = time()
logger.info(f"Data reading took {t1 - t0:.2f} seconds")

t0 = t1
wikipedia_contexts: list[str] = list(df_wikipedia_index["context"])
tfidfvectorizer: TfidfVectorizer = TfidfVectorizer(
    analyzer="word",
    stop_words="english",
)
tfidf_matrix_wikipedia_contexts = tfidfvectorizer.fit_transform(wikipedia_contexts)
t1 = time()
logger.info(f"Wikipedia TF-IDF vectors computation took {t1 - t0:.2f} seconds")
# shape: (wikipedia files number, vocababulary size)
logger.debug(f"TDIF matrix shape: {tfidf_matrix_wikipedia_contexts.shape}")

t0 = t1
ds_file_path: Path = ds_dir_path / ds_file_name
ds_dict = load_dataset("csv", data_files=[str(ds_file_path)])
if not isinstance(ds_dict, DatasetDict):
    raise TypeError("Expected a DatasetDict")
if not ds_dict.keys() == {"train"}:
    raise ValueError("Expected only a train dataset")
ds: Dataset = ds_dict["train"]
t1 = time()
logger.info(f"Dataset loading took {t1 - t0:.2f} seconds")

t0 = t1
dataset_prompts: list[str] = list(ds["prompt"])
tfidf_matrix_dataset_prompts = tfidfvectorizer.transform(dataset_prompts)
t1 = time()
logger.info(f"Dataset TF-IDF vectors computation took {t1 - t0:.2f} seconds")
# shape: (dataset size, vocababulary size)
logger.debug(f"TDIF matrix shape: {tfidf_matrix_dataset_prompts.shape}")

t0 = t1
df_similarity: pl.DataFrame = pl.from_numpy(
    sklearn.metrics.pairwise.cosine_similarity(
        tfidf_matrix_dataset_prompts,
        tfidf_matrix_wikipedia_contexts,
        dense_output=True,
    )
)
t1 = time()
logger.info(f"Dataset TF-IDF vectors computation took {t1 - t0:.2f} seconds")
# shape: (dataset size, wikipedia files number)
logger.debug(f"Similarity DataFrame shape: {df_similarity.shape}")


def find_closest_wikipedia_file(row: LazyRow) -> dict[str, Any]:
    if "id" not in row:
        raise KeyError
    row_idx = row["id"]
    if not isinstance(row_idx, int):
        raise TypeError

    s_similarity: pl.Series = pl.Series(df_similarity.row(row_idx))

    wikipedia_score = s_similarity.max()
    if not isinstance(wikipedia_score, float):
        raise TypeError
    wikipedia_idx = s_similarity.arg_max()
    if not isinstance(wikipedia_idx, int):
        raise TypeError

    wikipedia_file_data: dict[str, Any] = df_wikipedia_index.row(wikipedia_idx, named=True)
    d = {f"wikipedia_{key}": value for key, value in wikipedia_file_data.items()}
    d["wikipedia_score"] = wikipedia_score

    return d


t0 = t1
ds = ds.map(find_closest_wikipedia_file)
t1 = time()
logger.info(f"Search for closest wikipedia files took {t1 - t0:.2f} seconds")

t0 = t1
if wikipedia_index_limit is not None:
    wikipedia_file_name_suffix += f"_{wikipedia_index_limit}"
augmented_dataset_file_name: str = (
    f"{ds_file_name.stem}{wikipedia_file_name_suffix}{ds_file_name.suffix}"
)
augmented_dataset_file_path: Path = ds_dir_path / augmented_dataset_file_name
ds.to_csv(augmented_dataset_file_path)
t1 = time()
logger.info(f"File writing took {t1 - t0:.2f} seconds")
