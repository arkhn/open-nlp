import os
import random
from typing import List

import pandas as pd
from datasets import load_dataset
from quickumls import QuickUMLS
from tqdm import tqdm

OUTPUT_PATH = "datasets/health/qa_clinical_alpacare_raw_umls_v4"
RANDOM_SEED = 42
SEED_SIZE = 30000
PROMPT_PATH = "datasets/preprocessing/health/prompt.txt"

tqdm.pandas()


class KeywordExtractor:
    def __init__(self):
        self.matcher = QuickUMLS(quickumls_fp=os.getenv("QUICKUMLS_PATH"))

    def extract_keywords(self, text: str) -> str:
        if pd.isna(text):
            return ""
        matches = self.matcher.match(
            text.strip(),
            best_match=True,
            ignore_syntax=False,
        )

        # Allowed TUI codes
        filtered_terms = []
        for match_group in matches:
            for match in match_group:
                filtered_terms.append(match["ngram"])
        return ", ".join(set(filtered_terms))


class DataProcessor:
    def __init__(
        self,
        random_seed: int = RANDOM_SEED,
        seed_size: int = SEED_SIZE,
    ):
        self.seed_size = seed_size
        random.seed(random_seed)
        self.extractor = KeywordExtractor()

    def load_and_sample_dataset(self) -> List[dict]:
        return load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")

    @staticmethod
    def reformat_data_sample(sample: dict) -> dict:
        if "input" in sample and sample["input"]:
            sample["instruction"] = f"{sample['instruction']} {sample['input']}"
            del sample["input"]
        if "output" in sample:
            sample["response"] = sample["output"]
            del sample["output"]
        return sample

    def process_data(
        self,
        seed_output_path: str = f"{OUTPUT_PATH}/private_seed.parquet",
        gen_output_path: str = f"{OUTPUT_PATH}/private.parquet",
    ):
        sampled_data = [
            self.reformat_data_sample(sample) for sample in self.load_and_sample_dataset()["train"]
        ]
        df = pd.DataFrame(sampled_data)[:70000]
        df["instruction"] = df["instruction"].apply(
            lambda x: x.removeprefix(
                "If you are a doctor, please answer the medical questions based on "
                "the patient's description."
            )
        )
        df["instruction_keywords"] = df["instruction"].progress_apply(
            self.extractor.extract_keywords
        )
        df["response_keywords"] = df["response"].progress_apply(self.extractor.extract_keywords)
        seed_df = self.process_dataset(df[: self.seed_size].copy())
        gen_df = self.process_dataset(df[self.seed_size :].copy())

        for path in [seed_output_path, gen_output_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        seed_df.to_parquet(seed_output_path, index=False)
        gen_df.to_parquet(gen_output_path, index=False)

    @staticmethod
    def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
        with open(PROMPT_PATH, "r") as f:
            prompt = f.read()
        df["new_instruction"] = df.apply(
            lambda row: prompt.replace(
                "{keywords}",
                str(row["instruction_keywords"]) + ", " + str(row["response_keywords"]),
            ),
            axis=1,
        )
        df["output"] = df["response"]
        df["response"] = df["instruction"]
        df["instruction"] = df["new_instruction"]
        return df[["instruction", "response", "output"]]


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_data()
