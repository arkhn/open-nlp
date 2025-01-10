import os
import random
from typing import List

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from groq import Groq
from quickumls import QuickUMLS
from tqdm import tqdm

SAMPLE_SIZE = 1500
RANDOM_SEED = 42
SEED_SIZE = 500
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.7
MAX_TOKENS = 2048
OUTPUT_PATH = f"datasets/health/model={MODEL_NAME}_t={TEMPERATURE}_size={SAMPLE_SIZE}"
PROMPT_PATH = "datasets/preprocessing/health/prompt.txt"

load_dotenv()


class KeywordExtractor:
    def __init__(self):
        self.matcher = QuickUMLS(quickumls_fp=os.getenv("QUICKUMLS_PATH"))

    def extract_keywords(self, text: str) -> List[str]:
        if pd.isna(text):
            return []
        matches = self.matcher.match(
            text.removeprefix(
                "If you are a doctor, please answer the medical questions based on "
                "the patient's description."
            ).strip(),
            best_match=True,
            ignore_syntax=False,
        )
        return [match[0]["term"] for match in matches]


class DataProcessor:
    def __init__(
        self,
        sample_size: int = SAMPLE_SIZE,
        random_seed: int = RANDOM_SEED,
        seed_size: int = SEED_SIZE,
    ):
        self.sample_size = sample_size
        self.seed_size = seed_size
        random.seed(random_seed)
        self.extractor = KeywordExtractor()

    def load_and_sample_dataset(self) -> List[dict]:
        dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
        return random.sample(list(dataset["train"]), self.sample_size)

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
            self.reformat_data_sample(sample) for sample in self.load_and_sample_dataset()
        ]
        df = pd.DataFrame(sampled_data)
        df["instruction_keywords"] = df["instruction"].apply(self.extractor.extract_keywords)

        seed_df = self.process_dataset(df[: self.seed_size].copy())
        gen_df = self.process_dataset(df[self.seed_size :].copy())

        for path in [seed_output_path, gen_output_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        seed_df.to_parquet(seed_output_path, index=False)
        gen_df.to_parquet(gen_output_path, index=False)

    @staticmethod
    def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
        prompt = open(PROMPT_PATH).read()
        df["new_instruction"] = df["instruction_keywords"].apply(
            lambda x: prompt.replace("{keywords}", ", ".join(x))
        )
        df["output"] = df["response"]
        df["response"] = df["instruction"]
        df["instruction"] = df["new_instruction"]
        return df[["instruction", "response", "output"]]


def generate_public_seed(
    input_path: str = f"{OUTPUT_PATH}/private_seed.parquet",
    output_path: str = f"{OUTPUT_PATH}/public_seed.parquet",
):
    df = pd.read_parquet(input_path)
    client = Groq()

    outputs = []
    for prompt in tqdm(df["instruction"], desc="Generating responses"):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        outputs.append(response.choices[0].message.content)

    pd.DataFrame({"instruction": df["instruction"], "response": outputs}).to_parquet(output_path)


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_data()
    generate_public_seed()
