import os
import random
from typing import List

import nltk
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from groq import Groq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from openai import OpenAI
from tqdm import tqdm

SAMPLE_SIZE = 1500
RANDOM_SEED = 42
SEED_SIZE = 500
MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0.7
MAX_TOKENS = 2048
OUTPUT_PATH = f"datasets/finance/model={MODEL_NAME}_t={TEMPERATURE}_size={SAMPLE_SIZE}"
PROMPT_PATH = "datasets/preprocessing/finance/prompt.txt"


load_dotenv()
client = OpenAI()


class KeywordExtractor:
    def __init__(self, lexicon_path: str = "datasets/preprocessing/finance/lexicons.csv"):
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english"))
        self.lm_lexicon = set(pd.read_csv(lexicon_path)["Word"].str.lower())

    @staticmethod
    def clean_text(text: str) -> str:
        if pd.isna(text):
            return ""
        return text.split("}.")[1] if "}." in text else text

    def extract_keywords(self, text: str) -> List[str]:
        if pd.isna(text):
            return []
        cleaned_text = self.clean_text(text)
        tokens = word_tokenize(cleaned_text.lower())
        filtered_tokens = [
            word for word in tokens if word.isalpha() and word not in self.stop_words
        ]
        return [word for word in filtered_tokens if word.lower() in self.lm_lexicon]


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
        dataset = load_dataset("FinGPT/fingpt-sentiment-train")

        # Get dataset samples
        samples = random.sample(list(dataset["train"]), self.sample_size)
        for sample in samples:
            prompt = (
                f"Replace all person names in this text with unique fictional names, "
                f"maintaining consistency. "
                f"Return the modified text and only this. Text: {sample['input']} Answer:"
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                stop=None,
            )

            # Update the instruction with anonymized text
            sample["input"] = response.choices[0].message.content.strip()
        return samples

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
        df["instruction"] = df["instruction"].apply(self.extractor.clean_text)
        df.response = df.response.apply(lambda x: x.split(" ")[1] if len(x.split(" ")) > 1 else x)

        seed_df = self.process_dataset(df[: self.seed_size].copy())
        gen_df = self.process_dataset(df[self.seed_size :].copy())

        for path in [seed_output_path, gen_output_path, "datasets/finance/eval/"]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        seed_df.to_parquet(seed_output_path, index=False)
        gen_df.to_parquet(gen_output_path, index=False)

    @staticmethod
    def process_dataset(df: pd.DataFrame) -> pd.DataFrame:
        prompt = open(PROMPT_PATH).read()
        prompts = df["instruction_keywords"].apply(
            lambda x: prompt.replace("{keywords}", ", ".join(x))
        )
        df["new_instruction"] = [
            prompt.replace("{sentiment}", response)
            for prompt, response in zip(prompts, df["response"])
        ]
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
