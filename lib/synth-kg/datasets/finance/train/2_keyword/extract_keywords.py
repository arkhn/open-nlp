import os
from typing import List

import pandas as pd
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

load_dotenv()


class KeywordExtractor:
    def __init__(self):
        load_dotenv()
        nltk.download("punkt")
        nltk.download("stopwords")
        self.stop_words = set(stopwords.words("english"))
        self.lm_lexicon = set(pd.read_csv("datasets/finance/train/2_keyword/lexicons.csv")["Word"].str.lower())

    def extract_keywords(self, text: str) -> List[str]:
        if pd.isna(text):
            return []

        # Preprocess the text
        cleaned_text = text.split("}.")[1]
        tokens = word_tokenize(cleaned_text.lower())
        filtered_tokens = [word for word in tokens if word.isalpha()]

        # Remove stop words
        filtered_tokens = [word for word in filtered_tokens if word not in self.stop_words]

        # Extract keywords using the LM lexicon
        keywords = [
            word for word in filtered_tokens if word.lower() in self.lm_lexicon
        ]

        return keywords


def run(input_path: str, output_path: str) -> None:
    # Read the parquet file
    extractor = KeywordExtractor()
    df = pd.read_parquet(input_path)

    # Apply keyword extraction to create new column
    df["instruction_keywords"] = df["instruction"].apply(extractor.extract_keywords)

    # Select final columns
    print(df["instruction_keywords"][10])

    # Save the updated dataframe
    df.to_parquet(output_path)


if __name__ == "__main__":
    run(
        input_path="datasets/finance/train/1_raw/raw_data.parquet",
        output_path="datasets/finance/train/2_keyword/keyword_data.parquet",
    )
