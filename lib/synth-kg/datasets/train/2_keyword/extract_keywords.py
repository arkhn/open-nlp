import os
from typing import List

import pandas as pd
from dotenv import load_dotenv
from quickumls import QuickUMLS

load_dotenv()


class KeywordExtractor:
    def __init__(self):
        load_dotenv()
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


def run(input_path: str, output_path: str) -> None:
    # Read the parquet file
    extractor = KeywordExtractor()
    df = pd.read_parquet(input_path)

    # Apply keyword extraction to create new column
    df["instruction_keywords"] = df["instruction"].apply(extractor.extract_keywords)
    df["response_keywords"] = df["response"].apply(extractor.extract_keywords)

    # Select final columns
    print(df.head(10))

    # Save the updated dataframe
    df.to_parquet(output_path)


if __name__ == "__main__":
    run(
        input_path="datasets/train/1_raw/raw_data.parquet",
        output_path="datasets/train/2_keyword/keyword_data.parquet",
    )
