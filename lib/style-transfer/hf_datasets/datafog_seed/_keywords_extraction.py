import os
from pathlib import Path

from datasets import load_dataset
from quickumls import QuickUMLS
from tqdm import tqdm

tqdm.pandas()
matcher = QuickUMLS(os.getenv("QUICKUMLS_PATH"))


def main():
    # Load the dataset
    dataset = load_dataset("DataFog/medical-transcription-instruct")
    df = dataset["train"].to_pandas().iloc[:3000]

    # Function to extract keywords using QuickUMLS
    def extract_keywords(text):
        extracted_terms = matcher.match(text, best_match=True, ignore_syntax=False)
        keywords = [
            sorted(terms, key=lambda x: x["end"], reverse=True)[0] for terms in extracted_terms
        ]
        keywords = sorted(keywords, key=lambda x: x["start"])
        keywords = [term["ngram"] for term in keywords]
        return ", ".join(keywords)

    # Process the dataset
    df["keywords"] = df["transcription"].progress_apply(extract_keywords)
    df["id"] = df.index
    df = df[["id", "transcription", "keywords"]].rename(columns={"transcription": "text"})
    # Shuffle the DataFrame
    df = df.sample(frac=1)

    # Save to parquet
    output_path = Path("data/train.parquet")
    df.to_parquet(output_path)

    print(f"Processed dataset saved to {output_path}")


if __name__ == "__main__":
    main()
