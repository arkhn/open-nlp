import pandas as pd
from datasets import load_dataset
from huggingface_hub import notebook_login

notebook_login()


def _preprocess(text: str) -> str:
    text = text.split("\n")[-1].lower()
    return text


def _resample(df: pd.DataFrame, n_sample: int, n_section: int) -> pd.DataFrame:
    patterns = "allergies|history of present illness|past medical history|\
                discharge medications|social history|medications on admission"
    df["section_title"] = df["section_title"].apply(_preprocess)
    df = df[df.section_title.str.contains(patterns)]
    df = df.groupby("section_title").filter(lambda x: len(x) > n_sample)
    df = df.groupby("document_id").filter(lambda x: len(x) == n_section)
    df.rename(columns={"section_content": "summary"}, inplace=True)
    return df


def save_data_sample(input_path: str, split: str, output_path: str):
    df = load_dataset(input_path, split=split).to_pandas()
    df = _resample(df, 2, 6)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    save_data_sample("bio-datasets/mimoracle", "train", "./data/mimoracle_train.csv")
    save_data_sample("bio-datasets/mimoracle", "test", "./data/mimoracle_test.csv")
