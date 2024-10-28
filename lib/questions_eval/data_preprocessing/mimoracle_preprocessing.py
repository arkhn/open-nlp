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
    df["section_title"] = [_preprocess(x) for x in df["section_title"]]
    df = df[df.section_title.str.contains(patterns)]
    df = df.groupby("section_title").filter(lambda x: len(x) > n_sample)
    df = df.groupby("document_id").filter(lambda x: len(x) == n_section)
    return df


if __name__ == "__main__":
    df = load_dataset("bio-datasets/mimoracle", split="train").to_pandas()
    sample_df = _resample(df, 2, 6)
    sample_df.to_csv("mimoracle_sample.csv", index=False)
