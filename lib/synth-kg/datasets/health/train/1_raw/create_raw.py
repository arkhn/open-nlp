import random

import pandas as pd
from datasets import load_dataset


def main():
    random.seed(42)

    # Load the dataset
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")

    sampled_data = random.sample(list(dataset["train"]), 1500)

    # Reformat data sample
    for sample in sampled_data:
        if isinstance(sample, dict) and "input" in sample:
            if sample.get("input"):
                sample["instruction"] = f"{sample.get('instruction', '')} {sample.get('input', '')}"
            if "input" in sample:
                del sample["input"]

        if isinstance(sample, dict) and "output" in sample:
            sample["response"] = sample["output"]
            del sample["output"]

    df = pd.DataFrame(sampled_data)
    df.to_parquet("datasets/health/train/1_raw/raw_data.parquet", index=False)


if __name__ == "__main__":
    main()
