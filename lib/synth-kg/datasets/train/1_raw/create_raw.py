import random

import pandas as pd
from datasets import load_dataset


def main():
    # Set random seed for reproducibility
    random.seed(42)

    # Load the dataset
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")

    sampled_data = random.sample(list(dataset["train"]), 1500)

    # Reformat data sample
    for sample in sampled_data:
        if "input" in sample:
            if sample["input"]:
                sample["instruction"] = f"{sample['instruction']} {sample['input']}"
            del sample["input"]

        if "output" in sample:
            sample["response"] = sample["output"]
            del sample["output"]

    df = pd.DataFrame(sampled_data)
    df.to_parquet("datasets/train/1_raw/raw_data.parquet", index=False)


if __name__ == "__main__":
    main()
