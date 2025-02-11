import json
import os

import mauve
import pandas as pd

# Path to the gold standard file
private_path = (
    "datasets/health/model=xz97-AlpaCare-llama2-13b_t=0.7_size=1500-knowledge/private.parquet"
)

# Load the gold standard data
private_df = pd.read_parquet(private_path)
private_responses = private_df["response"]

# Directory containing the dataset
dataset_dir = "datasets/health"

# Dictionary to store results
mauve_scores = {}


# Function to compute MAUVE score
def compute_mauve_score(gold_texts, generated_texts):
    result = mauve.compute_mauve(p_text=gold_texts, q_text=generated_texts, device_id=0)
    return result.mauve


# Traverse the dataset directory
for root, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith("_dpo.parquet"):
            file_path = os.path.join(root, file)
            df = pd.read_parquet(file_path)
            if "chosen" in df.columns:
                chosen_texts = df["chosen"].dropna().astype(str).tolist()
                mauve_score = compute_mauve_score(private_responses, chosen_texts)
                mauve_scores[file_path] = mauve_score
                print(f"MAUVE score for {file_path}: {mauve_score:.4f}")
            else:
                print(f'Column "chosen" not found in {file_path}')

print(json.dumps(mauve_scores, indent=4))
