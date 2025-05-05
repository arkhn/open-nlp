import argparse
import os

import numpy as np
import pandas as pd


def preprocess_parquet(input_file):
    """
    Preprocess the parquet file by separating candidates of the same row into multiple rows.
    Each row will contain a candidate and an ID for the shared example.

    Args:
        input_file (str): Path to the input parquet file

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Load the parquet file
    df = pd.read_parquet(input_file)

    # Get the number of candidates
    num_candidates = sum(1 for col in df.columns if col.startswith("response_"))

    # Create a list to store the expanded rows
    expanded_rows = []

    # Assign a unique ID for each example (row in the original dataframe)
    example_ids = np.arange(len(df))

    # For each row in the original dataframe
    for i, row in df.iterrows():
        example_id = example_ids[i]
        instruction = row["instruction"]

        # For each candidate
        for candidate_idx in range(1, num_candidates + 1):
            # Create a new row
            new_row = {
                "example_id": example_id,
                "instruction": instruction,
                "candidate_id": candidate_idx,
                "response": row[f"response_{candidate_idx}"],
                "similarity_score": row[f"similarity_score_{candidate_idx}"],
                "preference_score": row[f"preference_score_{candidate_idx}"],
                "is_medical": row[f"is_medical_{candidate_idx}"],
                "educational_score": row[f"educational_score_{candidate_idx}"],
                "educational_response": row[f"educational_response_{candidate_idx}"],
                "bleu_score": row[f"bleu_score_{candidate_idx}"],
            }

            expanded_rows.append(new_row)

    # Create a new dataframe from the expanded rows
    return pd.DataFrame(expanded_rows)


def filter_by_medical(df):
    """
    Filter examples where at least one candidate has is_medical=True.
    Returns all candidates for these examples.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    # Group by example_id and check if any candidate has is_medical=True
    medical_examples = df.groupby("example_id")["is_medical"].any()

    # Filter examples where at least one candidate has is_medical=True
    return df[df["example_id"].isin(medical_examples[medical_examples].index)]


def create_sft_dataset(sorted_df, output_prefix, sort_key):
    """
    Create SFT dataset from the top 5000 examples of the sorted dataframe.

    Args:
        sorted_df (pd.DataFrame): Sorted dataframe
        output_prefix (str): Prefix for the output file
        sort_key (str): Key used for sorting

    Returns:
        pd.DataFrame: SFT dataset
    """
    output_file = f"{output_prefix}_sft_{sort_key}.parquet"
    # Dataset 1: SFT - just the head(5000) of the filtered dataset
    sft_df = sorted_df.head(5000)
    sft_df.to_parquet(output_file)
    return sft_df


def create_kto_dataset(sorted_df, output_prefix, sort_key):
    """
    Create KTO dataset from the best 5000 and worst 5000 examples.

    Args:
        sorted_df (pd.DataFrame): Sorted dataframe
        output_prefix (str): Prefix for the output file
        sort_key (str): Key used for sorting

    Returns:
        pd.DataFrame: KTO dataset
    """
    output_file = f"{output_prefix}_kto_{sort_key}.parquet"
    # Dataset 2: KTO - concatenation of the head of 5000 (the best) and the tail of 5000 (the worst)
    best_examples = sorted_df.head(5000).copy()
    worst_examples = sorted_df.tail(5000).copy()

    # Add KTO column: 1 for best, 0 for worst
    best_examples["label"] = True
    worst_examples["label"] = False

    # Concatenate best and worst
    kto_df = pd.concat([best_examples, worst_examples])

    # Keep only needed columns
    kto_df = kto_df[["instruction", "response", "label"]]
    kto_df["completion"] = kto_df["response"]
    kto_df["prompt"] = kto_df["instruction"]
    kto_df.to_parquet(output_file)
    return kto_df


def create_dpo_dataset(sorted_df, output_prefix, sort_key):
    """
    Create DPO dataset from the best 5000 examples and their corresponding worst examples.
    The chosen and rejected examples are paired in the same row.

    Args:
        sorted_df (pd.DataFrame): Sorted dataframe
        output_prefix (str): Prefix for the output file
        sort_key (str): Key used for sorting

    Returns:
        pd.DataFrame: DPO dataset
    """
    output_file = f"{output_prefix}_dpo_{sort_key}.parquet"
    # Dataset 3: DPO - take the 5000 best examples and for each,
    # find the worst matching example with the same id

    best_examples = sorted_df.head(5000).copy()

    # Get unique example_ids from the best examples
    best_example_ids = best_examples["example_id"].unique()

    # Create a list to store paired examples
    paired_examples = []

    # For each best example, find the worst matching example with the same id
    for example_id in best_example_ids:
        # Get all examples with this id
        examples_with_same_id = sorted_df[sorted_df["example_id"] == example_id]

        # The best example is the first one (as the dataframe is sorted)
        best_example = examples_with_same_id.iloc[0]

        # The worst example is the last one
        worst_example = examples_with_same_id.iloc[-1]

        # Create a paired row
        paired_row = {
            "instruction": best_example["instruction"],
            "chosen": best_example["response"],
            "rejected": worst_example["response"],
            "chosen_similarity_score": best_example["similarity_score"],
            "rejected_similarity_score": worst_example["similarity_score"],
            "chosen_preference_score": best_example["preference_score"],
            "rejected_preference_score": worst_example["preference_score"],
            "chosen_educational_score": best_example["educational_score"],
            "rejected_educational_score": worst_example["educational_score"],
            "chosen_bleu_score": best_example["bleu_score"],
            "rejected_bleu_score": worst_example["bleu_score"],
            "chosen_is_medical": best_example["is_medical"],
            "rejected_is_medical": worst_example["is_medical"],
        }
        paired_examples.append(paired_row)

    # Create DataFrame from the paired examples
    dpo_df = pd.DataFrame(paired_examples)

    dpo_df.to_parquet(output_file)
    return dpo_df


def create_eval_dataset(sorted_df, output_prefix):
    """
    Create SFT dataset from the top 5000 examples of the sorted dataframe.

    Args:
        sorted_df (pd.DataFrame): Sorted dataframe
        output_prefix (str): Prefix for the output file
        sort_key (str): Key used for sorting

    Returns:
        pd.DataFrame: SFT dataset
    """
    # Dataset 1: SFT - just the head(5000) of the filtered dataset
    sft_df = sorted_df.head(5000)
    sft_df.to_parquet(output_prefix)
    return sft_df


def process_data(input_file):
    """
    Process the input file and create all datasets with different sorting methods

    Args:
        input_file (str): Path to the input parquet file
    """
    # Extract output prefix from input file (remove extension)
    output_prefix = os.path.splitext(input_file)[0]

    print(f"Preprocessing data from {input_file}...")
    # Preprocess: separate candidates into different rows
    preprocessed_df = preprocess_parquet(input_file)

    print("Filtering by medical...")
    # Filter by is_medical=True for at least one candidate
    filtered_df = filter_by_medical(preprocessed_df)
    filtered_df = filtered_df[filtered_df["educational_score"] < 6]

    # Define sorting methods
    sorting_methods = {
        "mes": ["is_medical", "educational_score", "similarity_score"],
        "ms": ["is_medical", "similarity_score"],
        "s": ["similarity_score"],
        "e": ["educational_score"],
        "se": ["similarity_score", "educational_score"],
    }

    # Process each sorting method
    for sort_key, sort_columns in sorting_methods.items():
        print(f"Processing sort method: {sort_key} ({', '.join(sort_columns)})")
        # Sort the dataframe
        sorted_df = filtered_df.sort_values(by=sort_columns, ascending=False)

        # Create output file
        filtered_output_file = f"{output_prefix}_{sort_key}.parquet"
        print(f"Saving filtered results to {filtered_output_file}...")
        sorted_df.to_parquet(filtered_output_file)

        # Create datasets
        print(f"Creating SFT dataset with {sort_key} sorting...")
        create_sft_dataset(sorted_df, output_prefix, sort_key)

        print(f"Creating KTO dataset with {sort_key} sorting...")
        create_kto_dataset(sorted_df, output_prefix, sort_key)

        print(f"Creating DPO dataset with {sort_key} sorting...")
        create_dpo_dataset(sorted_df, output_prefix, sort_key)

    eval_sorted_df = filtered_df.sort_values(by=["is_medical"], ascending=False)
    create_eval_dataset(eval_sorted_df, f"{os.path.splitext(input_file)[0]}_eval.parquet")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Preprocess and filter parquet data for different dataset types."
    )
    parser.add_argument("input_file", help="Path to the input parquet file")

    # Parse arguments
    args = parser.parse_args()

    # Process the data
    process_data(args.input_file)

    print("Done!")
