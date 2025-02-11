import argparse
import os

import pandas as pd


def join_datasets(
    evaluation: str, private_seed_path: str, join_col: str = "instruction"
) -> pd.DataFrame:
    """
    Join two parquet files using a specified column

    Args:
        evaluation: Path to first parquet file
        private_seed_path: Path to second parquet file
        join_col: Column name to join on

    Returns:
        Joined DataFrame
    """
    # Read parquet files
    df1 = pd.read_parquet(evaluation)
    df2 = pd.read_parquet(private_seed_path)[["instruction", "output"]]

    # Perform inner join
    joined_df = df1.merge(df2, on=join_col, how="inner")
    joined_df = joined_df[["response", "output"]]
    joined_df.rename(columns={"response": "instruction"}, inplace=True)
    joined_df.rename(columns={"output": "response"}, inplace=True)
    return joined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join two parquet files on a specified column")
    parser.add_argument("--evaluation", type=str, help="Path to first parquet file")
    parser.add_argument("--private_seed", type=str, help="Path to second parquet file")
    parser.add_argument("--output_path", type=str, help="Path to output file")
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    output_path = args.output_path
    result = join_datasets(args.evaluation, args.private_seed)
    result.to_parquet(os.path.join(output_path, "evaluation_sft.parquet"))
