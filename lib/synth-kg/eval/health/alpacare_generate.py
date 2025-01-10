import os

import pandas as pd
from vllm import LLM, SamplingParams
import argparse
from generate import generate_response


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output parquet file"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to input dataset parquet file"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    df = pd.read_parquet(args.dataset)
    # Extract the specific column
    prompts = df["response"]

    # Initialize the LLM with your chosen model
    llm = LLM(model="xz97/AlpaCare-llama2-13b")

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
        stop=["\n"],
    )

    # Generate response per prompt
    responses = generate_response(llm, prompts, sampling_params)

    # Create output dataframe
    output_data = {"instruction": prompts, "response": responses}

    # Create output dataframe and save
    df_output = pd.DataFrame(output_data)
    os.makedirs(f"{args.output_path}", exist_ok=True)
    output_file = os.path.join(args.output_path, "alpacare_evaluation_sft.parquet")
    df_output.to_parquet(output_file)


if __name__ == "__main__":
    main()
