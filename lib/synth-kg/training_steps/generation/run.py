import os

import pandas as pd
from vllm import LLM, SamplingParams
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output parquet file"
    )
    parser.add_argument("--model", type=str, required=True, help="Name or path of the model to use")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to input dataset parquet file"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=5,
        help="Number of different sequences to generate per prompt",
    )
    return parser.parse_args()


def generate_responses(model, prompts, sampling_params, num_sequences):
    response = model.generate([prompts], sampling_params=sampling_params)
    all_responses = []
    for _ in range(num_sequences - 1):
        all_responses.append([output.outputs[0].text for output in response])
    return all_responses


def main():
    args = parse_arguments()

    # we use a contextual path only because we deploy the script via skypilot
    df = pd.read_parquet(args.dataset)
    # Extract the specific column
    prompts = df["instruction"]

    # Initialize the LLM with your chosen model
    llm = LLM(model=args.model)

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
        stop=["\n"],
        n=args.num_sequences,
    )

    # Generate multiple responses per prompt
    responses = generate_responses(llm, prompts, sampling_params, num_sequences=args.num_sequences)

    # Create output dataframe with multiple response columns
    output_data = {"instruction": prompts}
    for i in range(args.num_sequences):
        output_data[f"response_{i+1}"] = responses[i]

    # Create output dataframe and save
    df_output = pd.DataFrame(output_data)
    os.makedirs(f"{args.output_path}", exist_ok=True)
    output_file = os.path.join(args.output_path, "public_generated_data.parquet")
    df_output.to_parquet(output_file)


if __name__ == "__main__":
    main()
