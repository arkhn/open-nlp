import argparse
import os

import pandas as pd
from generate import generate_response
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output parquet file"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to input dataset parquet file"
    )
    parser.add_argument("--tp", type=int, required=True, help="Number of GPUs")
    return parser.parse_args()


def main():
    args = parse_arguments()
    df = pd.read_parquet(args.dataset)
    # Extract the specific column
    prompts = df["response"].tolist()
    instructions = [
        (
            f"Human: If you are a doctor, please answer the medical questions based "
            f"on the patient's description.\n\n"
            f"{prompt[:1500]}\n"
            f"Assistant:"
        )
        for prompt in prompts
    ]
    prompts = [(f"Patient: {prompt[:1500]}\n" f"ChatDoctor:") for prompt in prompts]

    # Initialize the LLM with your chosen model
    hf_model = AutoModelForCausalLM.from_pretrained("xz97/AlpaCare-llama2-13b")
    hf_tokenizer = AutoTokenizer.from_pretrained("xz97/AlpaCare-llama2-13b")
    hf_model.save_pretrained("model/alpacare")
    hf_tokenizer.save_pretrained("model/alpacare")
    llm = LLM(
        model="./model/alpacare",
        tensor_parallel_size=args.tp,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
        truncate_prompt_tokens=1024,
        stop=["Human:", "Patient:", "User:", "ChatDoctor", "Assistant", "Answer", "</s>"],
    )
    # Generate response per prompt
    responses = generate_response(llm, prompts, sampling_params)

    # Create output dataframe
    output_data = {"instruction": instructions, "response": responses}

    # Create output dataframe and save
    df_output = pd.DataFrame(output_data)
    os.makedirs(f"{args.output_path}", exist_ok=True)
    output_file = os.path.join(args.output_path, "evaluation_alpacare_sft.parquet")
    df_output.to_parquet(output_file)
    df_output.to_parquet("./evaluation_alpacare_sft.parquet")


if __name__ == "__main__":
    main()
