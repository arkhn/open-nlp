import argparse
import os

import pandas as pd
from vllm import LLM, SamplingParams


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output parquet file"
    )
    parser.add_argument("--model", type=str, required=True, help="Name or path of the model to use")
    parser.add_argument("--tp", type=int, default=1, help="Number of GPUs to use for inference")
    return parser.parse_args()


def generate_response(model, prompts, sampling_params):
    response = model.generate(prompts, sampling_params=sampling_params)
    return [output.outputs[0].text for output in response]


def main():
    args = parse_arguments()

    df = pd.read_json(
        "datasets/health/eval/reference_outputs/claude-2/iCliniq_output.jsonl", lines=True
    )
    # Extract the specific column
    prompts = df["prompt"].apply(
        lambda x: x.split(
            "Human: If you are a doctor, please answer the medical questions based on the patient's description."
        )[1]
        .split("Assistant")[0]
        .strip()
    )
    prompts = prompts.apply(
        lambda x: (
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n ###Instruction:\n {x}\n\n ###Response:\n"
        )
    )
    # Initialize the LLM with your chosen model
    llm = LLM(model=args.model, tensor_parallel_size=args.tp)

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        stop=["Human:", "Patient:", "User:", "ChatDoctor", "Assistant", "Answer", "</s>", "#"],
    )

    # Generate response per prompt
    responses = generate_response(llm, prompts, sampling_params)

    # Create output dataframe
    output_data = {"instruction": prompts, "response": responses}

    # Create output dataframe and save
    df_output = pd.DataFrame(output_data)
    os.makedirs(f"{args.output_path}", exist_ok=True)
    output_file = os.path.join(args.output_path, "evaluation.parquet")
    df_output.to_parquet(output_file)


if __name__ == "__main__":
    main()
