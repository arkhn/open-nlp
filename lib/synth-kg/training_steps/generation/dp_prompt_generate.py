import argparse
import torch
import os
import uuid
from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
from vllm import LLM, SamplingParams


def prompt_template_fn(private_doc):
    return f"Document : {private_doc}\nParaphrase of the document :"


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
        "--tp",
        type=int,
        default=1,
        help="Number of GPUs to use for inference",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="Number of node to use for inference",
    )
    return parser.parse_args()


def clipped_logits_processor(token_ids, logits):
    return torch.clamp(logits, -5, 5)


def generate_responses(model, prompts):

    all_responses = []
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=2048,
        stop=["</s>"],
        n=1,
        logits_processors=[clipped_logits_processor],
    )
    response = model.generate(
        [prompt_template_fn(prompt) for prompt in prompts], sampling_params=sampling_params
    )
    all_responses.append([output.outputs[0].text for output in response])

    return all_responses


def main():
    args = parse_arguments()

    # we use a contextual path only because we deploy the script via skypilot
    df = pd.read_parquet(args.dataset)
    # Extract the specific column
    prompts = df["response"][:10]

    # Initialize the LLM with your chosen model
    hf_model = AutoModelForCausalLM.from_pretrained(args.model)
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
    alpacare_path = f"model/alpacare-{str(uuid.uuid4())[:7]}"
    hf_model.save_pretrained(alpacare_path)
    hf_tokenizer.save_pretrained(alpacare_path)
    llm = LLM(
        model=alpacare_path,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
    )
    # Generate multiple responses per prompt
    responses = generate_responses(llm, prompts)

    # Create output dataframe with multiple response columns
    output_data = {"instruction": prompts, "response": responses}

    # Create output dataframe and save
    df_output = pd.DataFrame(output_data)
    os.makedirs(f"{args.output_path}", exist_ok=True)
    output_file = os.path.join(args.output_path, "public_generated.parquet")
    df_output.to_parquet(output_file)


if __name__ == "__main__":
    main()
