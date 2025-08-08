import argparse
import os
import random

import pandas as pd
from vllm import LLM, SamplingParams


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
    parser.add_argument(
        "--few_shot",
        action="store_true",
        help="Enable few-shot mode with examples from dataset",
    )
    return parser.parse_args()


def create_few_shot_examples(dataset_df):
    """Sample 50 examples from dataset and create 10 different combinations"""
    # Step 1: Sample 50 examples from dataset
    sampled_examples = dataset_df.sample(n=150, random_state=42)

    # Step 2: Create 10 different combinations (3 examples each)
    combinations = []
    examples_list = sampled_examples.to_dict("records")

    for _ in range(100):
        combo = random.sample(examples_list, 3)
        combinations.append(combo)

    return combinations


def add_few_shot_to_prompt(original_prompt, few_shot_examples):
    """Add few-shot examples to prompt using self-instruct format"""
    few_shot_text = "Here are some examples:\n\n"

    # Add each example
    for example in few_shot_examples:
        few_shot_text += f"Instruction: {example['instruction']}\n"
        few_shot_text += f"Output: {example['response']}\n\n"

    # Add current instruction
    few_shot_text += f"Instruction: {original_prompt}\n"
    few_shot_text += "Output:"

    return few_shot_text


def generate_responses(model, prompts, num_sequences):
    all_responses = []

    for _ in range(num_sequences):
        # Randomly select parameters within a desired range
        temperature = random.uniform(0.7, 0.9)
        top_p = random.uniform(0.8, 0.95)
        top_k = random.choice([40, 50, 60])

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=2048,
            seed=random.randint(0, 2**32 - 1),
            stop=["</s>"],
            presence_penalty=1.0,
            frequency_penalty=1.2,
            n=1,
        )

        response = model.generate(prompts, sampling_params=sampling_params)
        all_responses.append([output.outputs[0].text for output in response])

    return all_responses


def main():
    args = parse_arguments()

    # Load dataset
    df = pd.read_parquet(args.dataset)
    prompts = df["instruction"]

    # Handle few-shot injection if enabled
    if args.few_shot:
        few_shot_combinations = create_few_shot_examples(df)
        enhanced_prompts = []
        for prompt in prompts:
            # Pick random combination for this prompt
            combo = random.choice(few_shot_combinations)
            enhanced_prompt = add_few_shot_to_prompt(prompt, combo)
            enhanced_prompts.append(enhanced_prompt)
        prompts = enhanced_prompts

    # Initialize the LLM
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
    )

    # Generate responses
    responses = generate_responses(llm, prompts, num_sequences=args.num_sequences)

    # Create output dataframe
    output_data = {"instruction": prompts}
    for i in range(args.num_sequences - 1):
        output_data[f"response_{i+1}"] = responses[i]

    # Save results
    df_output = pd.DataFrame(output_data)
    os.makedirs(f"{args.output_path}", exist_ok=True)
    output_file = os.path.join(args.output_path, "public_generated.parquet")
    df_output.to_parquet(output_file)


if __name__ == "__main__":
    main()
