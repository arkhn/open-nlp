import argparse
import gc
import re
from typing import Any, List, Tuple

import nltk
import pandas as pd
import torch
import wandb
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_datasets(
    private_dataset_path: str, public_dataset_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load private and public datasets from parquet files.

    Args:
        private_dataset_path: Path to the private dataset parquet file
        public_dataset_path: Directory containing the public_generated.parquet file

    Returns:
        Tuple containing private and public datasets as pandas DataFrames
    """
    private_dataset = pd.read_parquet(private_dataset_path)
    public_dataset = pd.read_parquet(f"{public_dataset_path}/public_generated.parquet")
    return private_dataset, public_dataset


def preprocess_text(text: Any) -> Any:
    """
    Preprocess text by splitting at '###' if present.

    Args:
        text: Input text to preprocess

    Returns:
        Processed text
    """
    if isinstance(text, str) and "###" in text:
        return text.split("###")[0].strip()
    return text


def compute_similarities(model_name: str, texts1: List[str], texts2: List[str]) -> List[float]:
    """
    Compute similarity scores between two sets of texts using a combined approach.

    Args:
        model: SentenceTransformer model
        texts1: First list of texts
        texts2: Second list of texts

    Returns:
        List of similarity scores
    """

    model = SentenceTransformer(model_name)
    # Standard global approach
    embeddings1 = model.encode(texts1, convert_to_tensor=True)
    embeddings2 = model.encode(texts2, convert_to_tensor=True)
    global_scores = cos_sim(embeddings1, embeddings2).diagonal().tolist()

    # Initialize tokenizer using the same model
    tokenizer = AutoTokenizer.from_pretrained(model._first_module().auto_model.config._name_or_path)

    # Sliding window approach for longer texts
    window_size = 128
    window_scores = []

    for i, text2 in enumerate(texts2):
        # Always apply sliding window regardless of text length
        if isinstance(text2, str):
            # Tokenize the text
            tokens = tokenizer.encode(text2, add_special_tokens=False)

            # Create chunks with sliding window
            chunks = []
            for j in range(0, max(1, len(tokens) - window_size + 1), window_size // 2):
                end_idx = min(j + window_size, len(tokens))
                chunk_tokens = tokens[j:end_idx]
                chunk_text = tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)

            if chunks:
                # Get embeddings for each chunk
                chunk_embeddings = model.encode(chunks, convert_to_tensor=True)
                text1_embedding = embeddings1[i].unsqueeze(0)

                # Calculate similarity for each chunk
                chunk_scores = cos_sim(text1_embedding, chunk_embeddings).squeeze(0).tolist()

                # Combine mean + min for better sensitivity to bad chunks
                min_score = min(chunk_scores)
                window_scores.append(min_score)
            else:
                window_scores.append(global_scores[i])
        else:
            window_scores.append(global_scores[i])

    # Combine both approaches
    final_scores = [
        (global_score + window_score) / 2
        for global_score, window_score in zip(global_scores, window_scores)
    ]
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return final_scores


def initialize_wandb(args: argparse.Namespace) -> Any:
    """
    Initialize wandb for experiment tracking.

    Args:
        args: Command line arguments

    Returns:
        Initialized wandb run
    """
    return wandb.init(
        project="synth-kg",
        name=f"eval-{args.wdb_id}",
        group=args.group_id,
        job_type="evaluation",
        reinit=True,
    )


def calculate_statistics(public_dataset: pd.DataFrame, n: int, run: Any) -> None:
    """
    Calculate and log statistics of similarity scores.

    Args:
        public_dataset: DataFrame containing scored responses
        n: Number of response columns
        run: Active wandb run
    """
    score_columns = [f"similarity_score_{i}" for i in range(1, n + 1)]
    all_scores = public_dataset[score_columns].values.flatten()
    mean_score = all_scores.mean()
    max_score = all_scores.max()
    min_score = all_scores.min()
    median_score = pd.Series(all_scores).median()

    run.log(
        {
            "score/mean": mean_score,
            "score/max": max_score,
            "score/min": min_score,
            "score/median": median_score,
        }
    )
    print(f"Mean: {mean_score:.4f}")
    print(f"Max: {max_score:.4f}")
    print(f"Min: {min_score:.4f}")
    print(f"Median: {median_score:.4f}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Score generated responses using a sentence transformer model"
    )
    parser.add_argument("--wdb_id", type=str, help="Optional wandb run ID to resume a run")
    parser.add_argument("--group_id", type=str, help="Optional group ID to gather a run")
    parser.add_argument(
        "--sts_model", type=str, required=True, help="Path to sentence transformer model"
    )
    parser.add_argument(
        "--private_dataset", type=str, required=True, help="Path to private dataset parquet"
    )
    parser.add_argument(
        "--public_dataset", type=str, required=True, help="Path to directory with public dataset"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output files")
    parser.add_argument(
        "--n", type=int, required=True, help="Number of text columns in public dataset"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Number of GPUs to use for inference",
    )
    return parser.parse_args()


def filter_medical_content(llm, instructions: List[str]) -> List[bool]:
    """
    Filter instructions to identify those related to medicine using AlpaCare LLama model.

    Args:
        instructions: List of instruction texts to analyze

    Returns:
        List of boolean values indicating whether each instruction involves medicine
    """
    # Initialize vllm with AlpaCare model
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    # Prepare prompts with the medical filter question
    prompts = [
        (
            f"If you are a doctor, "
            f"please answer the medical questions based  on the patient’s description."
            f" Patient: {instruction}  Does my instruction involves medicine?"
        )
        for instruction in instructions
    ]

    # Process all prompts with vllm
    outputs = llm.generate(prompts, sampling_params)

    # Extract results and determine if medicine-related
    results = []
    for output in outputs:
        response = output.outputs[0].text.strip().lower()
        # Check if response contains "yes"
        is_medical = ("y" or "t") in response
        results.append(is_medical)

    return results


def compute_educational_score(llm, responses: List[str]) -> dict:
    """
    Compute educational scores for responses using AlpaCare LLama model.

    Args:
        responses: List of response texts to analyze
        tp: Number of GPUs to use for inference

    Returns:
        Dictionary containing two lists: educational_scores and educational_responses
    """
    # Read the educational scoring prompt from file
    prompt_path = "training_steps/score/prompt_educational_scores.txt"
    with open(prompt_path, "r") as f:
        prompt_template = f.read().strip()

    # Initialize vllm with AlpaCare model
    sampling_params = SamplingParams(temperature=0.5, max_tokens=512)

    # Prepare prompts with the educational scoring prompt
    prompts = [prompt_template.replace("{INSTRUCTION}", response) for response in responses]
    # Process all prompts with vllm
    outputs = llm.generate(prompts, sampling_params)

    # Extract results - both the overall score and full response
    scores = []
    full_responses = []

    for output in outputs:
        full_response = output.outputs[0].text.strip()
        full_responses.append(full_response)

        # Extract overall_score using regex
        overall_index = full_response.lower().find("overall")
        score_match = re.search(r"\d{1}", full_response[overall_index:], re.IGNORECASE)
        if score_match:
            score = float(score_match.group(0))
        else:
            score = 0.0  # Default score if no match found

        scores.append(score)

    return {"educational_scores": scores, "educational_responses": full_responses}


def compute_preference_score(
    llm, private_responses: List[str], public_responses: List[str], instructions: List[str]
) -> List[float]:
    """
    Compute preference scores between public and private responses using LLama model.

    The algorithm:
    1. Generate responses with public at position 1, private at position 2
    2. Generate responses with positions swapped
    3. Compute win rate, incrementing score when public output is preferred or there's a tie

    Args:
        private_responses: List of responses from private dataset
        public_responses: List of responses from public dataset
        instructions: List of instruction prompts
        tp: Number of GPUs to use for inference

    Returns:
        List of preference scores (between 0 and 1) for each pair of responses
    """
    # Read the preference scoring prompt from file
    prompt_path = "training_steps/score/prompt_preferences.txt"
    with open(prompt_path, "r") as f:
        prompt_template = f.read().strip()

    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)

    # Prepare both sets of prompts (swapping positions)
    # First set: public (a), private (b)
    prompts_1 = [
        prompt_template.format(
            instruction=instruction, output_1=public_response, output_2=private_response
        )
        for instruction, public_response, private_response in zip(
            instructions, public_responses, private_responses
        )
    ]

    # Second set: private (a), public (b)
    prompts_2 = [
        prompt_template.format(
            instruction=instruction, output_1=private_response, output_2=public_response
        )
        for instruction, public_response, private_response in zip(
            instructions, public_responses, private_responses
        )
    ]

    # Generate results for first set
    outputs_1 = llm.generate(prompts_1, sampling_params)

    # Generate results for second set
    outputs_2 = llm.generate(prompts_2, sampling_params)

    # Compute preference scores
    preference_scores = []

    # Process results from first set (public=a, private=b)
    for output in outputs_1:
        response = output.outputs[0].text.strip().lower()
        # Score 1 if output (a) (public) is preferred or if there's a tie
        if "a" in response or "tie" in response:
            score_1 = 1.0
        else:
            score_1 = 0.0
        preference_scores.append(score_1)

    # Process results from second set (private=a, public=b)
    for i, output in enumerate(outputs_2):
        response = output.outputs[0].text.strip().lower()
        # Score 1 if output (b) (public) is preferred or if there's a tie
        if "b" in response or "tie" in response:
            score_2 = 1.0
        else:
            score_2 = 0.0

        # Average the scores from both prompts
        preference_scores[i] = (preference_scores[i] + score_2) / 2.0

    return preference_scores


def compute_bleu_score(references: List[str], candidates: List[str]) -> List[float]:
    """
    Compute BLEU scores between reference texts and candidate texts.

    Args:
        references: List of reference texts (ground truth)
        candidates: List of candidate texts to evaluate

    Returns:
        List of BLEU scores between 0 and 1
    """
    smoothie = SmoothingFunction().method1

    bleu_scores = []
    for ref, cand in zip(references, candidates):
        if not isinstance(ref, str) or not isinstance(cand, str):
            bleu_scores.append(0.0)
            continue

        # Tokenize sentences
        ref_tokens = nltk.word_tokenize(ref.lower())
        cand_tokens = nltk.word_tokenize(cand.lower())

        if len(cand_tokens) == 0:
            bleu_scores.append(0.0)
            continue

        # Calculate BLEU score with smoothing
        bleu = sentence_bleu(
            [ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie
        )
        bleu_scores.append(bleu)

    return bleu_scores


def truncate_responses(responses, tokenizer) -> list[str]:
    max_length = 512

    # Truncate private dataset responses
    truncated_responses = []
    for response in responses:
        tokens = tokenizer.encode(response, add_special_tokens=False)
        if len(tokens) > max_length:
            truncated_tokens = tokens[:max_length]
            truncated_response = tokenizer.decode(truncated_tokens)
            truncated_responses.append(truncated_response)
        else:
            truncated_responses.append(response)
    return truncated_responses


def score_dataset(n, sts_model, private_dataset, public_dataset, tp):
    # Initialize tokenizer for truncation
    tokenizer = AutoTokenizer.from_pretrained("xz97/AlpaCare-llama2-13b")
    private_responses = truncate_responses(private_dataset["response"].tolist(), tokenizer)
    llm = LLM(model="xz97/AlpaCare-llama2-13b", tensor_parallel_size=tp, pipeline_parallel_size=1)
    for i in range(1, n + 1):
        # Truncate public dataset responses
        public_responses = truncate_responses(public_dataset[f"response_{i}"].tolist(), tokenizer)
        # Compute Similarities
        scores = compute_similarities(
            sts_model,
            private_responses,
            public_responses,
        )
        public_dataset[f"similarity_score_{i}"] = scores

        # Compute Llama Preferences
        preference_scores = compute_preference_score(
            llm,
            private_responses,
            public_responses,
            public_dataset["instruction"].tolist(),
        )
        public_dataset[f"preference_score_{i}"] = preference_scores

        # Compute Llama Filter
        medical_flags = filter_medical_content(llm, public_responses)
        public_dataset[f"is_medical_{i}"] = medical_flags

        # Compute Educational Scores
        educational_results = compute_educational_score(llm, public_responses)
        public_dataset[f"educational_score_{i}"] = educational_results["educational_scores"]
        public_dataset[f"educational_response_{i}"] = educational_results["educational_responses"]

        # Compute BLEU Score
        bleu_scores = compute_bleu_score(private_responses, public_responses)
        public_dataset[f"bleu_score_{i}"] = bleu_scores


def main() -> None:
    """Main function to score and create datasets for DPO training."""
    args = parse_arguments()

    run = initialize_wandb(args)

    private_dataset, public_dataset = load_datasets(args.private_dataset, args.public_dataset)

    # Score each response
    score_dataset(args.n, args.sts_model, private_dataset, public_dataset, args.tp)

    # Log to wandb
    # TODO Test
    # scored_table = wandb.Table(dataframe=public_dataset.head(1000))
    # wandb.log({"scored_table": scored_table})
    # artifact = wandb.Artifact(name="scored_table", type="dataset")
    # artifact.add(scored_table, "scored_table")
    # wandb.log_artifact(artifact)

    calculate_statistics(public_dataset, args.n, run)

    # Save scored dataset
    scored_parquet_path = (
        f"{args.output_path}/model={args.sts_model.replace('/','-')}_scored.parquet"
    )
    public_dataset.to_parquet(scored_parquet_path)


if __name__ == "__main__":
    main()
