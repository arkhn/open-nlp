import argparse

import pandas as pd
import wandb
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer


def load_datasets(private_dataset_path: str, public_dataset_path: str):
    private_dataset = pd.read_parquet(private_dataset_path)
    public_dataset = pd.read_parquet(f"{public_dataset_path}/public_generated.parquet")
    return private_dataset, public_dataset


def preprocess_text(text):
    if isinstance(text, str) and "###" in text:
        return text.split("###")[0].strip()
    return text


def compute_similarities(model, texts1, texts2):
    # Standard global approach
    embeddings1 = model.encode(texts1, convert_to_tensor=True)
    embeddings2 = model.encode(texts2, convert_to_tensor=True)
    global_scores = cos_sim(embeddings1, embeddings2).diagonal().tolist()

    # Initialize tokenizer using the same model
    tokenizer = AutoTokenizer.from_pretrained(model._first_module().auto_model.config._name_or_path)

    # Sliding window approach for longer texts
    window_size = 128
    window_scores = []

    for i, (text1, text2) in enumerate(zip(texts1, texts2)):
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
    return final_scores


def initialize_wandb(args):
    return wandb.init(
        project="synth-kg",
        name=f"eval-{args.wdb_id}",
        group=args.wdb_id,  # Pour le relier visuellement au run d'entraînement
        job_type="evaluation",
        config={},  # vide = pas d’écrasement
        reinit=True,
    )


def calculate_statistics(public_dataset, n, run):
    score_columns = [f"similarity_score_{i}" for i in range(1, n + 1)]
    all_scores = public_dataset[score_columns].values.flatten()
    mean_score = all_scores.mean()
    max_score = all_scores.max()
    min_score = all_scores.min()
    median_score = pd.Series(all_scores).median()

    run.log(
        {
            "mean_score": mean_score,
            "max_score": max_score,
            "min_score": min_score,
            "median_score": median_score,
        }
    )
    print(f"Mean: {mean_score:.4f}")
    print(f"Max: {max_score:.4f}")
    print(f"Min: {min_score:.4f}")
    print(f"Median: {median_score:.4f}")


def get_best_worst_indices(public_dataset, n):
    score_columns = [f"similarity_score_{i}" for i in range(1, n + 1)]
    best_response_idx = (
        public_dataset[score_columns].idxmax(axis=1).apply(lambda x: int(x.split("_")[-1]))
    )
    worst_response_idx = (
        public_dataset[score_columns].idxmin(axis=1).apply(lambda x: int(x.split("_")[-1]))
    )
    return best_response_idx, worst_response_idx


def create_final_dataset(public_dataset, best_response_idx, worst_response_idx):
    final_dataset = pd.DataFrame()
    final_dataset["prompt"] = public_dataset["instruction"]
    final_dataset["chosen"] = public_dataset.apply(
        lambda row: row[f"response_{best_response_idx[row.name]}"], axis=1
    )
    final_dataset["chosen_score"] = public_dataset.apply(
        lambda row: row[f"similarity_score_{best_response_idx[row.name]}"], axis=1
    )

    final_dataset["rejected_score"] = public_dataset.apply(
        lambda row: row[f"similarity_score_{worst_response_idx[row.name]}"], axis=1
    )
    final_dataset["rejected"] = public_dataset.apply(
        lambda row: row[f"response_{worst_response_idx[row.name]}"], axis=1
    )
    return final_dataset


def save_datasets(final_dataset, args):
    final_dataset = final_dataset.sort_values(by="chosen_score", ascending=False)
    eval_dataset = pd.DataFrame()
    eval_dataset["instruction"] = final_dataset["prompt"]
    eval_dataset["response"] = final_dataset["chosen"]

    eval_parquet_path = (
        f"{args.output_path}/model={args.evaluator_path.replace('/','-')}_eval.parquet"
    )
    eval_dataset.to_parquet(eval_parquet_path)

    final_dataset = final_dataset[final_dataset["chosen"].str.len() > 100]
    final_dataset = final_dataset[final_dataset["rejected"].str.len() > 100]
    final_dataset["std"] = final_dataset["chosen_score"] - final_dataset["rejected_score"]

    final_dataset = final_dataset.sort_values(by="chosen_score", ascending=False)
    dpo_parquet_path = (
        f"{args.output_path}/model={args.evaluator_path.replace('/','-')}_dpo.parquet"
    )
    final_dataset.copy().head(1000).to_parquet(dpo_parquet_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wdb_id", type=str, help="Optional wandb run ID to resume a run")
    parser.add_argument("--evaluator_path", type=str, required=True)
    parser.add_argument("--private_dataset", type=str, required=True)
    parser.add_argument("--public_dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--n", type=int, required=True, help="Number of text columns in public dataset"
    )
    args = parser.parse_args()

    run = initialize_wandb(args)

    model = SentenceTransformer(args.evaluator_path)
    private_dataset, public_dataset = load_datasets(args.private_dataset, args.public_dataset)

    for i in range(1, args.n + 1):
        column_name = f"response_{i}"
        public_dataset[column_name] = public_dataset[column_name].apply(preprocess_text)
        scores = compute_similarities(
            model, private_dataset["response"].tolist(), public_dataset[column_name].tolist()
        )
        public_dataset[f"similarity_score_{i}"] = scores

    scored_parquet_path = (
        f"{args.output_path}/model={args.evaluator_path.replace('/','-')}_scored.parquet"
    )
    public_dataset.to_parquet(scored_parquet_path)

    scored_table = wandb.Table(dataframe=public_dataset.head(3000))
    wandb.log({"scored_table": scored_table})
    artifact = wandb.Artifact(name="scored_table", type="dataset")
    artifact.add(scored_table, "scored_table")
    wandb.log_artifact(artifact)

    calculate_statistics(public_dataset, args.n, run)

    best_response_idx, worst_response_idx = get_best_worst_indices(public_dataset, args.n)

    final_dataset = create_final_dataset(public_dataset, best_response_idx, worst_response_idx)

    save_datasets(final_dataset, args)


if __name__ == "__main__":
    main()
