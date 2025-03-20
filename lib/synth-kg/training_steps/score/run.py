import argparse

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import wandb  # Add this import


def load_datasets(private_dataset_path: str, public_dataset_path: str):
    private_dataset = pd.read_parquet(private_dataset_path)
    public_dataset = pd.read_parquet(f"{public_dataset_path}/public_generated.parquet")
    return private_dataset, public_dataset


def preprocess_text(text):
    if isinstance(text, str) and "###" in text:
        return text.split("###")[0].strip()
    return text


def compute_similarities(model, texts1, texts2):
    embeddings1 = model.encode(texts1, convert_to_tensor=True)
    embeddings2 = model.encode(texts2, convert_to_tensor=True)
    scores = cos_sim(embeddings1, embeddings2).diagonal().tolist()
    return scores


def main():
    # Initialize wandb
    wandb.init(project="your_project_name", entity="your_entity_name")  # Replace with your project and entity names
    parser.add_argument("--evaluator_path", type=str, required=True)
    parser.add_argument("--private_dataset", type=str, required=True)
    parser.add_argument("--public_dataset", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--n", type=int, required=True, help="Number of text columns in public dataset"
    )
    args = parser.parse_args()
    model = SentenceTransformer(args.evaluator_path)
    private_dataset, public_dataset = load_datasets(args.private_dataset, args.public_dataset)

    for i in range(1, args.n + 1):
        column_name = f"response_{i}"
        public_dataset[column_name] = public_dataset[column_name].apply(preprocess_text)
        scores = compute_similarities(
            model, private_dataset["response"].tolist(), public_dataset[column_name].tolist()
        )
        public_dataset[f"similarity_score_{i}"] = scores

    scored_parquet_path = f"{args.output_path}/model={args.evaluator_path.replace('/','-')}_scored.parquet"
    public_dataset.to_parquet(scored_parquet_path)

    # Log the scored parquet file to wandb
    wandb.save(scored_parquet_path)
    # Calculate and print score statistics
    score_columns = [f"similarity_score_{i}" for i in range(1, args.n + 1)]
    all_scores = public_dataset[score_columns].values.flatten()
    # Log score statistics to wandb
    wandb.log({
        "mean_score": mean_score,
        "max_score": max_score,
        "min_score": min_score,
        "median_score": median_score
    })
    print(f"Mean: {all_scores.mean():.4f}")
    print(f"Max: {all_scores.max():.4f}")
    print(f"Min: {all_scores.min():.4f}")
    print(f"Median: {pd.Series(all_scores).median():.4f}")

    # Get the best and worst scores for each row
    score_columns = [f"similarity_score_{i}" for i in range(1, args.n + 1)]
    best_response_idx = (
        public_dataset[score_columns].idxmax(axis=1).apply(lambda x: int(x.split("_")[-1]))
    )
    worst_response_idx = (
        public_dataset[score_columns].idxmin(axis=1).apply(lambda x: int(x.split("_")[-1]))
    )

    # Create final dataset with chosen and rejected responses
    final_dataset = pd.DataFrame()
    final_dataset["prompt"] = public_dataset["instruction"]
    final_dataset["chosen"] = public_dataset.apply(
        lambda row: row[f"response_{best_response_idx[row.name]}"], axis=1
    )
    final_dataset["chosen_score"] = public_dataset.apply(
        lambda row: row[f"similarity_score_{best_response_idx[row.name]}"], axis=1
    )
    final_dataset["rejected"] = public_dataset.apply(
        lambda row: row[f"response_{worst_response_idx[row.name]}"], axis=1
    )

    # Create evaluation dataset with only instruction and response

    final_dataset = final_dataset.copy()
    eval_dataset = pd.DataFrame()
    eval_dataset["instruction"] = final_dataset["prompt"]
    eval_dataset["response"] = final_dataset["chosen"]

    eval_parquet_path = f"{args.output_path}/model={args.evaluator_path.replace('/','-')}_eval.parquet"
    eval_dataset.to_parquet(eval_parquet_path)

    # Log the evaluation parquet file to wandb
    wandb.save(eval_parquet_path)

    final_dataset = final_dataset[final_dataset["chosen"].apply(lambda x: len(x.split()) >= 20)]
    final_dataset = final_dataset.sort_values(by="chosen_score", ascending=False)
    dpo_parquet_path = f"{args.output_path}/model={args.evaluator_path.replace('/','-')}_dpo.parquet"
    final_dataset.copy().head(1000).to_parquet(dpo_parquet_path)

    # Log the DPO parquet file to wandb
    wandb.save(dpo_parquet_path)


if __name__ == "__main__":
    main()
