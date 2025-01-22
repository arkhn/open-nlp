import argparse

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def load_datasets(private_dataset_path: str, public_dataset_path: str):
    private_dataset = pd.read_parquet(private_dataset_path)
    public_dataset = pd.read_parquet(public_dataset_path)
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
    parser = argparse.ArgumentParser()
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

    public_dataset.to_parquet(
        f"{args.output_path}/model={args.evaluator_path.replace('/','-')}_scored.parquet"
    )
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
    final_dataset["rejected"] = public_dataset.apply(
        lambda row: row[f"response_{worst_response_idx[row.name]}"], axis=1
    )

    final_dataset.to_parquet(
        f"{args.output_path}/model={args.evaluator_path.replace('/','-')}_dpo.parquet"
    )

    # Create evaluation dataset with only instruction and response
    eval_dataset = pd.DataFrame()
    eval_dataset["instruction"] = final_dataset["prompt"]
    eval_dataset["response"] = final_dataset["chosen"]

    eval_dataset.to_parquet(
        f"{args.output_path}/model={args.evaluator_path.replace('/','-')}_eval.parquet"
    )


if __name__ == "__main__":
    main()
