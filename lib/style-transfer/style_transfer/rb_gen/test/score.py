import hydra
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import InputExample, SentenceTransformer, losses, util


def load_data(csv_path):
    return Dataset.from_pandas(pd.read_csv(csv_path))


def split_dataset(dataset, train_size=0.6):
    split_data = dataset.train_test_split(train_size=train_size)
    return split_data["train"], split_data["test"]


def encode_and_score_texts(model, texts1, texts2, batch_size=8):
    enc1 = model.encode(texts1, batch_size=batch_size)
    enc2 = model.encode(texts2, batch_size=batch_size)
    return [util.cos_sim(e1, e2)[0][0].item() for e1, e2 in zip(enc1, enc2)]


def create_input_examples(texts1, texts2, scores, offset=1):
    return [
        InputExample(texts=[t1, t2], label=score + offset)
        for t1, t2, score in zip(texts1, texts2, scores)
    ]


def process_ground_predictions(dataset, model):
    split_point = len(dataset["ground_texts"]) // 2
    split1 = dataset["ground_texts"][:split_point]
    split2 = dataset["ground_texts"][split_point:]
    scores = encode_and_score_texts(model, split1, split2)
    return {"ground_scores": scores, "text_1": split1, "text_2": split2}


def process_generation_scores(dataset, model):
    ground_enc = model.encode(dataset["ground_texts"], batch_size=8)
    score_dict = {}
    for seq in range(4):
        prediction_enc = model.encode(dataset[f"generation_{seq}"], batch_size=8)
        scores = [
            util.cos_sim(g_enc, p_enc)[0][0].item()
            for g_enc, p_enc in zip(ground_enc, prediction_enc)
        ]
        score_dict[f"evaluator_scores_{seq}"] = scores
    return score_dict


def train_model(model, train_examples, batch_size=8, epochs=2, warmup_steps=50):
    train_dataloader = torch.utils.data.DataLoader(
        train_examples,
        batch_size=batch_size,
    )
    train_loss = losses.ContrastiveLoss(model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
    )


def main():
    # Load and split dataset
    csv = load_data("./test/score.csv")
    train_dataset, test_dataset = split_dataset(csv)

    # Initialize model and examples list
    eval_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    train_examples = []

    # Process training data
    train_ground_dict = process_ground_predictions(train_dataset, eval_model)
    train_examples.extend(
        create_input_examples(
            train_ground_dict["text_1"],
            train_ground_dict["text_2"],
            train_ground_dict["ground_scores"],
        )
    )

    # Process test data
    gen_ground_dict = process_ground_predictions(test_dataset, eval_model)
    train_examples.extend(
        create_input_examples(
            gen_ground_dict["text_1"], gen_ground_dict["text_2"], gen_ground_dict["ground_scores"]
        )
    )

    # Process generation scores for training
    train_score_dict = process_generation_scores(train_dataset, eval_model)
    scores = []
    for seq in range(4):
        scores.extend(train_score_dict[f"evaluator_scores_{seq}"])
    print("before learning:")
    print(f"Global mean: {np.mean(scores):.4f}, Global std: {np.std(scores):.4f}")
    for seq in range(4):
        train_examples.extend(
            create_input_examples(
                train_dataset[f"generation_{seq}"],
                train_dataset["ground_texts"],
                train_score_dict[f"evaluator_scores_{seq}"],
                offset=-1,
            )
        )

    # Train the model
    train_model(eval_model, train_examples)

    # Generate final scores
    score_dict = process_generation_scores(test_dataset, eval_model)
    scores = []
    for seq in range(4):
        scores.extend(score_dict[f"evaluator_scores_{seq}"])
    print("after learning:")
    print(f"Global mean: {np.mean(scores):.4f}, Global std: {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
