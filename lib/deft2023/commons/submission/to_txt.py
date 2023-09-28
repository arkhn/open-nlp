import wandb
from commons.submission.to_letters import to_letters


def log_predictions(predictions_classes, dataset_sub):
    """Log predictions to wandb.

    Args:
        predictions_classes: List of predictions in letters.
        dataset_sub: Subset of the dataset.
    """

    answer_ids = [f"answer_{qid}" for qid in ["a", "b", "c", "d", "e"]]
    data = []
    for row, prediction in zip(dataset_sub, predictions_classes):
        question = row["question"]
        answers = [row[q_id] for q_id in answer_ids]
        data.append([question] + answers + [f"{'|'.join(prediction)}"])
    table = wandb.Table(columns=["question"] + answer_ids + ["prediction"], data=data)
    wandb.log({"predictions": table})


def to_txt(predictions, dataset_sub, file_path, id2label):

    predictions_classes = to_letters(predictions, id2label)
    ids = [d["id"] for d in dataset_sub]
    log_predictions(predictions_classes, dataset_sub)
    f = open(file_path, "w")
    for i, p in zip(ids, predictions_classes):
        f.write(f"{i};{'|'.join(p)}\n")
    f.close()
