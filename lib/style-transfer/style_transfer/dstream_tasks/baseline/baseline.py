import json
import logging

import datasets
import hydra
import pandas as pd
import torch
import wandb
from sentence_transformers import SentenceTransformer, util


@hydra.main(version_base="1.3", config_path="../configs", config_name="default.yaml")
def main(cfg):
    api = wandb.Api()
    test_dataset = api.artifact(cfg.test_dataset)
    json_file = json.load(test_dataset.files()[0].download(replace=True))
    df = pd.DataFrame(data=json_file["data"], columns=json_file["columns"])
    test_dataset = datasets.Dataset.from_pandas(df)

    logging.info("Loading the Semantic Model 🐈‍")
    sem_model = SentenceTransformer(cfg.sem_model.name)
    model_artifact = api.artifact(cfg.sem_model.checkpoint)
    model_dir = model_artifact.download()
    sem_model = sem_model.load(model_dir)

    def sem_score(cfg, dataset, sem_model):
        ground_enc = sem_model.encode(
            dataset["ground_texts"],
            batch_size=cfg.batch_size,
        )
        prediction_enc = sem_model.encode(
            dataset["keywords"],
            batch_size=cfg.batch_size,
        )
        scores = [
            util.cos_sim(ground_enc, pred_enc)[0][0].item()
            for ground_enc, pred_enc in zip(ground_enc, prediction_enc)
        ]
        return scores

    test_dataset = test_dataset.map(
        lambda x: {"keywords": x["prompts"].split("Keywords: ")[1].removesuffix("[/INST]\n")}
    )
    score = torch.mean(torch.tensor(sem_score(cfg, test_dataset, sem_model)))
    print("mean: ", score)


if __name__ == "__main__":
    main()
