import json
import logging

import datasets
import hydra
import mii
import pandas as pd
import torch
import wandb
from fastchat.conversation import get_conv_template
from omegaconf import omegaconf
from sentence_transformers import SentenceTransformer, util
from style_transfer.utils import EVAL_PROMPT
from tqdm import tqdm


@hydra.main(version_base="1.3", config_path="../configs", config_name="score.yaml")
def main(cfg):
    wandb.config = omegaconf.OmegaConf.to_container(
        cfg,
    )
    with wandb.init(project="score-style-transfer") as run:
        dataset = run.use_artifact(cfg.dataset)
        json_file = json.load(dataset.files()[0].download(replace=True))
        df = pd.DataFrame(data=json_file["data"], columns=json_file["columns"])
        dataset = datasets.Dataset.from_pandas(df)

        logging.info("Loading the Semantic Model 🐈‍⬛")
        sem_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

        def add_prompt(data_point):
            for seq in range(cfg.num_generated_sequences):
                data_point[f"eval_prompt_{seq}"] = str.format(
                    EVAL_PROMPT,
                    data_point["prompts"],
                    data_point[f"generation_{seq}"],
                    data_point["ground_texts"],
                )
                conv = get_conv_template("llama-2")
                conv.set_system_message("You are a fair evaluator language model.")
                conv.append_message(conv.roles[0], data_point[f"eval_prompt_{seq}"])
                conv.append_message(conv.roles[1], None)
                data_point[f"eval_prompt_{seq}"] = conv.get_prompt()
            return data_point

        dataset = dataset.map(
            add_prompt,
            batched=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
        )

        client = mii.client("models/evaluator/") if cfg.use_g_score else None

        def g_scores(batch, seq):
            responses = client(batch[f"eval_prompt_{seq}"], max_new_tokens=cfg.max_new_tokens)
            scores = [response.generated_text[-1] for response in responses]
            scores = [
                float(score) if score.isdigit() and 0 <= float(score) <= 5 else 0
                for score in scores
            ]
            feedbacks = [
                response.generated_text.split("[RESULT]")[0].strip() for response in responses
            ]
            batch.setdefault(f"eval_scores_{seq}", []).extend(scores)
            batch.setdefault(f"eval_feedbacks_{seq}", []).extend(feedbacks)

        def sem_scores(batch, seq):
            ground_enc = sem_model.encode(batch["ground_texts"])
            prediction_enc = sem_model.encode(batch[f"generation_{seq}"])
            scores = [
                util.cos_sim(ground_enc, pred_enc)[0][0].item()
                for ground_enc, pred_enc in zip(ground_enc, prediction_enc)
            ]
            batch.setdefault(f"eval_sem_scores_{seq}", []).extend(scores)

        dataset = []
        for batch in tqdm(dataloader):
            for seq in range(cfg.num_generated_sequences):
                if cfg.use_sem_score:
                    sem_scores(batch, seq)
                if cfg.use_g_score:
                    g_scores(batch, seq)

            df = pd.DataFrame(batch)
            dataset.append(df)

        wandb.log({"score_dataset": wandb.Table(dataframe=pd.concat(dataset))})
    if cfg.use_g_score:
        client.terminate_server()
    wandb.finish()


if __name__ == "__main__":
    main()
