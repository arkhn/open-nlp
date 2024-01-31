import json
import logging

import datasets
import hydra
import mii
import pandas as pd
import torch
import wandb
from fastchat.conversation import get_conv_template
from style_transfer.utils import EVAL_PROMPT
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@hydra.main(version_base="1.3", config_path="../configs", config_name="score.yaml")
def main(cfg):
    with wandb.init(project="score-style-transfer") as run:
        dataset = run.use_artifact(cfg.dataset)
        json_file = json.load(dataset.files()[0].download(replace=True))
        df = pd.DataFrame(data=json_file["data"], columns=json_file["columns"])
        dataset = datasets.Dataset.from_pandas(df)

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

        logging.info("Model + Tokenizer saved at models/merged/")
        logging.info("Loading model to pipeline 🐉 ...")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=cfg.evaluator,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.save_pretrained("models/evaluator/")
        tokenizer = AutoTokenizer.from_pretrained(cfg.evaluator)
        tokenizer.save_pretrained("models/evaluator/")
        del model
        del tokenizer
        pipe = mii.pipeline("models/evaluator/")
        logging.info("Model loaded to pipeline ! 🎉")

        new_dataset = []
        for batch in tqdm(dataloader):
            for seq in range(cfg.num_generated_sequences):
                responses = pipe(batch[f"eval_prompt_{seq}"], max_new_tokens=cfg.max_new_tokens)
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

            new_dataset.extend([dict(zip(batch, t)) for t in zip(*batch.values())])
            table = wandb.Table(dataframe=pd.DataFrame(batch))
            wandb.log({"generation_predictions": table})
        df = pd.DataFrame(new_dataset)
        wandb.log({"dataframe_table": wandb.Table(dataframe=df)})
    wandb.finish()


if __name__ == "__main__":
    main()
