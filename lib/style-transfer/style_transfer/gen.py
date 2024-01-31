import logging

import hydra
import mii
import pandas as pd
import torch
import wandb
from peft import AutoPeftModelForCausalLM
from style_transfer.utils import PROMPT, build_dataset, split_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


@hydra.main(version_base="1.3", config_path="../configs", config_name="gen.yaml")
def main(cfg):
    dataset = build_dataset(
        dataset_name=cfg.dataset,
        model_name=cfg.model,
        max_sampler_length=cfg.max_seq_length,
    )

    def add_prompt(data_point):
        data_point["prompts"] = (
            "<s>[INST]"
            + str.format(
                PROMPT,
                data_point["keywords"],
            )
            + "[/INST]\n"
        )
        return data_point

    dataset = dataset.map(
        add_prompt,
        batched=False,
    )
    _, gen_dataset, _ = split_dataset(dataset, cfg.sft_ratio, cfg.gen_ratio)
    gen_dataset = dataset.remove_columns(["input_ids", "max_gen_len"])
    dataloader = torch.utils.data.DataLoader(
        gen_dataset,
        batch_size=cfg.batch_size,
    )

    with wandb.init(project="gen-style-transfer") as run:
        model_artifact = run.use_artifact(cfg.checkpoint)
        model_dir = model_artifact.download()

        model = AutoPeftModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = model.merge_and_unload()
        model.save_pretrained("models/merged/")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        tokenizer.save_pretrained("models/merged/")
        del model
        del tokenizer
        logging.info("Model + Tokenizer saved at models/merged/")
        logging.info("Loading model to pipeline 🐉 ...")
        pipe = mii.pipeline("models/merged/")
        logging.info("Model loaded to pipeline ! 🎉")

        new_dataset = []
        for batch in tqdm(dataloader):
            generated_sequences = []
            for _ in range(cfg.num_generated_sequences):
                responses = pipe(batch["prompts"], max_new_tokens=cfg.max_new_tokens)
                generated_sequences.append([response.generated_text for response in responses])

            responses = list(map(list, zip(*generated_sequences)))
            flattened_gs_dict = {
                f"generation_{reponse_id}": response
                for reponse_id, response in enumerate(responses)
            }
            batch_logs = {
                "prompts": batch["prompts"],
                "ground_texts": batch["ground_texts"],
            }
            batch_logs = {**batch_logs, **flattened_gs_dict}
            new_dataset.extend([dict(zip(batch_logs, t)) for t in zip(*batch_logs.values())])
            table = wandb.Table(dataframe=pd.DataFrame(batch_logs))
            wandb.log({"generation_predictions": table})
        df = pd.DataFrame(new_dataset)
        wandb.log({"dataframe_table": wandb.Table(dataframe=df)})
    wandb.finish()


if __name__ == "__main__":
    main()
