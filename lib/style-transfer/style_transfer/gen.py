import hydra
import mii
import pandas as pd
import torch
import wandb
from omegaconf import omegaconf
from style_transfer.utils import PROMPT, build_dataset, split_dataset
from tqdm import tqdm


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
    client = mii.client("models/merged/")

    wandb.config = omegaconf.OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True,
    )
    wandb.init(project="gen-style-transfer")
    for batch in tqdm(dataloader):
        flattened_gs_dict = {}
        for g_seq in range(cfg.num_generated_sequences):
            responses = client.generate(
                batch["prompts"],
                max_new_tokens=cfg.max_new_tokens,
            )
            flattened_gs_dict[f"generation_{g_seq}"] = [
                response.generated_text for response in responses
            ]
        batch_logs = {
            "prompts": batch["prompts"],
            "ground_texts": batch["ground_texts"],
        }
        batch_logs = {**batch_logs, **flattened_gs_dict}
        df = pd.DataFrame.from_dict(batch_logs)
        wandb.log({"generation_predictions": wandb.Table(dataframe=df)})

    client.terminate_server()
    wandb.finish()


if __name__ == "__main__":
    main()
