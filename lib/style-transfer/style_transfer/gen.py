import logging

import hydra
import mii
import pandas as pd
import torch
import wandb
from omegaconf import omegaconf
from peft import AutoPeftModelForCausalLM
from style_transfer.utils import PROMPT, build_dataset, split_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


@hydra.main(version_base="1.3", config_path="../configs", config_name="gen.yaml")
def gen(cfg):
    api = wandb.Api()
    model_artifact = api.artifact(cfg.checkpoint)
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
    client = mii.serve(
        "models/merged/",
        tensor_parallel=4,
        deployment_name=cfg.checkpoint,
    )
    logging.info("Model loaded to pipeline ! 🎉")
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
    sft_dataset, gen_dataset, test_dataset = split_dataset(dataset, cfg.sft_ratio, cfg.gen_ratio)
    gen_dataset = gen_dataset.remove_columns(["input_ids", "max_gen_len"])
    test_dataset = test_dataset.remove_columns(["input_ids", "max_gen_len"])
    dataloader = torch.utils.data.DataLoader(
        gen_dataset,
        batch_size=cfg.batch_size,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
    )
    dataset = []
    wandb.config.update(
        omegaconf.OmegaConf.to_container(
            cfg,
        )
    )
    wandb.config["sft_dataset_size"] = len(sft_dataset)
    wandb.config["gen_dataset_size"] = len(gen_dataset)
    wandb.config["test_dataset_size"] = len(test_dataset)

    wandb.init(
        project="gen-style-transfer",
        name=f"sft-ratio-{cfg.sft_ratio}_gen-ratio-{cfg.gen_ratio}",
    )
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
        print(flattened_gs_dict["generation_0"])
        batch_logs = {**batch_logs, **flattened_gs_dict}
        gen_df = pd.DataFrame.from_dict(batch_logs)
        dataset.append(gen_df)

    wandb.log({"gen_dataset": wandb.Table(dataframe=pd.concat(dataset))})

    test_dataset = []
    for batch in tqdm(test_dataloader):
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
        test_df = pd.DataFrame.from_dict(batch_logs)
        test_dataset.append(test_df)

    wandb.log({"test_dataset": wandb.Table(dataframe=pd.concat(test_dataset))})
    wandb.finish()


if __name__ == "__main__":
    gen()
