import logging
import os
import shutil

import pandas as pd
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig, omegaconf
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from vllm import LLM

os.environ["WANDB_START_METHOD"] = "thread"


def generate(
    cfg: DictConfig,
    best_model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    gen_dataset: Dataset,
    test_dataset: Dataset,
) -> Dataset:
    """Generate the synthetic candidates for the dataset.
    To improve the speed of the prediction we use VLLM pipeline.
    The number of generated sequences is defined in the config file under
    model.num_generated_sequences.

    Args:
        cfg: The configuration for the generation.
        best_model_path: The path to the best model.
        tokenizer: The tokenizer to use.
        gen_dataset: The dataset to generate the synthetic candidates for.
        test_dataset: The dataset to generate the synthetic candidates for.

    Returns:
        The generated dataset.
    """
    logging.info("✨ Merging Model and save Tokenizer both at models/merged/")
    model = AutoPeftModelForCausalLM.from_pretrained(
        best_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).merge_and_unload()
    model.save_pretrained("models/merged/")
    tokenizer.save_pretrained("models/merged/")
    del model
    del tokenizer
    logging.info("🫧 Building VLLM Pipeline ...")
    llm = LLM(model="models/merged/")

    logging.info("🎉 And it's done!")

    gen_dataloader = torch.utils.data.DataLoader(
        gen_dataset.remove_columns(["input_ids", "max_gen_len"]),
        batch_size=cfg.gen.batch_size,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset.remove_columns(["input_ids", "max_gen_len"]),
        batch_size=cfg.gen.batch_size,
    )

    wandb.init(
        project="style-transfer_gen",
    )
    wandb.config.update(
        omegaconf.OmegaConf.to_container(
            cfg,
        )
    )

    gen_pred_dataset = batch_generate(cfg, gen_dataloader, llm, "gen_dataset")
    _ = batch_generate(cfg, test_dataloader, llm, "test_dataset")
    del llm
    shutil.rmtree("models/merged/")
    wandb.finish()
    return gen_pred_dataset


def batch_generate(cfg, dataloader, llm, wb_ds_name):
    dataset = []
    for batch in tqdm(dataloader):
        flattened_gs_dict = {}
        for g_seq in range(cfg.model.num_generated_sequences):
            responses = llm.generate(batch["query"])
            flattened_gs_dict[f"generation_{g_seq}"] = [
                response.outputs[0].text for response in responses
            ]
        batch_logs = {
            "prompts": batch["query"],
            "ground_texts": batch["text"],
        }
        batch_logs = {**batch_logs, **flattened_gs_dict}
        gen_df = pd.DataFrame.from_dict(batch_logs)
        dataset.append(gen_df)
    wandb.log({wb_ds_name: wandb.Table(dataframe=pd.concat(dataset))})
    return Dataset.from_pandas(pd.concat(dataset))
