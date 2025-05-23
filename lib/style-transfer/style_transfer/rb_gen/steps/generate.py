import gc
import json
import logging
import os
import sqlite3
from typing import Callable

import hydra
import pandas as pd
import torch
import wandb
from datasets import Dataset
from omegaconf import DictConfig
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from vllm import LLM
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

os.environ["WANDB_START_METHOD"] = "thread"


def generate(
    cfg: DictConfig,
    step: int,
    best_model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    gen_dataset: Dataset,
) -> Dataset:
    """Generate the synthetic candidates for the dataset.
    To improve the speed of the prediction we use VLLM pipeline.
    The number of generated sequences is defined in the config file under
    model.num_generated_sequences.

    Args:
        cfg: The configuration for the generation.
        step: The current step.
        best_model_path: The path to the best model.
        tokenizer: The tokenizer to use.
        gen_dataset: The dataset to generate the synthetic candidates for.

    Returns:
        The generated dataset.
    """
    logging.info("✨ Merging Model and save Tokenizer both at models/merged/")
    wandb.config.update({"state": f"gen/{step}"}, allow_val_change=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        best_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).merge_and_unload()
    model.save_pretrained(f"models/{wandb.run.id}/merged/")
    tokenizer.save_pretrained(f"models/{wandb.run.id}/merged/")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    logging.info("🫧 Building VLLM Pipeline ...")
    llm = hydra.utils.instantiate(cfg.gen.llm, model=f"models/{wandb.run.id}/merged/")

    logging.info("🎉 And it's done!")

    shuffled_gen_dataset = gen_dataset.shuffle(seed=cfg.seed + step)
    subset_gen_dataset = (
        shuffled_gen_dataset.select(range(cfg.dataset.num_generated_samples))
        if step != 0
        else shuffled_gen_dataset
    )
    gen_dataloader = torch.utils.data.DataLoader(
        subset_gen_dataset,
        batch_size=cfg.gen.batch_size,
    )
    gen_pred_dataset = batch_generate(
        cfg, step, gen_dataloader, llm, f"{wandb.config['state']}/gen_dataset"
    )

    wandb.log_artifact(f".cache-{wandb.run.id}.sqlite", type="data")
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return gen_pred_dataset


def batch_generate(cfg, step, dataloader, llm, wb_ds_name) -> Dataset:
    dataset = []
    sampling_params = hydra.utils.instantiate(cfg.gen.sampling_params)
    for batch in tqdm(dataloader):
        flattened_gs_dict = {}
        for g_seq in range(cfg.model.num_generated_sequences):
            flattened_gs_dict[f"generation_{g_seq}"] = predict(
                llm=llm,
                prompts=batch["query"],
                sampling_params=sampling_params,
                id=f"{wandb.run.id}_{step}_{g_seq}_{cfg}_{wb_ds_name}",
            )
        batch_logs = {
            "prompts": batch["query"],
            "ground_texts": batch["ground_texts"],
        }
        batch_logs = {**batch_logs, **flattened_gs_dict}
        gen_df = pd.DataFrame.from_dict(batch_logs)
        dataset.append(gen_df)
    wandb.log({wb_ds_name: wandb.Table(dataframe=pd.concat(dataset))})
    return Dataset.from_pandas(pd.concat(dataset))


def cached(func: Callable) -> Callable:
    """Cache the results of the function using SQLite.

    Args:
        func: The function to cache.

    Returns:
        The cached function.
    """

    def wrapper(**kwargs):
        conn = sqlite3.connect(f".cache-{wandb.run.id}.sqlite")
        cursor = conn.cursor()
        cursor.execute(
            """
                CREATE TABLE IF NOT EXISTS cache (
                    id TEXT PRIMARY KEY,
                    prompts TEXT,
                    result TEXT
                )
            """
        )
        func_kwargs = {k: v for k, v in kwargs.items() if k != "id"}
        key = f"{kwargs['id']}-{json.dumps(kwargs['prompts'])}"

        cursor.execute("SELECT result FROM cache WHERE id = ?", (key,))
        row = cursor.fetchone()
        if row:
            result = json.loads(row[0])
        else:
            result = func(**func_kwargs)
            cursor.execute(
                "INSERT INTO cache (id, prompts, result) VALUES (?, ?, ?)",
                (key, json.dumps(kwargs["prompts"]), json.dumps(result)),
            )
            conn.commit()

        conn.close()
        return result

    return wrapper


@cached
def predict(llm: LLM, prompts: list[str], sampling_params) -> list[str]:
    """Predict next tokens for the prompts using the LLM.

    Args:
        llm: The LLM model.
        prompts: The prompts to generate the next token for.

    Returns:
        The generated tokens.
    """
    return [response.outputs[0].text for response in llm.generate(prompts, sampling_params)]
