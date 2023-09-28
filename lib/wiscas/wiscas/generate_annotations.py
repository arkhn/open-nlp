import ast
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import datasets
import pandas as pd
import typer
import wandb
from tqdm import tqdm
from wiscas.metrics import eval, log_wandb
from wiscas.models import LlamaModel, OpenAIModel
from wiscas.models.base import Model
from wiscas.paths import ROOT
from wiscas.prompters import PromptifyPrompter


def main(
    train_parquet_dataset_path: Path,
    test_parquet_dataset_path: Path,
    n_test_examples: Optional[int] = None,
    model_type: Optional[str] = "openai",
    prompt_type: Optional[str] = "promptify",
    n_ic_examples: int = 10,
    seed: int = 1998,
    model_kwargs: str = typer.Option("{}", callback=ast.literal_eval),
):
    """Generate annotations using a model and a prompter.

    Run metrics by comparing to human annotations and log these metrics to wandb.

    Args:
        train_parquet_dataset_path: Path to the train parquet dataset.
        test_parquet_dataset_path: Path to the test parquet dataset.
        n_test_examples: Maximum number of examples to use for testing.
        model_type: Type of model to use.
        prompt_type: Type of prompter to use.
        n_ic_examples: Number of in context examples to use.
        seed: Seed to use for reproducibility.
        model_kwargs: Additional arguments to pass to the model.
    """
    random.seed(seed)
    params = locals()

    test_dataset: datasets.Dataset = datasets.Dataset.from_parquet(test_parquet_dataset_path)
    train_dataset: datasets.Dataset = datasets.Dataset.from_parquet(train_parquet_dataset_path)
    if n_test_examples:
        test_dataset = test_dataset.select(range(n_test_examples))

    model: Model
    if model_type == "openai":
        model = OpenAIModel(**model_kwargs)  # type: ignore
    elif model_type == "llama":
        if "llama_url" not in model_kwargs:
            raise ValueError("Please provide an url for the llama model.")
        model = LlamaModel(**model_kwargs)  # type: ignore

    else:
        raise ValueError(
            f"model_type {model_type} not supported, supported values are: openain, llama"
        )

    if prompt_type == "promptify":
        prompter = PromptifyPrompter(train_dataset=train_dataset, n_examples=n_ic_examples)
    else:
        raise ValueError(
            f"prompt_type {prompt_type} not supported, supported values are: promptify"
        )

    weak_dataset_records = []

    for example in tqdm(test_dataset):
        input_text = example["text"]

        prompt = prompter.get_prompt(text_input=input_text)
        response = model.run(prompt=prompt)
        entities = prompter.response_to_entities(response=response, input_text=input_text)
        weak_dataset_records.append({"text": input_text, "entities": entities})

    weak_dataset_df = pd.DataFrame.from_records(data=weak_dataset_records)
    weak_dataset = datasets.Dataset.from_pandas(weak_dataset_df)
    output_dir = ROOT / "data" / f"dataset_{model_type}_{prompt_type}_{datetime.now()}"
    os.makedirs(output_dir, exist_ok=True)
    weak_dataset.to_parquet(output_dir / "dataset.parquet")

    config = {
        "params": params,
        "model": {
            attr: getattr(model, attr)
            for attr in [
                "model",
                "temperature",
                "top_p",
            ]
        },
        "prompt": prompt,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f)

    metrics = eval(test_dataset=test_dataset, weak_dataset=weak_dataset)

    run = wandb.init(  # type: ignore
        project="wiscas",
        group="test_predictions",
        job_type="analytics",
        config={"weak_dataset_config": config},
    )
    log_wandb(classification_report=metrics, run=run)

    run.finish()


if __name__ == "__main__":
    typer.run(main)
