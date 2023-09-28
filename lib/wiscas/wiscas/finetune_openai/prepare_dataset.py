import json
import os
from pathlib import Path
from typing import Optional

import datasets
import typer
from wiscas.paths import ROOT
from wiscas.prompters import PromptifyPrompter


def main(
    parquet_dataset_path: Path,
    prompt_type: str = "promptify",
    max_train_samples: Optional[int] = None,
    output_path: str = str(ROOT / "data" / "finetune_openai"),
):
    os.makedirs(output_path, exist_ok=True)

    train_dataset: datasets.Dataset = datasets.Dataset.from_parquet(parquet_dataset_path)

    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(max_train_samples))

    if prompt_type == "promptify":
        prompter = PromptifyPrompter(train_dataset=train_dataset)
    else:
        raise ValueError(f"Prompter type {prompt_type} not supported.")

    train_examples = []
    for dataset_example in train_dataset:
        completion = prompter.dataset_example_to_completion(example=dataset_example)
        train_examples.append(
            {
                "prompt": prompter.get_prompt(text_input=dataset_example["text"]),
                "completion": completion,
            }
        )

    with open(os.path.join(output_path, "dataset.json"), "w") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    with open(os.path.join(output_path, "config.json"), "w") as f:
        f.write(
            json.dumps(
                {
                    "parquet_dataset_path": parquet_dataset_path,
                    "prompt_type": prompt_type,
                    "max_train_samples": max_train_samples,
                }
            )
        )


if __name__ == "__main__":
    typer.run(main)
