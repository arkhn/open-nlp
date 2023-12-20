"""To run the script, use the following command:
python spider/in_context_learning.py -m prompts=default,icl-1,icl-2,icl-3,icl-4  \
pretrained_model_name_or_path=mistralai/Mixtral-8x7B-Instruct-v0.1

It will run the sweep over the different prompts and the pretrained model.
"""

import json
import os

import clearml
import hydra
import torch
import transformers
import wandb
from clearml import Task
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from wandb import wandb_run

from spider._path import CONFIG_PATH, DATASET_PATH, ENV_FILE_PATH
from spider.clearml_utils import setup_clearml
from spider.evaluation import build_foreign_key_map_from_json, evaluate
from spider.utils import extract_sql, is_valid_sql

OUTPUT_FILE = DATASET_PATH / "output.txt"
gold = DATASET_PATH / "dev_gold.sql"
db_dir = DATASET_PATH / "database"
table = DATASET_PATH / "tables.json"


@hydra.main(version_base="1.3", config_path=str(CONFIG_PATH), config_name="eval.yaml")
def main(cfg) -> None:
    transformers.set_seed(cfg.seed)
    # loggers
    setup_clearml(env_file_path=ENV_FILE_PATH)
    task = Task.init(
        project_name="spider",
        task_name="vanilla_llm",
        task_type=clearml.Task.TaskTypes.testing,
    )
    run = wandb.init(project="spider")
    if not isinstance(run, wandb_run.Run):
        raise TypeError("Run is not a valid Wandb run")

    dev_json = json.load(open(DATASET_PATH / "dev.json"))
    dev_json = [
        {
            "db_id": data_point["db_id"],
            "question": data_point["question"],
        }
        for data_point in dev_json
    ]
    # Model loading
    model = AutoModelForCausalLM.from_pretrained(
        cfg.pretrained_model_name_or_path,
        device_map="auto",
        quantization_config=hydra.utils.instantiate(cfg.bnb_config),
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    generation_config = hydra.utils.instantiate(cfg.generation_config)
    # write results to file
    if os.path.exists(OUTPUT_FILE):
        # If it exists, delete it
        os.remove(OUTPUT_FILE)
    with open(OUTPUT_FILE, "w") as file:
        # Evaluation
        if cfg.max_samples is not None:
            dev_json = dev_json[: cfg.max_samples]
        for data_point in tqdm(dev_json, total=len(dev_json)):
            schema_paths = list((DATASET_PATH / "database" / data_point["db_id"]).glob("*.sql"))
            if schema_paths:
                schema_path = str(schema_paths[0])
                with open(schema_path, "r") as schema_file:
                    data_point["schema"] = schema_file.read()
            else:
                data_point["schema"] = ""
            prompt = cfg.prompts.template.format(data_point["schema"], data_point["question"])
            messages = [
                {"role": "user", "content": prompt},
            ]
            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
            model_inputs = encodeds.to("cuda")
            outputs = model.generate(
                model_inputs,
                generation_config=generation_config,
                pad_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[1]
            response = extract_sql(message=response, if_first_answer=True)
            messages.append({"role": "assistant", "content": response})

            # Check if the response is executable
            is_valid, error = is_valid_sql(sql=response, db_id=data_point["db_id"])
            if not is_valid:
                # call the model again to edit the previous response
                edit_prompt = cfg.prompts.edit_template.format(error)
                messages.append({"role": "user", "content": edit_prompt})
                encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
                model_inputs = encodeds.to("cuda")
                outputs = model.generate(
                    model_inputs,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[
                    1
                ]
                response = extract_sql(message=response, if_first_answer=False)

            print("RESPONSE: ", response)
            file.write(response + "\n")

    kmaps = build_foreign_key_map_from_json(table)
    scores = evaluate(gold, OUTPUT_FILE, db_dir, cfg.eval_type, kmaps)
    run.log(scores)
    task.upload_artifact("output.txt", artifact_object=OUTPUT_FILE)
    task.upload_artifact("score", artifact_object=scores)
    task.connect(scores, name="scores")

    task.close()
    run.config.update(OmegaConf.to_container(cfg))
    run.finish()


if __name__ == "__main__":
    main()
