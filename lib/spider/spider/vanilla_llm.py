import json
import os

import clearml
import torch
import transformers
from clearml import Task
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from spider._path import _ROOT
from spider.clearml_utils import setup_clearml
from spider.evaluation import build_foreign_key_map_from_json, evaluate
from spider.utils import extract_sql, is_valid_sql

SEED = 42
DATASET_PATH = _ROOT / "data" / "spider"
PROMPT_TEMPLATE = """/* Given the following database schema: */
{}

/* Answer the following: {} Very important: generate only the SQL query and nothing else.*/
SELECT
"""
PROMPT_TEMPLATE_OLD = """Create a SQL query to answer the following question using the database
schema provided below.
Question:
{}

Database schema:
{}

Very important: generate only the SQL query and nothing else.
Query: """

EDIT_PROMPT_TEMPLATE = """Take in account the Error returned to edit or change the SQL query
to make it executable.
Error Returned: {}

Very important: generate only the edited SQL query and nothing else.
SELECT"""
BNB_CONFIG = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
PRETRAINED_MODEL_NAME_OR_PATH: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
GENERATION_CONFIG = dict(
    num_beams=1,
    no_repeat_ngram_size=2,
    num_return_sequences=1,
    early_stopping=True,
    max_new_tokens=200,
)
EVAL_TYPE = "all"
MAX_SAMPLES = 10

IF_LOG_CLEAR_ML = False


if EVAL_TYPE not in ["all", "exec", "match"]:
    raise ValueError("Unknown evaluation method, must be all, exec or match")

OUTPUT_FILE = DATASET_PATH / "output.txt"

gold = DATASET_PATH / "dev_gold.sql"
db_dir = DATASET_PATH / "database"
table = DATASET_PATH / "tables.json"


transformers.set_seed(SEED)


if IF_LOG_CLEAR_ML:
    setup_clearml(env_file_path=_ROOT / ".env")
    task = Task.init(
        project_name="spider",
        task_name="vanilla_llm",
        task_type=clearml.Task.TaskTypes.testing,
    )
    task.set_parameters(
        dict(
            pretrained_model_name_or_path=PRETRAINED_MODEL_NAME_OR_PATH,
            bnb_config=BNB_CONFIG,
            generation_config=GENERATION_CONFIG,
            dataset_path=DATASET_PATH,
            seed=SEED,
            prompt_template=PROMPT_TEMPLATE,
            eval_type=EVAL_TYPE,
        )
    )

# Dataset parsing
dev_json = json.load(open(DATASET_PATH / "dev.json"))
dev_json = [
    {
        "db_id": data_point["db_id"],
        "question": data_point["question"],
    }
    for data_point in dev_json
]

# Model loading
bnb_config = BitsAndBytesConfig(**BNB_CONFIG)
model = AutoModelForCausalLM.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    device_map="auto",
    quantization_config=bnb_config,
)

tokenizer = AutoTokenizer.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH,
    padding_side="left",
)
tokenizer.pad_token = tokenizer.eos_token

# write results to file
if os.path.exists(OUTPUT_FILE):
    # If it exists, delete it
    os.remove(OUTPUT_FILE)

with open(OUTPUT_FILE, "w") as file:
    # Evaluation
    if MAX_SAMPLES is not None:
        dev_json = dev_json[:MAX_SAMPLES]
    for data_point in dev_json:
        schema_paths = list((DATASET_PATH / "database" / data_point["db_id"]).glob("*.sql"))
        if schema_paths:
            schema_path = str(schema_paths[0])
            with open(schema_path, "r") as schema_file:
                data_point["schema"] = schema_file.read()
        else:
            data_point["schema"] = ""
        prompt = PROMPT_TEMPLATE.format(data_point["schema"], data_point["question"])
        # print("FIRST PROMPT: ", prompt)
        messages = [
            {"role": "user", "content": prompt},
        ]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to("cuda")
        outputs = model.generate(
            model_inputs, **GENERATION_CONFIG, pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = extract_sql(message=response, if_first_answer=True)
        messages.append({"role": "assistant", "content": response})
        # print("RESPONSE: ", response)

        # Check if the response is executable
        is_valid, error = is_valid_sql(sql=response, db_id=data_point["db_id"])
        if not is_valid:
            # call the model again to edit the previous response
            edit_prompt = EDIT_PROMPT_TEMPLATE.format(error)
            messages.append({"role": "user", "content": edit_prompt})
            # print("EDIT PROMPT: ", edit_prompt)
            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
            model_inputs = encodeds.to("cuda")
            outputs = model.generate(
                model_inputs, **GENERATION_CONFIG, pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = extract_sql(message=response, if_first_answer=False)
            # print("RESPONSE: ", response)
        file.write(response + "\n")

        print("-" * 100)


kmaps = build_foreign_key_map_from_json(table)

scores = evaluate(gold, OUTPUT_FILE, db_dir, EVAL_TYPE, kmaps)

if IF_LOG_CLEAR_ML:
    task.upload_artifact("output.txt", artifact_object=OUTPUT_FILE)
    task.upload_artifact("score", artifact_object=scores)
    task.connect(scores, name="scores")
    task.close()
