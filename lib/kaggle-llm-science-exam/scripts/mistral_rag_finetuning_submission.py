"""
This script was a submission for the Kaggle LLM Science Exam competition.
It uses a FAISS index computed on Wikipedia articles (only titles and first sentence) to find the
closest article to the prompt. Then the article is split into chunks of 1000 characters and a new
FAISS index is computed on the chunks. The closest chunk to the prompt is used as context in the
prompt.
Result: The notebook crashed after creating the submission file
"""

# Requirements:
# !pip install accelerate
# !pip install bitsandbytes
# !pip install peft
# !pip install transformers
# !pip install trl
# !pip install polars
# !pip install faiss-gpu
# !pip install sentence-transformers

import gc
import re

import faiss
import numpy as np
import pandas as pd
import polars as pl
import torch
import transformers
from datasets import Dataset
from faiss import read_index
from peft import LoraConfig, PeftConfig, PeftModel, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

seed = 42
transformers.set_seed(seed)

prompt_template = """### Context: {}

### Question: {}
A: {}
B: {}
C: {}
D: {}
E: {}
Answer by only giving the letter corresponding to the correct response.

### Answer:"""
response_template = "Answer:"
base_model_name = "/kaggle/input/mistral/pytorch/7b-instruct-v0.1-hf/1"
embedding_model_name = "/kaggle/input/all-minilm-l6-v2/all-MiniLM-L6-v2"
bits_and_bytes_config = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": False,
}
lora_config = {
    "r": 64,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
training_config = {
    "output_dir": "/kaggle/working/results",
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "optim": "paged_adamw_32bit",
    "save_steps": 20000,
    "logging_steps": 5,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "fp16": False,
    "bf16": False,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "group_by_length": True,
    "lr_scheduler_type": "constant",
    "report_to": "none",
}
generation_config = {
    "max_new_tokens": 1,
    "do_sample": False,
    "num_beams": 3,
    "num_beam_groups": 3,
    "diversity_penalty": 4.0,
    "num_return_sequences": 3,
    "return_dict_in_generate": True,
    "output_scores": True,
    "pad_token_id": 50256,
}

df_train_dataset = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv")
df_test_dataset = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv")

embedding_model = SentenceTransformer(embedding_model_name)
wiki_index = read_index("/kaggle/input/wikipedia-2023-07-faiss-index/wikipedia_202307.index")
wiki_df = pl.read_parquet(
    "/kaggle/input/wikipedia-20230701/wiki_2023_index.parquet", columns=["id", "file"]
)


def get_contexts(df_dataset):
    def merge_question_answers(row):
        row["question"] = "\n".join([row.prompt, row.A, row.B, row.C, row.D, row.E])
        return row

    df_dataset = df_dataset.apply(merge_question_answers, axis=1)
    question_embeddings = embedding_model.encode(df_dataset["question"].values)
    faiss.normalize_L2(question_embeddings)
    scores, indexes = wiki_index.search(question_embeddings, 1)  # noqa: F821

    contexts = []
    for question_embedding, index in tqdm(
        zip(question_embeddings, indexes), total=len(question_embeddings)
    ):
        # Find the full wiki text
        file = wiki_df.item(row=int(index[0]), column="file")  # noqa: F821
        id_ = wiki_df.item(row=int(index[0]), column="id")  # noqa: F821
        file_df = pl.read_parquet(f"/kaggle/input/wikipedia-20230701/{file}")
        wiki_text = file_df.filter(pl.col("id") == id_).item(row=0, column="text")

        # remove titles
        wiki_text = re.sub("(==.*?==)", "", wiki_text)

        # split text into chunks
        wiki_text_chunks = [wiki_text[i : i + 1000] for i in range(0, len(wiki_text), 1000)]

        # create a new faiss index with the chunks
        chunks_embeddings = embedding_model.encode(wiki_text_chunks)
        chunk_index = faiss.IndexFlatL2(chunks_embeddings.shape[1])
        faiss.normalize_L2(chunks_embeddings)
        chunk_index.add(chunks_embeddings)

        chunk_scores, chunk_indexes = chunk_index.search(np.array([question_embedding]), 1)
        contexts.append(wiki_text_chunks[chunk_indexes[0][0]])
        del chunks_embeddings
        del chunk_index
        gc.collect()

    del question_embeddings
    gc.collect()

    return contexts


df_train_dataset["context"] = get_contexts(df_train_dataset)
df_test_dataset["context"] = get_contexts(df_test_dataset)

del wiki_index
del wiki_df

gc.collect()
torch.cuda.empty_cache()


def create_prompt(row, with_answer=True):
    prompt = prompt_template.format(
        row.context,
        row.prompt,
        row.A,
        row.B,
        row.C,
        row.D,
        row.E,
    )
    if with_answer:
        prompt += " " + row.answer
    row["text"] = prompt
    return row


df_train_dataset = df_train_dataset.apply(lambda row: create_prompt(row, with_answer=True), axis=1)
df_test_dataset = df_test_dataset.apply(lambda row: create_prompt(row, with_answer=False), axis=1)
train_dataset = Dataset.from_pandas(df_train_dataset)
test_dataset = Dataset.from_pandas(df_test_dataset)

bnb_config = BitsAndBytesConfig(**bits_and_bytes_config)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
base_model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

peft_params = LoraConfig(**lora_config)
base_model = prepare_model_for_kbit_training(base_model)
training_args = TrainingArguments(**training_config)
trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    args=training_args,
    data_collator=DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    ),
    peft_config=peft_params,
    max_seq_length=2000,
)

trainer.train()

trainer.model.save_pretrained("/kaggle/working/peft_model")

del base_model
del tokenizer
del trainer
del peft_params
del training_args
gc.collect()
torch.cuda.empty_cache()


config = PeftConfig.from_pretrained("/kaggle/working/peft_model")
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(base_model, "/kaggle/working/peft_model", is_trainable=False)
model.config.use_cache = True

gc.collect()


submission_records = []

for example in tqdm(test_dataset, total=len(test_dataset)):
    model_inputs = tokenizer(example["text"], return_tensors="pt").to("cuda:0")
    outputs = model.generate(**model_inputs, **generation_config)
    predictions_scores = []
    for output, score in zip(outputs["sequences"], outputs["sequences_scores"]):
        response = tokenizer.decode(output, skip_special_tokens=True)
        predictions = re.findall(f"{response_template.lower()}  ?(a|b|c|d|e)", response.lower())
        if predictions:
            predictions_scores.append((predictions[0].upper(), float(score)))
        predictions_scores.sort(key=lambda item: item[1], reverse=True)
    submission_records.append(
        {"id": example["id"], "prediction": " ".join([item[0] for item in predictions_scores])}
    )

submission = pd.DataFrame.from_records(submission_records)

submission.to_csv("/kaggle/working/submission.csv", index=False)
