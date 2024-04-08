import asyncio
import json
import os
from operator import itemgetter
from uuid import UUID

import pandas as pd
import typer
import wandb
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.output_parsers import OutputFixingParser
from langchain_community.cache import SQLiteCache
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import AzureChatOpenAI
from tqdm import tqdm

os.environ["DEPLOYMENT_NAME"] = "gpt-4-turbo-preview"
load_dotenv()
set_llm_cache(SQLiteCache(database_path=".langchain.db"))


def load_wandb_ds(ds_name):
    api = wandb.Api()
    test_dataset = api.artifact(ds_name)
    test_dataset.logged_by()
    json_file = json.load(test_dataset.files()[0].download(replace=True))
    json_config = json.loads(test_dataset.logged_by().json_config)
    stf_ratio = json_config["sft_ratio"]["value"]
    dpo_gen = json_config["dpo_gen"]["value"]
    checkpoint = (
        json_config["sem_model"]["value"]["checkpoint"].split("run-")[1].split("-")[0]
        if json_config["sem_model"]["value"]["checkpoint"]
        else None
    )
    return (
        pd.DataFrame(data=json_file["data"], columns=json_file["columns"]),
        stf_ratio,
        dpo_gen,
        checkpoint,
    )


def get_chain():
    # Schema based on https://academic.oup.com/jamia/advance-article/doi/10.1093/jamia/ocad259/7590607
    def get_iob_tokens(_dict):
        # Tokenize the text into words

        entities = [
            ents
            for ents in _dict["entities"]
            if isinstance(ents, dict) and "type" in ents.keys() and "entity" in ents.keys()
        ]

        text = _dict["text"]

        words = text.split(" ")
        # Initialize an empty list for storing the IOB tokens
        iob_tokens = []

        # Iterate over each word in the text
        for word in words:
            # Initialize the IOB token for the current word as 'O'
            iob_token = "O"

            # Check if the word is part of any entity
            for entity in entities:
                if word in entity["entity"].split():
                    # If the word is the start of the entity,
                    # the IOB token should be 'B-' followed by the entity type
                    if word == entity["entity"].split()[0]:
                        iob_token = "B-" + entity["type"]
                    # If the word is inside the entity but not the start,
                    # the IOB token should be 'I-' followed by the entity type
                    else:
                        iob_token = "I-" + entity["type"]
                    # Once we found a matching entity, we can break the loop
                    break

            # Add the IOB token to the list
            iob_tokens.append(iob_token)
        # remove Inner entities that are not part of any entity
        for i, token in enumerate(iob_tokens):
            if "I-" in token and iob_tokens[i - 1] == "O":
                iob_tokens[i] = "B-" + token.split("-")[1]
        return {"words": words, "entities": iob_tokens, "score": _dict["score"]}

    with open("data/prompt.txt") as f:
        text = f.read()

    model = AzureChatOpenAI(
        model=os.environ["DEPLOYMENT_NAME"],
    )
    # model = ChatGroq(
    #    temperature=0,
    #    groq_api_key=os.getenv("GROQ_API_KEY"),
    #    model_name="llama3-70b-8192",
    # )

    template = ChatPromptTemplate.from_messages(
        [
            ("human", text),
        ]
    )

    parser = JsonOutputParser()
    fix_parser = OutputFixingParser.from_llm(model, parser)
    return {
        "entities": template
        | model
        | RunnableLambda(
            lambda x: (
                "[" + x.content.split("[")[1].split("]")[0] + "]" if "[" in x.content else "[]"
            )
        )
        | fix_parser,
        "text": itemgetter("input"),
        "score": itemgetter("score"),
    } | RunnableLambda(get_iob_tokens)


def main(dataset: str, gold: bool = False, test: bool = False):
    chain = get_chain()

    ds, sft_ratio, dpo_gen, checkpoint = (
        load_wandb_ds(dataset) if not test else load_wandb_ds(dataset.replace("-gen", "-test"))
    )
    dataset_id = (
        (f"{sft_ratio}-{dpo_gen}-{checkpoint}" if checkpoint else f"{sft_ratio}-{dpo_gen}")
        if not gold
        else "gold"
    )

    cols = len(list(ds.filter(regex="generation.*").columns))
    ds = (
        ds[
            [
                "generation_0",
                "generation_1",
                "generation_2",
                "generation_3",
                "eval_sem_scores_0",
                "eval_sem_scores_1",
                "eval_sem_scores_2",
                "eval_sem_scores_3",
            ]
        ]
        if not gold
        else ds[["ground_texts"]]
    )

    ann_ds = []
    if not gold:
        for idx, row in tqdm(ds.iterrows(), total=ds.shape[0]):
            for i in range(cols):
                ann_ds.append(
                    {
                        "text": row[f"generation_{i}"],
                        "score": row[f"eval_sem_scores_{i}"],
                    }
                )
    else:
        for idx, row in tqdm(ds.iterrows(), total=ds.shape[0]):
            ann_ds.append({"text": row["ground_texts"], "score": 1})

    ann_ds = pd.DataFrame(ann_ds)

    class BatchCallback(BaseCallbackHandler):
        def __init__(self, total: int):
            super().__init__()
            self.count = 0
            self.progress_bar = tqdm(total=total)  # define a progress bar

        # Override on_llm_end method. This is called after every response from LLM
        def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id=None, **kwargs):
            self.count += 1
            self.progress_bar.update(1)

        def __enter__(self):
            self.progress_bar.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, exc_traceback):
            self.progress_bar.__exit__(exc_type, exc_value, exc_traceback)

        def __del__(self):
            self.progress_bar.__del__()

    with BatchCallback(len(ann_ds)) as cb:  # init callback
        output = asyncio.run(
            chain.abatch(
                [
                    {"input": d, "score": s}
                    for d, s in zip(ann_ds["text"].tolist(), ann_ds["score"].tolist())
                ],
                config={"callbacks": [cb], "max_concurrency": 20},
            )
        )
    ds = pd.DataFrame(output)
    ds.to_parquet(f"data/ner-{dataset_id}-{'train' if not test else 'test'}.parquet")


if __name__ == "__main__":
    typer.run(main)
