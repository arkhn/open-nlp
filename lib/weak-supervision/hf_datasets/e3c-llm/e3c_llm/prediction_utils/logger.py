import ast
import os
import pathlib

import datasets
import hydra
import pandas as pd
import spacy
import torch
import wandb
from omegaconf import DictConfig, omegaconf
from spacy.tokens import Doc
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from transformers import AutoTokenizer

data_path = pathlib.Path(__file__).parents[1] / "data"
nlp = spacy.blank("fr")


def wandb_logger(
    cfg,
    layer,
    prediction_path,
    wandb_run_name="test",
):
    run, e3c_dataset = init(name=wandb_run_name, cfg=cfg)
    prediction_dataset = datasets.Dataset.from_csv(
        os.path.join(data_path, prediction_path),
    )

    prediction_df = pd.read_csv(
        os.path.join(data_path, prediction_path),
    ).isna()
    if (
        malformed_examples := len(prediction_df[prediction_df["labels"] == True])  # noqa: E712
    ) > 0:
        wandb.termwarn(f"Found {malformed_examples} malformed examples !")

    prediction_dataset = prediction_dataset.filter(
        lambda examples: examples["labels"] is not None
        and examples["entities_offsets"] is not None
        and examples["text"] is not None
    )

    prediction_dataset = (
        prediction_dataset.map(
            lambda x: {"tokens_offsets": ast.literal_eval(x["entities_offsets"])}
        )
        .map(
            lambda x: {
                "clinical_entity_tags": [
                    e3c_dataset[layer].features["clinical_entity_tags"].feature.str2int(token)
                    for token in ast.literal_eval(x["labels"])
                ]
            }
        )
        .map(map_offset_to_text, batched=True)
        .map(tokenize_and_align_labels, batched=True)
        .sort("text")
    )
    prediction_labels = prediction_dataset["labels"]
    prediction_labels = torch.tensor(
        [entity for examples in prediction_labels for entity in examples]
    )
    prediction_labels[prediction_labels == -100] = 0
    feature_name = e3c_dataset["fr.layer1"].features["clinical_entity_tags"].feature.names
    e3c_dataset = e3c_dataset[layer].filter(lambda x: x["text"] in prediction_dataset["text"])
    if len(e3c_dataset) != len(prediction_dataset):
        dataframe = pd.DataFrame(e3c_dataset)
        dataframe.drop_duplicates(subset="input_ids", inplace=True)
        e3c_dataset = datasets.Dataset.from_pandas(dataframe)
    e3c_dataset = e3c_dataset.sort("text")
    e3c_labels = e3c_dataset["labels"]
    e3c_labels = torch.tensor([entity for examples in e3c_labels for entity in examples])

    # use torchmetrics to get the f1 score, the recall and the precision no aggregated
    no_agg_scores = MetricCollection(
        MulticlassF1Score(average=None, num_classes=3, ignore_index=-100),
        MulticlassPrecision(average=None, num_classes=3, ignore_index=-100),
        MulticlassRecall(average=None, num_classes=3, ignore_index=-100),
    )
    no_agg_scores.update(prediction_labels, e3c_labels)
    no_aggregated_metrics = no_agg_scores.compute()
    # log the metrics
    dict_metrics = {
        f"test/{log_key}/{label}": log_value[idx_label]
        for idx_label, label in enumerate(feature_name)
        for log_key, log_value in no_aggregated_metrics.items()
    }
    run.log(dict_metrics)
    # use torchmetrics to get the f1 score, the recall and the precision aggregated in
    # a macro average
    f1_macro = MulticlassF1Score(num_classes=3, average="macro", ignore_index=-100)
    f1_macro.update(prediction_labels, e3c_labels)
    f1_score_macro = f1_macro.compute()
    recall_macro = MulticlassRecall(num_classes=3, average="macro", ignore_index=-100)
    recall_macro.update(prediction_labels, e3c_labels)
    recall_score_macro = recall_macro.compute()
    precision_macro = MulticlassPrecision(num_classes=3, average="macro", ignore_index=-100)
    precision_macro.update(prediction_labels, e3c_labels)
    precision_score_macro = precision_macro.compute()
    run.log(
        {
            "test/tokens/MulticlassF1Score": f1_score_macro,
            "test/tokens/MulticlassPrecision": precision_score_macro,
            "test/tokens/MulticlassRecall": recall_score_macro,
        }
    )
    # log the metrics
    render_ner_clustering(
        e3c_dataset,
        prediction_dataset,
        run,
        [
            [0, 2],  # O -> I
            [0, 1],  # O -> B
            [2, 1],  # I -> B
            [1, 2],  # B -> I
        ],
        [
            "ner_clustering/O_to_I",
            "ner_clustering/O_to_B",
            "ner_clustering/I_to_B",
            "ner_clustering/B_to_I",
        ],
    )
    wandb.finish()


def render_ner_clustering(e3c_dataset, e3c_llm_dataset, run, clusters, table_names):
    for cluster, table_name in zip(clusters, table_names):
        texts = e3c_dataset.filter(
            lambda x, x_idx: any(
                (torch.tensor(x["labels"]) == cluster[0])
                & (torch.tensor(e3c_llm_dataset[x_idx]["labels"]) == cluster[1])
            ),
            with_indices=True,
        )["text"]
        render_ner_text(
            e3c_dataset,
            e3c_llm_dataset,
            run,
            texts,
            table_name,
        )


def render_ner_text(e3c_dataset, e3c_llm_dataset, run, texts, table_name):
    data_tags = []
    for text in texts:
        data_tags.append(
            [
                render_html(
                    e3c_dataset,
                    e3c_dataset["text"].index(text),
                    e3c_dataset.features["clinical_entity_tags"].feature,
                ),
                render_html(
                    e3c_llm_dataset,
                    e3c_llm_dataset["text"].index(text),
                    e3c_dataset.features["clinical_entity_tags"].feature,
                ),
            ]
        )
    run.log({f"{table_name}": wandb.Table(data=data_tags, columns=["e3c", "e3c-llm"])})


def tokenize_and_align_labels(examples: dict) -> dict:
    """Tokenize the text and align the labels with the sub-tokens.

    Args:
        examples: A dictionary containing the text and the labels.

    Returns:
        A dictionary containing the tokenized text and the labels.
    """

    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["clinical_entity_tags"]):
        # Map tokens to their respective word.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def map_offset_to_text(examples: dict) -> dict:
    """Map the offsets to the text. To compute the tokens.

    Args:
        examples: the examples to map.

    Returns: return the tokens of the text in a dict.
    """
    return {
        "tokens": [
            [text[offset[0] : offset[1]] for offset in offsets]
            for text, offsets in zip(examples["text"], examples["tokens_offsets"])
        ]
    }


def init(name, cfg):
    """Initialize the wandb run and the datasets.

    Args:
        name: The name of the run.

    Returns:
        The wandb run and the datasets.
    """
    run = wandb.init(
        project="weak-supervision-instructgpt-e3c",
        group="weak-annotation",
        job_type="analytics",
        name=name,
    )
    container = omegaconf.OmegaConf.to_container(cfg)
    container["str_static_examples"] = str(container["static_examples"])
    run.config.update(container)

    # load datasets
    e3c_dataset = (
        datasets.load_dataset("bio-datasets/e3c").sort("text").map(map_offset_to_text, batched=True)
    )

    e3c_dataset = e3c_dataset.map(tokenize_and_align_labels, batched=True).with_format(
        columns=["text", "input_ids", "attention_mask", "labels"]
    )
    return run, e3c_dataset


def render_html(dataset: datasets.DatasetDict, text_idx: int, feature) -> wandb.Html:
    """Render the NER tags of a given text.

    Args:
        dataset: A given e3c dataset.
        layer: A given e3c layer.
        text_idx: The index of the text in the layer.

    Returns:
        A wandb.Html object. Contains the rendered NER tags using spacy.
    """
    offset = dataset["tokens_offsets"][text_idx]
    spaces = [True] * len(offset)
    last_end = -1
    for o_idx, o in enumerate(offset):
        if last_end == o[0]:
            spaces[o_idx - 1] = False
        last_end = o[1]

    words = [token for token in dataset["tokens"][text_idx] if token != ""]
    spaces = [space for token, space in zip(dataset["tokens"][text_idx], spaces) if token != ""]
    entities = [
        entity
        for token, entity in zip(
            dataset["tokens"][text_idx], dataset["clinical_entity_tags"][text_idx]
        )
        if token != ""
    ]
    doc = Doc(
        nlp.vocab,
        words=words,
        spaces=spaces,
        ents=[f"B-{feature.int2str(entity)}" for entity in entities],
    )
    html = spacy.displacy.render(
        doc,
        style="ent",
        page=True,
        minify=True,
        options={
            "colors": {
                "B-CLINENTITY": "#28eba8",
                "I-CLINENTITY": "#cb3feb",
            },
        },
    )
    return wandb.Html(html)


@hydra.main(version_base="1.3", config_path="../configs", config_name="logger.yaml")
def main(cfg: DictConfig):
    """Create few-shot examples for a given layer.

    This method is used to create few-shot examples in a prompt format. The aim is to use this
    examples to guide a large language model in an extract task.

    Args:
        cfg: Hydra configuration.
    """
    wandb_logger(
        cfg=cfg,
        layer=cfg.layer,
        prediction_path=cfg.prediction_path,
        wandb_run_name=cfg.wandb_run_name,
    )


if __name__ == "__main__":
    main()
