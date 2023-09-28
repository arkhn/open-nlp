import datasets
import spacy
import torch
import wandb
from spacy.tokens import Doc
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
nlp = spacy.blank("fr")

"""
The following scripts are used to analyze the E3C dataset and the E3C-LLM dataset specifically
for the French layer. We analyze the number of sentences, tokens and BI tags in the dataset,
and we render the NER tags to compare the InstructGPT predictions with the "ground truth".
Also we compute the confusion matrix to see the annotation differences between the two datasets.
"""


def main():
    # set up wandb
    run = wandb.init(
        project="weak-supervision-instructgpt-e3c", group="analytics_french", job_type="analytics"
    )
    # load datasets
    e3c_dataset = (
        datasets.load_dataset("bio-datasets/e3c")
        .map(tokenize_and_align_labels, batched=True)
        .with_format(columns=["input_ids", "attention_mask", "labels"])
    )
    e3c_llm_dataset = (
        datasets.load_dataset("bio-datasets/e3c-llm")
        .map(map_offset_to_text, batched=True)
        .map(tokenize_and_align_labels, batched=True)
        .with_format(columns=["input_ids", "attention_mask", "labels"])
    )

    # get e3c-llm labels and convert it as a tensor
    e3c_llm_labels = e3c_llm_dataset["fr.layer1"]["labels"]
    e3c_llm_labels = torch.tensor([entity for examples in e3c_llm_labels for entity in examples])
    e3c_llm_labels[e3c_llm_labels == -100] = 0

    # instructgpt_performance(e3c_dataset, e3c_llm_labels, run)
    # render_quantities_table(e3c_dataset, e3c_llm_dataset, run)
    render_ner_clustering(
        e3c_dataset,
        e3c_llm_dataset,
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
    # render_ner_html_tags(e3c_dataset, e3c_llm_dataset, run)
    # render_confusion_matrix(e3c_dataset, e3c_llm_dataset)


def render_ner_clustering(e3c_dataset, e3c_llm_dataset, run, clusters, table_names):
    for layer in ["fr.layer2", "fr.layer2.validation"]:
        for cluster, table_name in zip(clusters, table_names):
            texts = e3c_dataset[layer].filter(
                lambda x, x_idx: any(
                    (torch.tensor(x["labels"]) == cluster[0])
                    & (torch.tensor(e3c_llm_dataset[layer][x_idx]["labels"]) == cluster[1])
                ),
                with_indices=True,
            )["text"]
            render_ner_text(
                e3c_dataset,
                e3c_llm_dataset,
                layer,
                run,
                texts,
                table_name,
                merge=False,
            )


def render_confusion_matrix(e3c_dataset, e3c_llm_dataset):
    # log confusion matrix
    for layer in ["fr.layer2", "fr.layer2.validation"]:
        ground_truth = torch.LongTensor(
            [tag for sentence in e3c_dataset[layer]["labels"] for tag in sentence]
        )
        predictions = torch.LongTensor(
            [tag for sentence in e3c_llm_dataset[layer]["labels"] for tag in sentence]
        )
        predictions = predictions.tolist()
        predictions = [prediction for prediction in predictions if prediction != -100]
        ground_truth = ground_truth.tolist()
        ground_truth = [tag for tag in ground_truth if tag != -100]
        class_names = e3c_dataset[layer].features["clinical_entity_tags"].feature.names
        wandb.log(
            {
                f"{layer}/confusion_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=ground_truth,
                    preds=predictions,
                    class_names=class_names,
                )
            }
        )


def render_ner(e3c_dataset, e3c_llm_dataset, run):
    for layer in ["fr.layer2", "fr.layer2.validation"]:
        render_ner_text(
            e3c_dataset,
            e3c_llm_dataset,
            layer=layer,
            run=run,
            texts=e3c_dataset[layer]["text"],
            table_name=f"{layer}/ner",
        )


def render_ner_text(
    e3c_dataset, e3c_llm_dataset, layer, run, texts, table_name, merge: bool = True
):
    data_tags = []
    for text in texts:
        data_tags.append(
            [
                render_html(
                    e3c_dataset, layer, e3c_dataset[layer]["text"].index(text), merge=merge
                ),
                render_html(
                    e3c_llm_dataset, layer, e3c_llm_dataset[layer]["text"].index(text), merge=merge
                ),
            ]
        )
    run.log({f"{layer}/{table_name}": wandb.Table(data=data_tags, columns=["e3c", "e3c-llm"])})


def render_quantities_table(e3c_dataset, e3c_llm_dataset, run):
    # get the number of sentences, tokens and BI tags
    data = []
    for dataset in [e3c_dataset, e3c_llm_dataset]:
        for layer in ["fr.layer2", "fr.layer1", "fr.layer2.validation"]:
            if layer in dataset.column_names.keys():
                data.append(
                    [
                        f"{dataset[layer].builder_name}/{layer}",
                        dataset[layer].num_rows,
                        get_tokens_number(dataset[layer]),
                        get_bi_tags_number(dataset[layer]),
                        get_tag_number(dataset[layer], 1),
                        get_tag_number(dataset[layer], 2),
                    ]
                )
    table = wandb.Table(
        columns=[
            "name",
            "sentences_number",
            "tokens_number",
            "BI_tags_number",
            "B_tag_number",
            "I_tag_number",
        ],
        data=data,
    )
    run.log({"e3c": table})


def instructgpt_performance(e3c_dataset, e3c_llm_labels, run):
    # get e3c labels and convert it as a tensor
    e3c_labels = e3c_dataset["fr.layer1"]["labels"]
    e3c_labels = torch.tensor([entity for examples in e3c_labels for entity in examples])
    # use torchmetrics to get the f1 score, the recall and the precision no aggregated
    no_agg_scores = MetricCollection(
        MulticlassF1Score(average=None, num_classes=3, ignore_index=-100),
        MulticlassPrecision(average=None, num_classes=3, ignore_index=-100),
        MulticlassRecall(average=None, num_classes=3, ignore_index=-100),
    )
    no_agg_scores.update(e3c_llm_labels, e3c_labels)
    no_aggregated_metrics = no_agg_scores.compute()
    # log the metrics
    dict_metrics = {
        f"test/{log_key}/{label}": log_value[idx_label]
        for idx_label, label in enumerate(
            e3c_dataset["fr.layer1"].features["clinical_entity_tags"].feature.names
        )
        for log_key, log_value in no_aggregated_metrics.items()
    }
    run.log(dict_metrics)
    # use torchmetrics to get the f1 score, the recall and the precision agragated in
    # a macro average
    f1_macro = MulticlassF1Score(num_classes=3, average="macro", ignore_index=-100)
    f1_macro.update(e3c_llm_labels, e3c_labels)
    f1_score_macro = f1_macro.compute()
    recall_macro = MulticlassRecall(num_classes=3, average="macro", ignore_index=-100)
    recall_macro.update(e3c_llm_labels, e3c_labels)
    recall_score_macro = recall_macro.compute()
    precision_macro = MulticlassPrecision(num_classes=3, average="macro", ignore_index=-100)
    precision_macro.update(e3c_llm_labels, e3c_labels)
    precision_score_macro = precision_macro.compute()
    # log the metrics
    run.log(
        {
            "test/tokens/MulticlassF1Score": f1_score_macro,
            "test/tokens/MulticlassPrecision": precision_score_macro,
            "test/tokens/MulticlassRecall": recall_score_macro,
        }
    )


def render_html(
    dataset: datasets.DatasetDict, layer: str, text_idx: int, merge: bool = False
) -> wandb.Html:
    """Render the NER tags of a given text.

    Args:
        dataset: A given e3c dataset.
        layer: A given e3c layer.
        text_idx: The index of the text in the layer.

    Returns:
        A wandb.Html object. Contains the rendered NER tags using spacy.
    """
    offset = dataset[layer]["tokens_offsets"][text_idx]
    spaces = [True] * len(offset)
    last_end = -1
    for o_idx, o in enumerate(offset):
        if last_end == o[0]:
            spaces[o_idx - 1] = False
        last_end = o[1]

    if merge:
        doc = Doc(
            nlp.vocab,
            words=dataset[layer]["tokens"][text_idx],
            spaces=spaces,
            ents=[
                f'{dataset[layer].features["clinical_entity_tags"].feature.int2str(entity)}'
                for entity in dataset[layer]["clinical_entity_tags"][text_idx]
            ],
        )
        html = spacy.displacy.render(doc, style="ent", page=True, minify=True)
    else:
        doc = Doc(
            nlp.vocab,
            words=dataset[layer]["tokens"][text_idx],
            spaces=spaces,
            ents=[
                f'B-{dataset[layer].features["clinical_entity_tags"].feature.int2str(entity)}'
                for entity in dataset[layer]["clinical_entity_tags"][text_idx]
            ],
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


def get_bi_tags_number(layer: dict) -> int:
    """Get the number of BI tags in a layer.

    Args:
        layer: A given e3c layer.

    Returns:
        The number of BI tags in the layer.
    """

    tokens = [
        [text[offset[0] : offset[1]] for offset, tag in zip(token_offset, tags) if tag != 0]
        for text, token_offset, tags in zip(
            layer["text"], layer["tokens_offsets"], layer["clinical_entity_tags"]
        )
    ]
    tokens = [token for token_list in tokens for token in token_list]
    return len(tokens)


def get_tag_number(layer: datasets.Dataset, tag_id: int) -> int:
    """Get the number of a specific tag in a layer.

    Args:
        layer: A given e3c layer.
        tag_id: A given tag id.

    Returns:
        The number of BI tags in the layer.
    """

    tokens = [
        [text[offset[0] : offset[1]] for offset, tag in zip(token_offset, tags) if tag == tag_id]
        for text, token_offset, tags in zip(
            layer["text"], layer["tokens_offsets"], layer["clinical_entity_tags"]
        )
    ]
    tokens = [token for token_list in tokens for token in token_list]
    return len(tokens)


def get_tokens_number(layer: datasets.DatasetDict) -> int:
    """Get the number of tokens in a layer.

    Args:
        layer: A given e3c layer.

    Returns:
        The number of tokens in the layer.
    """
    return torch.LongTensor(
        [token for sentence in layer["clinical_entity_tags"] for token in sentence]
    ).shape[0]


def tokenize_and_align_labels(examples: dict) -> dict:
    """Tokenize the text and align the labels with the sub-tokens.

    Args:
        examples: A dictionary containing the text and the labels.

    Returns:
        A dictionary containing the tokenized text and the labels.
    """

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


if __name__ == "__main__":
    main()
