from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm


def preprocess_row(row, nlp):
    """
    Processes a single row of a dataset.
    Tokenizes the 'text' field, generates NER tags, and merges entities.
    Returns the processed row as a dictionary.

    Args:
        row (dict): A single row of a dataset
        nlp (spacy.lang): A spaCy tokenizer

    Returns:
        dict: A dictionary containing the processed row
    """
    text = row["passages"][0]["text"][0]
    doc = nlp(text)
    tokens = [token.text for token in doc]
    offsets_mapping = [(token.idx, token.idx + len(token)) for token in doc]
    ner_tags = ["O"] * len(tokens)
    types = []

    # Sort entities by length (longest first)
    sorted_entities = sorted(
        row["entities"],
        key=lambda x: x["offsets"][0][1] - x["offsets"][0][0],
        reverse=True,
    )

    for entity in sorted_entities:
        try:
            entity_type = entity["type"]
            if entity_type not in types:
                types.append(entity_type)
            entity_offsets = entity["offsets"][0]
            start_offset, end_offset = entity_offsets

            start_idx = next(
                (idx for idx, offset in enumerate(offsets_mapping) if offset[0] == start_offset),
                None,
            )
            end_idx = next(
                (idx for idx, offset in enumerate(offsets_mapping) if offset[1] == end_offset),
                None,
            )

            if start_idx is not None and end_idx is not None:
                # Skip if the entity overlaps with a longer entity
                if any(tag != "O" for tag in ner_tags[start_idx : end_idx + 1]):
                    continue

                ner_tags[start_idx] = f"B-{entity_type}"
                for idx in range(start_idx + 1, end_idx + 1):
                    ner_tags[idx] = f"I-{entity_type}"
        except ValueError:
            print(f"Error processing entity: {entity}")

    return {
        "id": row["id"],
        "tokenized_sentence": tokens,
        "ner_tags": ner_tags,
        "sentence": text,
        "entities_types": types,
    }


def load_and_concat_datasets(config, dataset_names, logger):
    """
    Loads and concatenates all datasets in dataset_names.

    Args:
        config (DictConfig): Hydra config object
        dataset_names (list): List of dataset names
        logger (logging.Logger): Logger object

    Returns:
        datasets.Dataset: Concatenated dataset
    """
    datasets = []
    # TODO: replace by a list of excluded datasets from hydra config
    excluded_datasets = ["pubtator_central"]
    for dataset_name in tqdm(dataset_names):
        # exclude datasets that are too big
        if dataset_name in excluded_datasets:
            continue
        try:
            dataset = load_dataset(f"bigbio/{dataset_name}", name=f"{dataset_name}_bigbio_kb")
            datasets.append(dataset["train"])
        except Exception as e:
            logger.warning(f"Could not load {dataset_name}")
            logger.warning(e)

    # Concatenate all datasets into a single dataset
    big_dataset = concatenate_datasets(datasets)
    return big_dataset
