import logging

import hydra
import spacy
from omegaconf import DictConfig
from src.hub import get_ner_bigbio_datasets
from src.preprocess import load_and_concat_datasets, preprocess_row
from src.tasks import generate_instruction_ner

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_name="default", config_path="conf", version_base="1.2")
def main(config: DictConfig):
    """
    Main function to load, process, concatenate and format bigbio NER datasets.
    Resulting dataset is upload to Hugging Face Hub.

    Args:
        config (DictConfig): Hydra config object
    """
    dataset_names = get_ner_bigbio_datasets()

    # Initialize an empty list to store individual datasets
    big_dataset = load_and_concat_datasets(config, dataset_names, logger)

    # Initialize the spaCy tokenizer
    nlp = spacy.load(config.tokenizer, disable=["tagger", "parser", "ner"])

    # filter rows with no text:
    big_dataset = big_dataset.filter(
        lambda x: "passages" in x and len(x["passages"]) > 0 and len(x["passages"][0]["text"]) > 0
    )

    # Update the features of the big_dataset
    big_dataset = big_dataset.map(preprocess_row, fn_kwargs={"nlp": nlp})

    # filter rows with ner_tags that are all O
    big_dataset = big_dataset.filter(lambda x: any(tag != "O" for tag in x["ner_tags"]))

    # shuffle rows
    big_dataset = big_dataset.shuffle(seed=42)

    # Remove unnecessary columns
    big_dataset = big_dataset.remove_columns(
        ["document_id", "passages", "entities", "events", "coreferences", "relations"]
    )

    # Update the features of the big_dataset
    big_dataset = big_dataset.map(
        generate_instruction_ner,
        batched=True,
        remove_columns=[
            "id",
            "tokenized_sentence",
            "ner_tags",
            "sentence",
            "entities_types",
        ],
        # load_from_cache_file=False,
    )

    big_dataset = big_dataset.flatten()

    # rename columns by remove the "instructions" prefix
    columns_names = big_dataset.column_names
    for column_name in columns_names:
        if column_name.startswith("instructions"):
            new_column_name = column_name.replace("instructions.", "")
            big_dataset = big_dataset.rename_column(column_name, new_column_name)

    big_dataset.push_to_hub(config.hf_hub_repo_name)


if __name__ == "__main__":
    main()
