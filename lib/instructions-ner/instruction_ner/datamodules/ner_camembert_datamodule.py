import tempfile
import zipfile
from glob import glob
from pathlib import Path
from typing import Generator, Optional

import torch
from ariadne.contrib.inception_util import SENTENCE_TYPE, TOKEN_TYPE
from cassis import Cas, load_cas_from_xmi, load_typesystem
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, DataCollatorForTokenClassification

CUSTOM_SPAN = "custom.Span"


class NerCamembertDataModule(LightningDataModule):
    def __init__(
        self,
        num_workers=0,
        data_dir=None,
        batch_size: int = 16,
        pin_memory: bool = False,
        k_fold: int = 5,
        c_fold: int = 0,
        datasets: list = [],
        architecture: str = "camembert_base",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        self.datasets = [data_dir + dataset for dataset in datasets]
        self.batch_size = batch_size
        self.json_full_dataset: list = []
        self.full_dataset = None
        self.train_dataset = None
        self.eval_dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.architecture)
        self.weights: dict = {}
        self.labels2id: dict = {}
        self.id2labels: dict = {}

    def prepare_data(self):
        for file_path in self.datasets:
            for file in self.uncompress_inception_dataset(file_path):
                cas = self.load_cas(file["type_system"], file["cas_annotation"])
                for sentence in cas.select(SENTENCE_TYPE):
                    tokens = []
                    annotations = []
                    cas_tokens = cas.select_covered(TOKEN_TYPE, sentence)
                    cas_annotations = list(cas.select_covered(CUSTOM_SPAN, sentence))
                    sentence_text = sentence.get_covered_text()

                    for cas_token in cas_tokens:
                        tokens.append(
                            {
                                "text": cas_token.get_covered_text(),
                                "start": cas_token.begin,
                                "end": cas_token.end,
                            }
                        )

                    for cas_annotation in cas_annotations:
                        if cas_annotation.label:
                            annotations.append(
                                {
                                    "start": cas_annotation.begin,
                                    "end": cas_annotation.end,
                                    "label": cas_annotation.label,
                                }
                            )

                    self.json_full_dataset.append(
                        {"tokens": tokens, "annotations": annotations, "text": sentence_text}
                    )

    def setup(self, stage: Optional[str] = None):
        all_tokens: list = []
        all_tags: list = []
        all_tag_ids: list = []
        # Create label2id
        labels = list(
            set(
                [
                    label["label"]
                    for example in self.json_full_dataset
                    for label in example["annotations"]
                ]
            )
        )
        iob_labels = ["B-" + label for label in labels] + ["I-" + label for label in labels] + ["O"]
        self.labels2id = {label: id_label for id_label, label in enumerate(iob_labels)}
        self.id2labels = {id_label: label for id_label, label in enumerate(iob_labels)}

        for example in self.json_full_dataset:
            tags = []
            words = []

            annotations = example["annotations"]

            prev_tag = "O"
            for token in example["tokens"]:
                is_inside = False
                label = None
                for annotation in annotations:
                    label = annotation["label"]
                    if annotation["start"] <= token["start"] < annotation["end"]:
                        is_inside = True
                        break

                # Check that the NER annotation has been made (ie is not None) and that the label
                # belongs to the model classes.
                if not is_inside or label is None or label not in labels:
                    tag = "O"
                elif prev_tag == f"B-{label}":
                    tag = f"I-{label}"
                elif prev_tag == f"I-{label}":
                    tag = f"I-{label}"
                else:
                    tag = f"B-{label}"

                prev_tag = tag

                tags.append(tag)
                words.append(token["text"])
                if len(tags) != len(words):
                    raise ValueError("💔 between len of tokens & labels.")

            all_tokens.append(words)
            all_tags.append(tags)
            all_tag_ids.append([self.labels2id[tag] for tag in tags])

        y = [tag for tags in all_tag_ids for tag in tags]
        self.weights = torch.Tensor(
            compute_class_weight(
                class_weight="balanced",
                classes=list(set(y)),
                y=y,
            )
        )

        dataset = {
            "ids": [i for i in list(range(0, len(all_tokens)))],
            "tokens": all_tokens,
            "ner_tags": all_tags,
            "ner_tag_ids": all_tag_ids,
        }

        def tokenize_dataset(examples: Dataset) -> Dataset:
            """Tokenize and align the labels over the entire dataset"""
            return self.tokenize_and_align_labels(examples, self.tokenizer)

        self.full_dataset = Dataset.from_dict(dataset).map(
            tokenize_dataset,
            batched=True,
            remove_columns=["tokens", "ner_tags", "ids", "ner_tag_ids"],
        )
        k_fold = KFold(n_splits=self.hparams.k_fold, shuffle=True)
        data_train_ids, data_val_ids = list(
            k_fold.split(self.full_dataset),
        )[self.hparams.c_fold]
        self.train_dataset = Subset(self.full_dataset, data_train_ids.tolist())
        self.eval_dataset = Subset(self.full_dataset, data_val_ids.tolist())

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=DataCollatorForTokenClassification(tokenizer=self.tokenizer),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=32,
            collate_fn=DataCollatorForTokenClassification(tokenizer=self.tokenizer),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    @staticmethod
    def uncompress_inception_dataset(zip_path: str) -> Generator:
        """It takes a zip file, extracts it to a temporary directory,
        then extracts the zip files in the curation folder to the
        curation folder, and returns a generator of
        dictionaries containing the paths to the type system and the CAS annotation files.
        Args:
            zip_path: The path to the zip file containing the dataset
        """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir_name)
                curation_folder = tmp_dir_name + "/curation"
                for zip_subset_path in glob(curation_folder + "/**/*.zip"):
                    with zipfile.ZipFile(zip_subset_path) as zip_subset:
                        subset_folder = str(Path(zip_subset_path).parent)
                        zip_subset.extractall(subset_folder)
                        yield {
                            "type_system": glob(subset_folder + "/*.xml")[0],
                            "cas_annotation": glob(subset_folder + "/*.xmi")[0],
                        }

    @staticmethod
    def load_cas(ts_path: str, cas_path: str) -> Cas:
        """It loads a CAS from a pair of files, one containing
        the type system and the other containing the CAS
        Args:
            ts_path: The path to the typesystem file
            cas_path: The path to the XMI file that contains the CAS
        Returns:
            A Cas object
        """
        with open(ts_path, "rb") as file:
            typesystem = load_typesystem(file)
        with open(cas_path, "rb") as file:
            cas = load_cas_from_xmi(file, typesystem=typesystem)
        return cas

    @staticmethod
    def tokenize_and_align_labels(
        examples, tokenizer: AutoTokenizer, dict_key: str = "tokens"
    ) -> Dataset:
        """Tokenize examples, realign the tokens and labels, and truncate sequences to be no longer
        than maximum input length.

        Add the special tokens [CLS] and [SEP] and subword tokenization creates a mismatch
        between the input and labels.
        A single word corresponding to a single label may be split into two subwords.
        This function realigns the tokens and labels by:
        - Mapping all tokens to their corresponding word with the word_ids method.
        - Assigning the label -100 to the special tokens [CLS] and [SEP]
          so the PyTorch loss function ignores them.
        - Only labeling the first token of a given word.
          Assign -100 to other subtokens from the same word.

        Args:
            dict_key: the key inside dataset
            examples: Dataset to tokenize.
            tokenizer: The tokenizer used to preprocess the data.

        Returns:
            Tokenized dataset for token classification task, with data and labels.
        """
        tokenized_inputs = tokenizer(examples[dict_key], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["ner_tag_ids"]):
            # Map tokens to their respective word.
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            # Set the special tokens to -100.
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                # Only label the first token of a given word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
