from typing import Optional

import datasets
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, DataCollatorForTokenClassification


class E3CDataModule(LightningDataModule):
    """Basic e3c dataset with only instructGPT annotations or dictionary extraction annotations.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
        tokenizer: str = "camembert-base",
        layer: str = "fr.layer2",
        instructgpt_ws: bool = True,
        language: str = "fr",
        fold: int = 0,
    ):
        """Initialize a DataModule.

        Args:
            batch_size: the batch size to use for the dataloaders.
            num_workers: the number of workers to use for loading data.
            pin_memory: whether to copy Tensors into CUDA pinned memory.
            tokenizer: the tokenizer to use associated with the model.
            layer: the layer to use for the E3C dataset.
                In this format: "{language}.layer{layer number}".
            instructgpt_ws: whether to use the instructgpt_ws dataset.
            language: the language to use for the E3C dataset.
            fold: The current fold distribution . The dataset is split in k folds.
                This fold hyperparameter is used to select the n-th fold as a validation set and
                the other as a training set.
        """

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.labels: Optional[list] = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer, add_prefix_space=True
        )

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        This method setup the different huggingface datasets and the k-fold split.
        Several mappings are applied to the datasets to tokenize and align the labels.

        Args:
            stage: the stage to setup. Can be either 'fit' or 'test'.
        """
        e3c_dataset = (
            datasets.load_dataset("bio-datasets/e3c")
            .map(self.tokenize_and_align_labels, batched=True)
            .with_format(columns=["input_ids", "attention_mask", "labels"])
        )

        if self.hparams.instructgpt_ws:
            e3c_llm_dataset = (
                datasets.load_dataset("bio-datasets/e3c-llm")
                .map(self.map_offset_to_text, batched=True)
                .map(self.tokenize_and_align_labels, batched=True)
                .with_format(columns=["input_ids", "attention_mask", "labels"])
            )
            train_set = e3c_llm_dataset[f"{self.hparams.language}_layer2"]
        else:
            train_set = e3c_dataset[f"{self.hparams.language}.layer2"]
        test_set = e3c_dataset[f"{self.hparams.language}.layer1"]

        k_fold = KFold(n_splits=5, shuffle=True)
        self.labels = e3c_dataset[f"{self.hparams.language}.layer2"].features[
            "clinical_entity_tags"
        ]

        data_train_ids, data_val_ids = list(
            k_fold.split(train_set),
        )[self.hparams.fold]
        self.data_train = Subset(train_set, data_train_ids.tolist())
        self.data_val = Subset(train_set, data_val_ids.tolist())
        self.data_test = test_set

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader.

        Returns: the train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=DataCollatorForTokenClassification(tokenizer=self.tokenizer),
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        Returns: the validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=DataCollatorForTokenClassification(tokenizer=self.tokenizer),
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader.

        Returns: the test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=DataCollatorForTokenClassification(tokenizer=self.tokenizer),
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test.

        Args:
            stage: the stage to teardown. Can be either 'fit' or 'test'.
        """
        pass

    def state_dict(self) -> dict:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Things to do when loading checkpoint."""
        pass

    def tokenize_and_align_labels(self, examples: dict) -> dict:
        tokenized_inputs = self.tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

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

    @staticmethod
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
