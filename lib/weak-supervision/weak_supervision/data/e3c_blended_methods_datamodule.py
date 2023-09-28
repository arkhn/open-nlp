from typing import Optional

import datasets
import numpy as np
from datasets import Dataset
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from transformers import AutoTokenizer
from weak_supervision.data.e3c_datamodule import E3CDataModule


class E3CBlendedMethodsDataModule(E3CDataModule):
    """This dataset mix annotation from instructGPT and dictionary extraction given a ratio.

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
        language: str = "fr",
        fold: int = 0,
        ratio: float = 0.5,
    ):
        """Initialize a DataModule.

        Args:
            batch_size: the batch size to use for the dataloaders.
            num_workers: the number of workers to use for loading data.
            pin_memory: whether to copy Tensors into CUDA pinned memory.
            tokenizer: the tokenizer to use associated with the model.
            layer: the layer to use for the E3C dataset.
                In this format: "{language}.layer{layer number}".
            language: the language to use for the E3C dataset.
            fold: The current fold distribution . The dataset is split in k folds.
                This fold hyperparameter is used to select the n-th fold as a validation set and
                the other as a training set.
            ratio: the ratio to set a proportion between the dataset annotated with InstructGPT and
                the other annotated with dictionary extraction. A ratio equals to 1 means that the
                dataset is composed of only InstructGPT annotations.
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
            self.hparams.tokenizer,
            add_prefix_space=True,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        This method setup the different huggingface datasets and the k-fold split.
        Several mappings are applied to the datasets to tokenize and align the labels.

        Args:
            stage: the stage to setup. Can be either 'fit' or 'test' or None.
        """
        e3c_dataset = (
            datasets.load_dataset("bio-datasets/e3c")
            .map(self.tokenize_and_align_labels, batched=True)
            .with_format(columns=["input_ids", "attention_mask", "labels"])
        )

        e3c_llm_dataset = (
            datasets.load_dataset("bio-datasets/e3c-llm")
            .map(self.map_offset_to_text, batched=True)
            .map(self.tokenize_and_align_labels, batched=True)
            .with_format(columns=["input_ids", "attention_mask", "labels"])
        )

        train_set = e3c_llm_dataset[f"{self.hparams.language}_layer2"]
        test_set = e3c_dataset[f"{self.hparams.language}.layer1"]

        k_fold = KFold(n_splits=5, shuffle=True)
        self.labels = e3c_dataset[f"{self.hparams.language}.layer2"].features[
            "clinical_entity_tags"
        ]

        data_train_ids, data_val_ids = list(
            k_fold.split(train_set),
        )[self.hparams.fold]
        e3c_train_dataset = Dataset.from_dict(
            e3c_dataset[f"{self.hparams.language}.layer2"][data_train_ids]
        )
        e3c_llm_train_dataset = Dataset.from_dict(
            e3c_llm_dataset[f"{self.hparams.language}_layer2"][data_train_ids]
        )
        self.data_train = e3c_llm_train_dataset.map(
            (
                lambda x, x_idx: x
                if np.random.choice([0, 1], p=[self.hparams.ratio, 1.0 - self.hparams.ratio]) == 0
                else e3c_train_dataset[x_idx]
            ),
            with_indices=True,
        )
        self.data_val = Subset(train_set, data_val_ids.tolist())
        self.data_test = test_set
