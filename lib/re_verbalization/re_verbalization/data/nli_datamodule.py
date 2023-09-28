from typing import Optional

import datasets
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, DataCollatorWithPadding


class NliDataModule(LightningDataModule):
    """Example of LightningDataModule for E3C dataset.

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
        data_dir: str = "data",
        pin_memory: bool = False,
        tokenizer: str = "camembert-base",
        fold: int = 0,
    ):
        """Initialize a DataModule.

        Args:
            batch_size: the batch size to use for the dataloaders.
            num_workers: the number of workers to use for loading data.
            pin_memory: whether to copy Tensors into CUDA pinned memory.
            tokenizer: the tokenizer to use associated with the model.
        """

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.labels: Optional[list] = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)

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

        re_dataset = (
            datasets.Dataset.from_json(self.hparams.data_dir)
            .map(self.tokenize)
            .with_format(columns=["input_ids", "label"])
        )

        k_fold = KFold(n_splits=5, shuffle=True)
        self.labels = re_dataset.features["label"]

        data_train_ids, data_val_ids = list(
            k_fold.split(re_dataset),
        )[self.hparams.fold]
        self.data_train = Subset(re_dataset, data_train_ids.tolist())
        self.data_val = Subset(re_dataset, data_val_ids.tolist())

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader.

        Returns: the train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
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
            collate_fn=DataCollatorWithPadding(tokenizer=self.tokenizer),
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader.

        Returns: the test dataloader.
        """
        pass

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

    def tokenize(self, examples: dict) -> dict:
        args = (examples["premise"], examples["hypothesis"])
        result = self.tokenizer(*args, padding=True, truncation=True)

        return result
