from typing import Optional

import datasets
from datasets import concatenate_datasets
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from weak_supervision.data.e3c_datamodule import E3CDataModule


class E3CBlendedDataModule(E3CDataModule):
    """This dataset blend manual annotations (layer 2 validation) with weak supervision annotations.

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

        train_set = self.blend_dataset(
            train_set, e3c_dataset[f"{self.hparams.language}.layer2.validation"]
        )
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

    @staticmethod
    def blend_dataset(
        train_set: datasets.Dataset, validation_layer: datasets.Dataset
    ) -> datasets.Dataset:
        """Blend the validation layer with the validation set where some common examples
            has been corrected.

        Args:
            train_set: the training set
            validation_layer: the validation layer with corrected examples

        Returns:
            the blended dataset
        """

        filtered_train_dataset = train_set.filter(
            lambda x: x["text"] not in validation_layer["text"]
        )
        return concatenate_datasets([filtered_train_dataset, validation_layer])
