from typing import Optional

import datasets
from weak_supervision.data.e3c_datamodule import E3CDataModule


class E3CValidationDataModule(E3CDataModule):
    """This dataset use the validation dataset either from instructGPT or using manual extraction.

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
            train_set = e3c_llm_dataset[f"{self.hparams.language}_layer2_validation"]
        else:
            train_set = e3c_dataset[f"{self.hparams.language}.layer2.validation"]
        test_set = e3c_dataset[f"{self.hparams.language}.layer1"]

        self.labels = e3c_dataset[f"{self.hparams.language}.layer2.validation"].features[
            "clinical_entity_tags"
        ]
        self.data_train = train_set
        self.data_val = test_set
        self.data_test = test_set
