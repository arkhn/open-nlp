from typing import Any, Dict, Optional

import torch
from commons.data.load_and_preprocess_dataset import load_and_preprocess_dataset
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class T5DataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 6 key methods:
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
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 8,
        pin_memory: bool = False,
        seed: int = 42,
        tokenizer: str = "",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset: Dataset = None
        # data transformations

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        (
            self.data_train,
            self.data_val,
            self.data_test,
            self.data_predict,
        ) = load_and_preprocess_dataset()

    def train_dataloader(self):
        return DataLoader(
            collate_fn=self.collate_fn,
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            collate_fn=self.collate_fn,
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            collate_fn=self.collate_fn,
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            collate_fn=self.collate_fn,
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    def collate_fn(self, batch):
        prompts = []
        responses = []
        labels = []
        for point in batch:
            prompt = (
                "Please find the right answers between the possible answers "
                "to the following question.\n"
                "Question: {q}\n"
                "Possible answers:\n"
                "- a: {a} \n"
                "- b: {b} \n"
                "- c: {c} \n"
                "- d: {d} \n"
                "- e: {e}"
            )
            prompt = prompt.format(
                q=point["question"],
                a=point["answer_a"],
                b=point["answer_b"],
                c=point["answer_c"],
                d=point["answer_d"],
                e=point["answer_e"],
            )
            response = ",".join(
                [["a", "b", "c", "d", "e"][response] for response in point["correct_answers"]]
            )

            label = [0.0] * 5

            for answer_id in point["correct_answers"]:
                label[answer_id] = 1.0

            prompts.append(prompt)
            responses.append(response)
            labels.append(label)

        # Tokenize prompts with local padding and set max_length
        tokenized_prompts = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
        )
        tokenized_responses = self.tokenizer(responses, padding="longest")

        return (
            {
                "input_ids": torch.tensor(tokenized_prompts["input_ids"]),
                "attention_mask": torch.tensor(tokenized_prompts["attention_mask"]),
                "labels": torch.tensor(tokenized_responses["input_ids"]),
            },
            {"labels_text": responses, "input_ids_text": prompts, "ground_truth": labels},
        )


if __name__ == "__main__":
    _ = T5DataModule()
