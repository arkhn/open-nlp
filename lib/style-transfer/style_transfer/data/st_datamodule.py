from typing import Any, Dict, Optional, Union

import datasets
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class StyleTransferDataModule(LightningDataModule):
    """`LightningDataModule` for the StyleTransfer dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_combinations: int = 4,
        instruction: str = "",
        response_instruction="",
        name: str = "",
    ):
        """Initialize a `StyleTransferDataModule`.

        Args:
            data_dir: The data directory. Defaults to `"data/"`.
            batch_size: The batch size. Defaults to `64`.
            num_workers: The number of workers. Defaults to `0`.
            pin_memory: Whether to pin memory. Defaults to `False`.
            max_combinations: The maximum number of combinations to generate. Defaults to `4`.
            instruction: The instruction to use. Defaults to `""`.
            response_instruction: The response instruction to use. Defaults to `""`.
            name: The name of the dataset to use. Defaults to `""`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None  # type: ignore
        self.data_val: Optional[Dataset] = None  # type: ignore
        self.data_test: Optional[Dataset] = None  # type: ignore
        self.dataset: Optional[datasets.Dataset] = None  # type: ignore

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        self.dataset = datasets.load_dataset(self.hparams.name)  # type: ignore

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`,
        so be careful not to execute things like random split twice!
        Also, it is called after `self.prepare_data()` and there is a barrier in between
        which ensures that all the processes proceed to `self.setup()` once
        the data is prepared and available for use.

        Args:
            stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
                Defaults to ``None``.
        """

        # load and split datasets only if not loaded already
        def generate_dataset(dataset: Union[dict, list], max_combinations: int = 1):
            def create_combinations(examples):
                combinations = []
                for id_entry, entry in enumerate(range(len(examples["keywords"]))):
                    current_keywords = examples["keywords"][entry]
                    past_keywords = ", ".join(
                        [
                            keyword
                            for index, keyword in enumerate(examples["keywords"])
                            if index != id_entry
                        ]
                    )
                    combinations += [
                        {
                            "current_keywords": current_keywords,
                            "past_keywords": past_keywords,
                            "text": examples["text"][entry],
                        }
                    ]
                    if id_entry == max_combinations:
                        break

                return {"combinations": combinations}

            return datasets.Dataset.from_list(  # type: ignore
                [
                    entry
                    for history in dataset.map(create_combinations)["combinations"]  # type: ignore
                    for entry in history
                ]
            )

        self.data_train = generate_dataset(
            self.dataset["train"],  # type: ignore
            max_combinations=self.hparams.max_combinations,
        )
        self.data_val = generate_dataset(
            self.dataset["validation"],  # type: ignore
            max_combinations=self.hparams.max_combinations,
        )

        self.data_test = generate_dataset(
            self.dataset["test"],  # type: ignore
            max_combinations=self.hparams.max_combinations,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns:
            The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns:
            The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns:
            The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def teardown(self, stage: Optional[str] = None):
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        Args:
            stage: The stage being torn down.
            Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns:
            Dict[Any, Any]: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        Args:
            state_dict: The datamodule state dictionary to load from.
        """
        pass

    def collate_fn(self, batch: Any) -> Any:
        """Override to customize the default collate_fn.

        Args:
            batch: A batch of data.

        Returns:
            The batch of data.
        """

        x = [
            str.format(
                self.hparams.instruction,
                data_point["current_keywords"],
                data_point["past_keywords"],
            )
            for data_point in batch
        ]
        x = self.trainer.model.tokenizer(x, truncation=True, padding=True, return_tensors="pt")
        texts = [data_point["text"] for data_point in batch]
        decoder_x = self.trainer.model.tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt"
        )
        decoder_x["decoder_input_ids"] = decoder_x["input_ids"]
        decoder_x["decoder_attention_mask"] = decoder_x["attention_mask"]
        del decoder_x["input_ids"]
        del decoder_x["attention_mask"]
        return x, decoder_x, texts


if __name__ == "__main__":
    _ = StyleTransferDataModule()
