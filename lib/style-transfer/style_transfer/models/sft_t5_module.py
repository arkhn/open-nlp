from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import tensor
from torch.nn.functional import pad
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.text import BLEUScore, ROUGEScore
from transformers import AutoTokenizer, BatchEncoding, T5ForConditionalGeneration


class SftT5Module(LightningModule):
    """Example of a `LightningModule` for StyleTransfer classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model_name: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        max_length: int,
        num_beams: int,
        compile: bool,
    ) -> None:
        """Initialize a `StyleTransferModule`.

        Args:
            model_name (str): The model name.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler): The scheduler.
            max_length (int): The maximum length.
            num_beams (int): The number of beams.
            compile (bool): Whether to compile.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # metric objects for calculating and averaging accuracy across batches
        self.acc = Accuracy(
            task="multiclass",
            num_classes=self.tokenizer.vocab_size,
            ignore_index=self.tokenizer.pad_token_id,
        )
        self.train_acc = self.acc.clone()
        self.val_acc = self.acc.clone()
        self.test_acc = self.acc.clone()

        self.train_rouge = ROUGEScore()
        self.val_rouge = ROUGEScore()
        self.test_rouge = ROUGEScore()

        self.train_bleu = BLEUScore()
        self.val_bleu = BLEUScore()
        self.test_bleu = BLEUScore()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.wandb_logger: WandbLogger = self.trainer.loggers[0]
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(
        self, batch: Tuple[BatchEncoding, BatchEncoding, list], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, generated_ids, generated_text = self.model_step(batch[0], batch[1])
        # update and log metrics
        self.train_loss(loss)

        self.train_acc(
            preds=self.pad_max_length(generated_ids, pad_token_id=0),
            target=self.pad_max_length(
                batch[1]["decoder_input_ids"],
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )
        self.train_bleu(preds=generated_text, target=batch[2])
        self.train_rouge.update(preds=generated_text, target=batch[2])

        self.log("train/bleu", self.train_bleu, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.log_dict(
            {f"train/{name}": value for name, value in self.val_rouge.compute().items()},
            on_step=False,
            on_epoch=True,
        )

    def validation_step(
        self, batch: Tuple[BatchEncoding, BatchEncoding, list], batch_idx: int
    ) -> tensor:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, generated_ids, generated_text = self.model_step(batch[0], batch[1])
        # update and log metrics
        self.val_loss(loss)

        self.val_acc(
            preds=self.pad_max_length(generated_ids, pad_token_id=0),
            target=self.pad_max_length(
                batch[1]["decoder_input_ids"],
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )
        self.val_bleu(preds=generated_text, target=batch[2])
        self.val_rouge.update(preds=generated_text, target=batch[2])

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/bleu", self.val_bleu, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log_dict(
            {f"val/{name}": value for name, value in self.val_rouge.compute().items()},
            on_step=False,
            on_epoch=True,
        )

    def test_step(self, batch: Tuple[BatchEncoding, BatchEncoding, list], batch_idx: int) -> tensor:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, generated_ids, preds_text = self.model_step(batch[0], batch[1])
        self.test_loss(loss)
        self.test_acc(
            preds=self.pad_max_length(generated_ids, pad_token_id=0),
            target=self.pad_max_length(
                batch[1]["decoder_input_ids"],
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )
        self.test_bleu(preds=preds_text, target=batch[2])
        self.test_rouge.update(preds=preds_text, target=batch[2])

        self.wandb_logger.log_table(
            "train/preds_text",
            columns=["preds_text", "target_text"],
            data=[[text, target] for text, target in zip(preds_text, batch[2])],
        )
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/bleu", self.test_bleu, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log_dict(
            {f"test/{name}": value for name, value in self.test_rouge.compute().items()},
            on_step=False,
            on_epoch=True,
        )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers
            to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def pad_max_length(self, generated_ids: tensor, pad_token_id: int = -100):
        """Pad a tensor to the maximum length.

        Args:
            generated_ids (tensor): The tensor to pad.
            pad_token_id (int, optional): The pad token id. Defaults to -100.

        Returns:
            tensor: The padded tensor.
        """
        return pad(
            input=generated_ids,
            pad=(0, self.trainer.model.hparams.max_length - generated_ids.shape[1]),
            value=pad_token_id,
        )

    def model_step(
        self, batch_enc: BatchEncoding, batch_dec: BatchEncoding
    ) -> Tuple[torch.Tensor, torch.Tensor, list[str]]:
        """Perform a single model step on a batch of data.

        Args:
            batch_enc: The batch encoding.
            batch_dec: The batch decoding.

        Returns:
            The loss, generated ids and generated text.
        """
        output = self.generate_output(batch_dec, batch_enc)
        loss = output.loss
        preds_ids = self.model.generate(
            input_ids=batch_enc["input_ids"],
            attention_mask=batch_enc["attention_mask"],
            max_length=self.hparams.max_length + batch_enc["input_ids"].shape[-1],
            num_beams=self.hparams.num_beams,
            early_stopping=True,
        )
        preds_text = self.tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
        return loss, preds_ids, preds_text

    def generate_output(self, batch_dec, batch_enc):
        return self.model(
            **batch_enc,
            **batch_dec,
            labels=batch_dec["decoder_input_ids"],
        )

    def load_model(self) -> T5ForConditionalGeneration:
        """Load the model.

        Returns:
            The model.
        """
        return T5ForConditionalGeneration.from_pretrained(self.hparams.model_name)
