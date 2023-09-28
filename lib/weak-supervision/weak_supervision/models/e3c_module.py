import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.classification.accuracy import MulticlassAccuracy
from transformers import AutoModelForTokenClassification


class E3CTokenClassificationModule(LightningModule):
    """Example of LightningModule for E3C classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: str,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        lang: str,
    ):
        """Initialize model.
        We use the init to define the model architecture and the different metrics we want to track.

        Args:
            model: The model to use for the classification task.
            optimizer: The optimizer to use for the classification task.
            scheduler: The scheduler to use for the classification task.
            lang: The language of the model.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transformers = AutoModelForTokenClassification.from_pretrained(
            self.hparams.model, num_labels=3
        )

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.tokens_scores = MetricCollection(
            MulticlassAccuracy(average="macro", num_classes=3, ignore_index=-100),
            MulticlassF1Score(average="macro", num_classes=3, ignore_index=-100),
            MulticlassPrecision(average="macro", num_classes=3, ignore_index=-100),
            MulticlassRecall(average="macro", num_classes=3, ignore_index=-100),
        )
        self.phrases_scores = MetricCollection(
            BinaryF1Score(ignore_index=3),
            BinaryPrecision(ignore_index=3),
            BinaryRecall(ignore_index=3),
        )

        self.no_agg_scores = MetricCollection(
            MulticlassF1Score(average=None, num_classes=3, ignore_index=-100),
            MulticlassPrecision(average=None, num_classes=3, ignore_index=-100),
            MulticlassRecall(average=None, num_classes=3, ignore_index=-100),
        )

        self.train_tokens_metrics = self.tokens_scores.clone(prefix="train/tokens/")
        self.val_tokens_metrics = self.tokens_scores.clone(prefix="val/tokens/")
        self.test_tokens_metrics = self.tokens_scores.clone(prefix="test/tokens/")

        self.train_phrases_metrics = self.phrases_scores.clone(prefix="train/phrases/")
        self.val_phrases_metrics = self.phrases_scores.clone(prefix="val/phrases/")
        self.test_phrases_metrics = self.phrases_scores.clone(prefix="test/phrases/")
        self.test_no_agg_metrics = self.no_agg_scores.clone()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_f1_best = MaxMetric()

    def on_train_start(self) -> None:
        """Called when the train begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_f1_best.reset()

    def model_step(self, batch: dict) -> tuple[Tensor, Tensor]:
        """Perform a forward pass through the model.

        Args:
            batch: The batch to perform the forward pass on.

        Returns:
            The loss and the predictions probabilities.
        """
        outputs = self.transformers(**batch)
        return outputs.loss, outputs.logits

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """Perform a training step. We monitor the loss, the accuracy and the F1 score.

        Args:
            batch: The batch to perform the training step on.
            batch_idx: The index of the batch.

        Returns:
            The loss, the predictions probabilities and the gold labels.
        """
        loss, predictions = self.model_step(batch)
        labels = batch["labels"]
        # update and log metrics
        self.train_loss(loss)
        self.train_tokens_metrics(predictions.view(-1, 3), labels.view(-1))
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_tokens_metrics, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "predictions": predictions, "targets": batch["labels"]}

    def training_epoch_end(self, outputs: list) -> None:
        """Called at the end of the training epoch."""
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs
        # from all batches of the epoch
        # this may not be an issue when training on e3c
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """Perform a validation step. We monitor the loss, the accuracy and the F1 score.

        Args:
            batch: The batch to perform the validation step on.
            batch_idx: The index of the batch.

        Returns:
            The loss, the predictions probabilities and the gold labels.
        """

        loss, predictions = self.model_step(batch)
        labels = batch["labels"]
        binary_predictions = predictions.clone().softmax(dim=-1).argmax(dim=-1)
        binary_predictions[binary_predictions == 2] = 1
        binary_labels = labels.clone()
        binary_labels[binary_labels == 2] = 1
        binary_labels[binary_labels == -100] = 3
        # update and log metrics
        self.val_loss(loss)
        self.val_tokens_metrics(predictions.view(-1, 3), labels.view(-1))
        self.val_phrases_metrics(binary_predictions.view(-1), binary_labels.view(-1))
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_tokens_metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_phrases_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "predictions": predictions, "targets": batch["labels"]}

    def validation_epoch_end(self, outputs: list) -> None:
        """Called at the end of the validation epoch.
        We log the best so far validation accuracy and F1 score to select the best model.

        Args:
            outputs: The outputs of the validation step.
        """
        current_metrics = self.val_phrases_metrics.compute()
        self.val_f1_best(current_metrics["val/phrases/BinaryF1Score"])  # update best so far val f1
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        """Perform a test step. We monitor the loss, the accuracy and the F1 score.

        Args:
            batch: The batch to perform the test step on.
            batch_idx: The index of the batch.
        """

        loss, predictions = self.model_step(batch)
        labels = batch["labels"]
        binary_predictions = predictions.clone().softmax(dim=-1).argmax(dim=-1)
        binary_predictions[binary_predictions == 2] = 1
        binary_labels = labels.clone()
        binary_labels[binary_labels == 2] = 1
        binary_labels[binary_labels == -100] = 3
        self.test_loss(loss)
        self.test_tokens_metrics(predictions.view(-1, 3), labels.view(-1))
        self.test_phrases_metrics(binary_predictions.view(-1), binary_labels.view(-1))
        self.test_no_agg_metrics(predictions.view(-1, 3), labels.view(-1))
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_tokens_metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_phrases_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": predictions}

    def test_epoch_end(self, outputs: list) -> None:
        """Perform the test epoch end."""
        no_aggregated_metrics = self.test_no_agg_metrics.compute()
        dict_metrics = {
            f"test/{log_key}/{label}": log_value[idx_label]
            for idx_label, label in enumerate(self.trainer.datamodule.labels.feature.names)
            for log_key, log_value in no_aggregated_metrics.items()
        }
        self.log_dict(dict_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#
            configure-optimizers
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
