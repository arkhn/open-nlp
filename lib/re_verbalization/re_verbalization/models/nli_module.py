import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification.accuracy import MulticlassAccuracy
from transformers import AutoModelForSequenceClassification


class NliModule(LightningModule):
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
    ):
        """Initialize model.
        We use the init to define the model architecture and the different metrics we want to track.

        Args:
            model: The model to use for the classification task.
            optimizer: The optimizer to use for the classification task.
            scheduler: The scheduler to use for the classification task.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.transformers = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.model, num_labels=3
        )

        # metric objects for calculating and averaging accuracy across batches
        self.scores = MetricCollection(
            MulticlassAccuracy(average="macro", num_classes=3),
        )

        self.train_metrics = self.scores.clone(prefix="train/")
        self.val_metrics = self.scores.clone(prefix="val/")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def on_train_start(self) -> None:
        """Called when the train begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

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
        """Perform a training step. We monitor the loss and the accuracy.

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
        self.train_metrics(predictions.view(-1, 3), labels.view(-1))
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)

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

        # update and log metrics
        self.val_loss(loss)
        self.val_metrics(predictions.view(-1, 3), labels.view(-1))
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "predictions": predictions, "targets": batch["labels"]}

    def validation_epoch_end(self, outputs: list) -> None:
        """Called at the end of the validation epoch.
        We log the best so far validation accuracy and F1 score to select the best model.

        Args:
            outputs: The outputs of the validation step.
        """
        current_metrics = self.val_metrics.compute()
        self.val_acc_best(current_metrics["val/MulticlassAccuracy"])  # update best so far val f1
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int):
        """Perform a test step. We monitor the loss, the accuracy and the F1 score.

        Args:
            batch: The batch to perform the test step on.
            batch_idx: The index of the batch.
        """
        pass

    def test_epoch_end(self, outputs: list) -> None:
        """Perform the test epoch end."""
        pass

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
