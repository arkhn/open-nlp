from typing import Optional

import torch
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import classification_report
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torchmetrics import F1Score, MaxMetric, Precision, Recall
from transformers import AutoModelForTokenClassification


class NerCamembertModule(LightningModule):
    def __init__(
        self,
        labels2id: dict,
        id2labels: dict,
        optimizer: torch.optim.Optimizer,
        weights: Optional[Tensor],
        num_labels: int = 19,
        ignore_index: int = -100,
        architecture: str = "camembert-base",
    ):
        super().__init__()

        self.save_hyperparameters()
        self.transformers = AutoModelForTokenClassification.from_pretrained(
            self.hparams.architecture, num_labels=num_labels
        )
        self.transformers.config.labels2id = labels2id
        self.transformers.config.id2labels = id2labels
        self.o_id = self.transformers.config.labels2id["O"]
        self.train_f1 = F1Score(ignore_index=self.o_id, average="weighted", num_classes=num_labels)
        self.val_f1 = F1Score(ignore_index=self.o_id, average="weighted", num_classes=num_labels)
        self.val_recall = Recall(ignore_index=self.o_id, average="weighted", num_classes=num_labels)
        self.val_precision = Precision(
            ignore_index=self.o_id, average="weighted", num_classes=num_labels
        )
        self.val_f1_best = MaxMetric()
        self.val_recall_best = MaxMetric()
        self.val_precision_best = MaxMetric()
        self.criterion = CrossEntropyLoss(ignore_index=ignore_index, weight=weights)

    def step(self, batch: dict):
        y_hat = self.transformers(
            **{
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
            }
        ).logits
        return y_hat, batch["labels"]

    def fit_for_metrics(self, y_hat, labels):
        pred = y_hat.clone().view(-1, y_hat.shape[-1])
        targets = labels.clone().view(-1)
        targets[targets == -100] = self.train_f1.ignore_index
        return torch.sigmoid(pred), targets.long()

    def training_step(self, batch: dict, batch_idx: int):
        y_hat, labels = self.step(batch)
        loss = self.criterion(y_hat.view(-1, y_hat.shape[-1]), labels.view(-1))
        # log train metrics
        self.train_f1(*self.fit_for_metrics(y_hat, labels))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: list):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: dict, batch_idx: int):
        y_hat, labels = self.step(batch)
        value_for_metrics = self.fit_for_metrics(y_hat, labels)
        self.val_f1(*value_for_metrics)
        self.val_recall(*value_for_metrics)
        self.val_precision(*value_for_metrics)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"val/f1": self.val_f1, "val/value_for_metrics": value_for_metrics}

    def validation_epoch_end(self, outputs: list):
        f1 = self.val_f1.compute()
        self.val_f1_best.update(f1)
        self.log("val/f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=True)

        recall = self.val_recall.compute()
        self.val_recall_best.update(recall)
        self.log(
            "val/recall_best",
            self.val_recall_best.compute(),
            on_epoch=True,
            prog_bar=False,
        )

        precision = self.val_precision.compute()
        self.val_precision_best.update(precision)
        self.log(
            "val/precision_best",
            self.val_precision_best.compute(),
            on_epoch=True,
            prog_bar=False,
        )
        # print report
        y_hat = (
            torch.nn.utils.rnn.pad_sequence(
                [example["val/value_for_metrics"][0] for example in outputs],
                padding_value=self.o_id,
            )
            .argmax(dim=-1)
            .view(-1)
            .detach()
            .cpu()
        )
        y = (
            torch.nn.utils.rnn.pad_sequence(
                [example["val/value_for_metrics"][1] for example in outputs],
                padding_value=self.o_id,
            )
            .view(-1)
            .detach()
            .cpu()
        )
        target_names = [
            self.transformers.config.id2labels[label]
            for label in set(y.unique().tolist() + y_hat.unique().tolist())
        ]
        labels = [
            idx_target_name
            for idx_target_name, target_name in enumerate(target_names)
            if target_name != "O"
        ]

        self.log_dict(
            classification_report(
                y, y_hat, target_names=target_names, labels=labels, output_dict=True
            )
        )

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }
