import re
from typing import Any

import numpy as np
import torch
from commons.metrics.exact_match_ratio import exact_match_ratio
from commons.metrics.hamming_score import hamming_score
from lightning import LightningModule
from torchmetrics import MeanMetric


class Backbone(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
    ):
        super().__init__()
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: dict):
        return self.model(**x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch: Any):
        input_model, input_text = batch
        output = self.forward(input_model)
        loss = output.loss
        generated_text = self.model.generate(
            input_ids=input_model["input_ids"],
            attention_mask=input_model["attention_mask"],
            max_length=32,
            num_beams=4,
            early_stopping=True,
        )
        generated_text = self.tokenizer.batch_decode(generated_text, skip_special_tokens=True)
        return (loss, input_text, generated_text)

    @staticmethod
    def is_valid_output(output_string):
        return bool(re.match(r"^[a-z](,[a-z])*$|^[a-z]$", output_string))

    def training_step(self, batch: Any, batch_idx: int):
        loss, input_text, generated_text = self.model_step(batch)
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        # batch ({inputs, attention, label}, {label_text ...})
        loss, input_text, generated_text = self.model_step(batch)
        preds, targets = self.text_to_one_hot(generated_text, input_text)
        # update and log metrics
        self.val_loss(loss)
        self.log_dict(
            {
                "val/hamming_score": hamming_score(np.array(preds), np.array(targets)),
                "val/exact_match_score": exact_match_ratio(preds, targets),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def text_to_one_hot(self, generated_text, input_text):
        preds = []
        for text in generated_text:
            if not self.is_valid_output(text):
                ValueError("Badly formatted")
            predicted_answers_letters = text.split(",")
            predicted_answers_letters = [s.strip() for s in predicted_answers_letters]
            p = [int(letter in predicted_answers_letters) for letter in "abcde"]
            preds.append(p)
        targets = input_text["ground_truth"]
        return preds, targets

    def test_step(self, batch: Any, batch_idx: int):
        loss, input_text, generated_text = self.model_step(batch)
        preds, targets = self.text_to_one_hot(generated_text, input_text)
        # update and log metrics
        self.test_loss(loss)
        self.log_dict(
            {
                "test/hamming_score": hamming_score(np.array(preds), np.array(targets)),
                "test/exact_match_score": exact_match_ratio(preds, targets),
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        _, input_text, generated_text = self.model_step(batch)
        preds, targets = self.text_to_one_hot(generated_text, input_text)
        return preds

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
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
