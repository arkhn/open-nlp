from typing import Any, Tuple

import peft
import torch
from lightning import LightningModule
from style_transfer.models.rl.oracles.base import Oracle
from torch import tensor
from torch.nn.functional import pad
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.text import ROUGEScore
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BatchEncoding,
    T5ForConditionalGeneration,
)


class RlT5Module(LightningModule):
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
        compile: bool,
        oracles: list[Oracle],
        lora: peft.LoraConfig = None,
        tau_noise: float = 1e-9,
        noisy_lambda: float = 0.5,
        reward_lambda: float = 0.5,
    ):
        """Initialize a `StyleTransferModule`.

        Args:
            model_name: The model name.
            optimizer: The optimizer.
            scheduler: The scheduler.
            max_length: The maximum length.
            compile: Whether to compile.
            oracles: The oracle to evaluate the generated text.
            lora: Whether to use LoRA.
            tau_noise: The tau noise.
            noisy_lambda: The noisy factor to multiply the noisy loss.
            reward_lambda: The reward factor to multiply the reward loss.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = self.load_model()
        self.oracles = self.hparams.oracles
        self.frozen_model = self.load_model()
        self.get_lora()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        for param in self.frozen_model.parameters():
            param.requires_grad = False

        # log metrics
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

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.val_target_text: list = []
        self.val_preds_text: list = []
        self.test_target_text: list = []
        self.test_preds_text: list = []
        self.train_preds_text: list = []
        self.train_target_text: list = []

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def training_step(
        self, batch: Tuple[BatchEncoding, BatchEncoding, list[str]], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
            batch_idx: The index of the current batch.

        Returns:
            The loss.
        """

        noisy_loss = self.noisy_step(batch_dec=batch[1], batch_enc=batch[0])
        reward_loss, rewards_scores, text_preds = self.rewards_step(
            batch_enc=batch[0],
            batch_dec=batch[1],
            text_targets=batch[2],
        )
        print(text_preds)

        loss = noisy_loss * self.hparams.noisy_lambda + reward_loss * self.hparams.reward_lambda
        self.train_loss(loss)
        self.train_rouge.update(preds=[text for text in text_preds], target=batch[2])
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/rewards",
            torch.stack(rewards_scores).mean(),
            on_step=True,
            on_epoch=True,
        )
        self.train_preds_text.extend(text_preds)
        self.train_target_text.extend(batch[2])

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.log_dict(
            {f"train/{name}": value for name, value in self.val_rouge.compute().items()},
            on_step=False,
            on_epoch=True,
        )
        self.trainer.loggers[0].log_table(
            "train/preds_text",
            columns=["preds_text", "target_text"],
            data=[
                [text, target]
                for text, target in zip(self.train_preds_text, self.train_target_text)
            ],
            step=self.trainer.log_every_n_steps,
        )
        self.train_preds_text = []
        self.train_target_text = []

    def validation_step(
        self, batch: Tuple[BatchEncoding, BatchEncoding, list], batch_idx: int
    ) -> tensor:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
            batch_idx: The index of the current batch.

        Returns:
            The loss.
        """
        loss, preds_ids, preds_text = self.model_step(batch[0], batch[1])
        # update and log metrics
        self.val_loss(loss)

        self.val_acc(
            preds=self.pad_max_length(preds_ids, pad_token_id=0),
            target=self.pad_max_length(
                batch[1]["decoder_input_ids"],
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )
        self.val_rouge.update(preds=preds_text, target=batch[2])

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_target_text.extend(batch[2])
        self.val_preds_text.extend(preds_text)
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
        self.loggers[0].log_table(
            "val/preds_text",
            columns=["preds_text", "target_text"],
            data=[
                [text, target] for text, target in zip(self.val_preds_text, self.val_target_text)
            ],
            step=self.trainer.log_every_n_steps,
        )
        self.val_preds_text = []
        self.val_target_text = []

    def test_step(self, batch: Tuple[BatchEncoding, BatchEncoding, list], batch_idx: int) -> tensor:
        """Perform a single test step on a batch of data from the test set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
            batch_idx: The index of the current batch.

        Returns:
            The loss.
        """
        loss, preds_ids, preds_text = self.model_step(batch[0], batch[1])
        self.test_loss(loss)
        self.test_acc(
            preds=self.pad_max_length(preds_ids, pad_token_id=0),
            target=self.pad_max_length(
                batch[1]["decoder_input_ids"],
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )
        self.test_rouge.update(preds=preds_text, target=batch[2])

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.test_target_text.extend(batch[2])
        self.test_preds_text.extend(preds_text)
        # return loss or backpropagation will fail
        return loss

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log_dict(
            {f"test/{name}": value for name, value in self.test_rouge.compute().items()},
            on_step=False,
            on_epoch=True,
        )
        self.trainer.loggers[0].log_table(
            "test/preds_text",
            columns=["preds_text", "target_text"],
            data=[
                [text, target] for text, target in zip(self.test_preds_text, self.test_target_text)
            ],
            step=self.trainer.log_every_n_steps,
        )
        self.val_preds_text = []
        self.val_target_text = []

    def noisy_step(self, batch_dec: BatchEncoding, batch_enc: BatchEncoding) -> torch.Tensor:
        """Perform a single model step on a batch of data.

        Args:
            batch_enc: The batch encoding.
            batch_dec: The batch decoding.

        Returns:
            The loss, generated ids and generated text.
        """
        one_hot_labels = torch.nn.functional.one_hot(
            batch_dec["decoder_input_ids"],
            num_classes=self.model.config.vocab_size,
        ).float()
        if self.hparams.tau_noise > 0:
            labels = (
                torch.nn.functional.gumbel_softmax(
                    one_hot_labels,
                    tau=self.hparams.tau_noise,
                )
                + one_hot_labels
            )
        else:
            labels = one_hot_labels
        batch_dec["decoder_input_ids"] = torch.argmax(labels, dim=-1)
        output = self.forward_noisy_step(batch_dec, batch_enc, labels.argmax(dim=-1))
        return output.loss

    def rewards_step(
        self, batch_enc: BatchEncoding, batch_dec: BatchEncoding, text_targets: list[str]
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
        """Compute the loss taking into account the rewards.

        Args:
            batch_enc: The batch encoding.
            batch_dec: The batch decoding.
            text_targets: The target text.

        Returns:
            The loss.
        """
        preds = self.forward_step(batch_enc=batch_enc, batch_dec=batch_dec)
        preds_ids = torch.argmax(preds.logits, dim=-1)
        preds_text = self.decode_texts(preds_ids)

        frozen_likelihoods = self.compute_nll(
            preds_ids,
            self.frozen_model,
        )
        likelihoods = self.compute_nll(
            preds_ids,
            self.model,
        )

        rewards_scores = [oracle(preds=preds_text, targets=text_targets) for oracle in self.oracles]
        alpha = 1 - torch.stack(rewards_scores).mean()
        losses = (alpha.to(self.device) * likelihoods) * 0.5 + frozen_likelihoods * 0.5
        return losses.mean(), rewards_scores, preds_text

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
        output = self.forward_step(batch_dec, batch_enc)
        loss = output.loss
        preds_ids = torch.argmax(output.logits, dim=-1)
        preds_text = self.decode_texts(preds_ids)
        return loss, preds_ids, preds_text

    def forward_step(self, batch_dec, batch_enc):
        """Perform a single forward step on a batch of data.

        Args:
            batch_enc: The batch encoding.
            batch_dec: The batch decoding.

        Returns:
            The output.
        """

        return self.model(
            **batch_enc,
            **batch_dec,
            labels=batch_dec["decoder_input_ids"],
        )

    def forward_noisy_step(self, batch_dec, batch_enc, labels) -> torch.Tensor:
        """Perform a single forward step on a batch of data.

        Args:
            batch_enc: The batch encoding.
            batch_dec: The batch decoding.
            labels: The labels.

        Returns:
            The output.
        """
        return self.model(
            **batch_enc,
            **batch_dec,
            labels=labels,
        )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Args:
            stage: The stage being set up.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> dict[str, Any]:
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

    def load_model(self) -> T5ForConditionalGeneration:
        """Load the model.

        Returns:
            The model.
        """
        return AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_name,
        )

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

    def compute_nll(self, preds_ids, model) -> torch.Tensor:
        """Compute the negative log likelihood.

        Args:
            preds_ids: The predicted ids.
            model: The model.

        Returns:
            The negative log likelihood.
        """
        stride = 512
        seq_len = preds_ids.size(1)
        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + self.hparams.max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = preds_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        return torch.stack(nlls)

    def get_lora(self):
        if self.hparams.lora:
            lora = self.hparams.lora
            self.model = peft.get_peft_model(self.model, lora)

    def decode_texts(self, preds_ids) -> list[str]:
        """Decode the predicted ids to texts.

        Args:
            preds_ids: The predicted ids.

        Returns:
            The decoded texts.
        """
        return self.tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
