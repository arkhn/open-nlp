import inspect
from types import MethodType
from typing import Tuple

import torch
from style_transfer.models.oracles.base import Oracle
from style_transfer.models.sft_t5_module import SftT5Module
from transformers import BatchEncoding


class RlT5Module(SftT5Module):
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
        num_return_sequences: int,
        compile: bool,
        oracles: list[Oracle],
    ):
        """Initialize a `StyleTransferModule`.

        Args:
            model_name: The model name.
            optimizer: The optimizer.
            scheduler: The scheduler.
            max_length: The maximum length.
            num_beams: The number of beams.
            num_return_sequences: The number of return sequences.
            compile: Whether to compile.
            oracles: The oracle to evaluate the generated text.
        """
        super().__init__(
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            max_length=max_length,
            num_beams=num_beams,
            compile=compile,
        )
        self.oracles = oracles
        generate_with_grad = inspect.unwrap(self.model.generate)
        self.model.generate_with_grad = MethodType(generate_with_grad, self.model)

    def training_step(
        self, batch: Tuple[BatchEncoding, BatchEncoding, list], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        preds_ids, text_preds, preds_probs = self.model_train_step(batch_enc=batch[0])
        rewards_scores = [oracle(preds=text_preds, targets=batch[2]) for oracle in self.oracles]
        loss = self.compute_loss(preds_probs, rewards_scores)
        self.train_loss(loss)
        self.train_acc(
            preds=self.pad_max_length(preds_ids[:, 0, :], pad_token_id=self.tokenizer.pad_token_id),
            target=self.pad_max_length(
                batch[1]["decoder_input_ids"],
                pad_token_id=self.tokenizer.pad_token_id,
            ),
        )
        self.train_bleu.update(preds=[text for text in text_preds], target=batch[2])
        self.train_rouge.update(preds=[text for text in text_preds], target=batch[2])

        self.wandb_logger.log_table(
            "train/generated_text",
            columns=["generated_text", "target_text"],
            data=[[text, target] for text, target in zip(text_preds, batch[2])],
            step=5,
        )
        self.log("train/bleu", self.train_bleu, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/rewards",
            torch.stack(rewards_scores).mean(),
            on_step=True,
            on_epoch=True,
        )

        # return loss or backpropagation will fail
        return loss

    @staticmethod
    def compute_loss(log_probs: torch.Tensor, rewards_scores: list[torch.Tensor]) -> torch.Tensor:
        """Compute the loss taking into account the rewards.

        Args:
            log_probs: The log probabilities of the generated text.
            rewards_scores: The rewards scores.

        Returns:
            The loss.
        """
        losses = torch.stack(rewards_scores).mean(dim=1) * -log_probs.sum(1)
        return losses.sum()

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.log_dict(
            {f"val/{name}": value for name, value in self.val_rouge.compute().items()},
            on_step=False,
            on_epoch=True,
        )

    def model_train_step(
        self, batch_enc: BatchEncoding
    ) -> Tuple[torch.Tensor, list[str], torch.Tensor]:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch_enc: The batch encoding of the encoder.

        Returns:
            A tuple of the generated ids, the associated log probs for each sentence
            and the generated text.
        """
        output = self.model.generate_with_grad(
            input_ids=batch_enc["input_ids"],
            max_length=self.hparams.max_length + batch_enc["input_ids"].shape[-1],
            num_beams=self.hparams.num_beams,
            early_stopping=False,
            output_scores=True,
            num_return_sequences=1,
            return_dict_in_generate=True,
        )

        preds_ids = output.sequences.reshape(
            self.trainer.datamodule.hparams.batch_size,
            self.hparams.num_return_sequences,
            output.sequences.shape[-1],
        )
        preds_probs = self.model.compute_transition_scores(
            sequences=output["sequences"],
            scores=output["scores"],
            beam_indices=output["beam_indices"],
            normalize_logits=True,
        )
        preds_text = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        return (
            preds_ids,
            preds_text,
            preds_probs,
        )

    def generate_text(self, batch_enc, model) -> list[str]:
        """Generate text from a batch of data.

        Args:
            batch_enc: The batch encoding of the encoder.
            model: The model to use for generation.

        Returns:
            A list of generated text.
        """
        generated_ids = model.generate(
            input_ids=batch_enc["input_ids"],
            attention_mask=batch_enc["attention_mask"],
            max_length=self.hparams.max_length,
            num_beams=self.hparams.num_beams,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
