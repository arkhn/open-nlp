import peft
import torch
from style_transfer.models.rl.oracles.base import Oracle
from style_transfer.models.rl.rl_t5_module import RlT5Module
from transformers import AutoModelForCausalLM


class RlGptModule(RlT5Module):
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
        tau_noise: float = 2e-9,
        noisy_lambda: float = 0.5,
        reward_lambda: float = 0.5,
    ) -> None:
        """Initialize a `StyleTransferModule`.

        Args:
            model_name: The model name.
            optimizer: The optimizer.
            scheduler: The scheduler.
            max_length: The maximum length.
            compile: Whether to compile.
            oracles: The oracles to evaluate the model.
            lora: The lora config.
            tau_noise: The tau noise.
            noisy_lambda: The noisy lambda.
            reward_lambda: The reward lambda.
        """
        super().__init__(
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            max_length=max_length,
            compile=compile,
            oracles=oracles,
            lora=lora,
            tau_noise=tau_noise,
            noisy_lambda=noisy_lambda,
            reward_lambda=reward_lambda,
        )

    def load_model(self):
        return AutoModelForCausalLM.from_pretrained(self.hparams.model_name)

    def forward_step(self, batch_dec, batch_enc) -> torch.Tensor:
        """Forward step.

        Args:
            batch_dec: The batch for the decoder.
            batch_enc: The batch for the encoder.

        Returns:
            The output of the model.
        """
        enc_ids, enc_labels, enc_mask = self.concatenate_instructions(
            batch_dec, batch_enc, batch_dec["decoder_input_ids"]
        )
        return self.model(
            input_ids=enc_ids,
            attention_mask=enc_mask,
            labels=enc_labels,
        )

    def forward_noisy_step(self, batch_dec, batch_enc, labels) -> torch.Tensor:
        """Forward steps.

        Args:
            batch_dec: The batch for the decoder.
            batch_enc: The batch for the encoder.
            labels: The labels.

        Returns:
            The output of the model.
        """

        # Iterate over batch to create input_ids, attention_mask, and labels for the decoder model
        enc_ids, enc_labels, enc_mask = self.concatenate_instructions(batch_dec, batch_enc, labels)
        return self.model(
            input_ids=enc_ids,
            attention_mask=enc_mask,
            labels=enc_labels,
        )

    def concatenate_instructions(self, batch_dec, batch_enc, labels):
        """Concatenate the instructions with the response tokens and target labels.

        Args:
            batch_dec: The batch for the decoder.
            batch_enc: The batch for the encoder.
            labels: The labels is the decoder input ids.
            response_tokens: The response tokens, it the response instruction section.

        Returns:
            The concatenated instructions, labels, and attention mask.
        """
        # Tokenize response instruction
        response_tokens = self.tokenizer(
            self.trainer.datamodule.hparams.response_instruction,
            return_tensors="pt",
        )["input_ids"][0]

        enc_ids = []
        enc_mask = []
        enc_labels = []

        # Iterate over batch to create input_ids, attention_mask, and labels for the decoder model
        # We concatenate the encoder input ids with the decoder input ids and response tokens
        for ids, mask, dec_ids, dec_mask in zip(
            batch_enc["input_ids"],
            batch_enc["attention_mask"],
            labels,
            batch_dec["decoder_attention_mask"],
        ):
            # If the encoder input ids contain a 0, we only want to concatenate
            # the encoder input ids.
            # We have no 0 in the encoder input ids if the input had not been padded

            if 0 in mask:
                enc_ids.append(torch.cat([ids[: torch.where(mask == 0)[0][0]], dec_ids]))
                mask = mask[: torch.where(mask == 0)[0][0]][: -len(response_tokens)]
            else:
                enc_ids.append(torch.cat([ids, dec_ids]))
                mask = mask[: -len(response_tokens)]

            # We use torch.cat to concatenate the response tokens and decoder attention mask
            enc_mask.append(
                torch.cat(
                    [
                        mask,
                        torch.Tensor([1] * len(response_tokens)).to(self.device),
                        dec_mask,
                    ]
                )
            )
            # -100 is the ignore index for the loss function,
            # we use it to ignore the instructions tokens
            enc_labels = enc_ids
        # we pad the input ids, labels, and attention mask and truncate them to the maximum length
        enc_ids = torch.nn.utils.rnn.pad_sequence(
            enc_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )[:, : self.hparams.max_length]
        enc_mask = torch.nn.utils.rnn.pad_sequence(enc_mask, batch_first=True, padding_value=0)[
            :, : self.hparams.max_length
        ]
        enc_labels = torch.nn.utils.rnn.pad_sequence(
            enc_labels, batch_first=True, padding_value=-100
        )[:, : self.hparams.max_length]
        return enc_ids, enc_labels, enc_mask

    def decode_texts(self, preds_ids) -> list[str]:
        """Decode the predicted ids to texts.

        Args:
            preds_ids: The predicted ids.

        Returns:
            The decoded texts.
        """
        return [
            text.split(self.trainer.datamodule.hparams.response_instruction)[1]
            if self.trainer.datamodule.hparams.response_instruction in text
            else ""
            for text in self.tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
        ]
