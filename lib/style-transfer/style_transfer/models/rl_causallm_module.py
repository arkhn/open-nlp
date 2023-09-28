import torch
from style_transfer.models.oracles.base import Oracle
from style_transfer.models.rl_t5_module import RlT5Module
from transformers import AutoModelForCausalLM


class RlCausalLmModule(RlT5Module):
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
    ) -> None:
        """Initialize a `StyleTransferModule`.

        Args:
            model_name: The model name.
            optimizer: The optimizer.
            scheduler: The scheduler.
            max_length: The maximum length.
            num_beams: The number of beams.
            num_return_sequences: The number of return sequences.
            compile: Whether to compile.
            oracles: The oracles to evaluate the model.
        """
        super().__init__(
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            compile=compile,
            oracles=oracles,
        )

    def load_model(self):
        return AutoModelForCausalLM.from_pretrained(self.hparams.model_name)

    def generate_output(self, batch_dec, batch_enc):
        return self.model(
            input_ids=batch_dec["decoder_input_ids"],
            attention_mask=batch_dec["decoder_attention_mask"],
            labels=batch_dec["decoder_input_ids"],
        )
