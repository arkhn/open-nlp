import peft
import torch
from gen.gen.models.backbone import Backbone
from peft import get_peft_model, prepare_model_for_int8_training
from transformers import AutoTokenizer, T5ForConditionalGeneration


class T5XllModule(Backbone):
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
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        lora_config: peft.LoraConfig,
        model: str,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # Define LoRA Config

        lora_config = self.hparams.lora_config
        # prepare int-8 model for training
        model_id = model
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_id, load_in_8bit=True, device_map="auto"
        )

        self.model = prepare_model_for_int8_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
