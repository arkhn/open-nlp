import torch
from gen.gen.models.backbone import Backbone
from transformers import AutoTokenizer, T5ForConditionalGeneration


class T5Module(Backbone):
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
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, model: str
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
