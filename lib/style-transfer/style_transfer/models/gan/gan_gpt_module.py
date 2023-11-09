from typing import Any

import peft
import torch
import torchmetrics.functional
import pysbd
from lightning import LightningModule
from torch._C._nn import pad_sequence
from torchmetrics import MeanMetric
from torchmetrics.text import ROUGEScore
from transformers import AutoModelForCausalLM, AutoTokenizer


class GanGptModule(LightningModule):
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
        batch_accumulation: int,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        max_length: int,
        compile: bool,
        g_lora: peft.LoraConfig = None,
        d_lora: peft.LoraConfig = None,
        stride: int = 512,
        g_model: Any = AutoModelForCausalLM,
        d_model: Any = AutoModelForCausalLM,
        bnb_config: Any = None,
    ) -> None:
        """Initialize a `StyleTransferModule`.
        Args:
            model_name: The model name.
            optimizer: The optimizer.
            scheduler: The scheduler.
            max_length: The maximum length.
            compile: Whether to compile.
            lora: The lora config.
            stride: The stride used to compute the negative log likelihood.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.g_model.keywords["pretrained_model_name_or_path"]
        )

        # Disable automatic optimization to adapt to the GAN training loop using multiple optimizers
        self.automatic_optimization = False

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.generator = self.hparams.g_model(quantization_config=self.hparams.bnb_config())
        self.frozen_generator = self.hparams.g_model(quantization_config=self.hparams.bnb_config())
        for param in self.frozen_generator.parameters():
            param.requires_grad = False
        if self.hparams.g_lora:
            g_lora = self.hparams.g_lora
            self.generator = peft.get_peft_model(self.generator, g_lora)
        # log metrics
        self.train_rouge = ROUGEScore()
        self.val_rouge = ROUGEScore()
        self.test_rouge = ROUGEScore()

        # for averaging loss across batches
        self.generator_train_loss = MeanMetric()
        self.discriminator_train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
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

    def training_step(self, batch: dict, batch_idx: int):
        """Perform a single training step on a batch of data from the training set.
        We adapt the training loop to the GAN training loop using multiple optimizers.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
            batch_idx: The index of the current batch.
        """

        # get optimizers
        optimizer_g = self.optimizers()

        # do the forward pass
        x = self.tokenizer(batch["x"], truncation=True, padding=True, return_tensors="pt")
        input_ids = x["input_ids"]
        _ = self.generator(input_ids=input_ids, labels=input_ids)
        ground_truth_ids = self.tokenizer(
            batch["texts"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        preds_ids = self.generator.generate(
            input_ids=input_ids,
            min_length=len(ground_truth_ids[0]) + len(input_ids[0]),
            max_length=len(ground_truth_ids[0]) + len(input_ids[0]),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        frozen_preds_ids = self.generator.generate(
            input_ids=input_ids,
            min_length=len(ground_truth_ids[0]) + len(input_ids[0]),
            max_length=len(ground_truth_ids[0]) + len(input_ids[0]),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        padded_ids = pad_sequence([frozen_preds_ids[0], preds_ids[0]], batch_first=True)
        text_preds = self.decode_texts(preds_ids)
        rewards = self.evaluator_forward(**self.generate_d_prompts(batch, text_preds))
        sentences_input = self.tokenizer(
            rewards["sentences"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        loss = self.generator(**sentences_input, labels=sentences_input["input_ids"]).loss
        g_loss = (
            0.8 * (loss * rewards["g_rewards"])
            - 0.2
            * torchmetrics.functional.kl_divergence(
                padded_ids[0].unsqueeze(dim=0), padded_ids[1].unsqueeze(dim=0)
            )
        ) / self.hparams.batch_accumulation
        self.manual_backward(g_loss)
        self.generator_train_loss(g_loss)
        if (batch_idx + 1) % self.hparams.batch_accumulation == 0:
            optimizer_g.step()
            optimizer_g.zero_grad()

        # update and log metrics
        self.train_rouge.update(preds=[text for text in text_preds], target=batch["texts"])
        self.log(
            "train/generator_loss",
            self.generator_train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.train_preds_text.extend(text_preds)
        self.train_target_text.extend(batch["texts"])

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

    def validation_step(self, batch: dict, batch_idx: int) -> torch.tensor:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
            batch_idx: The index of the current batch.

        Returns:
            The loss.
        """
        x = self.tokenizer(batch["x"], truncation=True, padding=True, return_tensors="pt")
        input_ids = x["input_ids"]
        output = self.generator.generate(
            input_ids=input_ids,
            min_length=256,
            max_length=self.hparams.max_length,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        preds_text = self.decode_texts(output)
        # update and log metrics
        self.val_rouge.update(preds=preds_text, target=batch["texts"])
        self.val_target_text.extend(batch["texts"])
        self.val_preds_text.extend(preds_text)
        return 0

    def on_validation_epoch_end(self):
        "Lightning hook that is called when a validation epoch ends."
        # otherwise metric would be reset by lightning after each epoch
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

    def test_step(self, batch: dict, batch_idx: int) -> torch.tensor:
        pass

    def evaluator_forward(self, prompts, sentences) -> dict:
        """Forward step.

        Args:
            batch_dec: The batch for the decoder.

        Returns:
            The output of the model.
        """
        # Parse prediction
        scores = 0
        print(len(prompts))
        for prompt in prompts:
            prompt_ids = self.tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")[
                "input_ids"
            ]
            response = self.frozen_generator.generate(
                prompt_ids,
                min_length=128,
                max_length=self.hparams.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            response = self.tokenizer.batch_decode(response, skip_special_tokens=True)[0]
            response = (
                eval(response.split("=")[1].strip()[:4].split(" ")[0])
                if "Result =" in response
                else 0
            )
            print("score response", response)
            scores += response if response < 1 else 0

        print("scores: ", scores)
        return {
            "g_rewards": torch.tensor(scores) / len(prompts),
            "sentences": sentences,
        }

    def decode_texts(self, preds_ids) -> list[str]:
        """Decode the predicted ids to texts.

        Args:
            preds_ids: The predicted ids.

        Returns:
            The decoded texts.
        """
        decoded_preds = self.tokenizer.batch_decode(preds_ids)[0]
        # print to remove
        print(f"{decoded_preds}\n{'*'*100}\n")
        return [
            text.split(self.trainer.datamodule.hparams.generator_response)[1]
            if self.trainer.datamodule.hparams.generator_response in text
            else ""
            for text in self.tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
        ]

    def configure_optimizers(self) -> list[dict]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers
            to be used for training.
        """
        g_optimizer = self.hparams.g_optimizer(params=self.generator.parameters())
        if self.hparams.scheduler is not None:
            g_scheduler = self.hparams.scheduler(optimizer=g_optimizer)
            return [
                {
                    "optimizer": g_optimizer,
                    "lr_scheduler": {
                        "scheduler": g_scheduler,
                        "monitor": "val/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                },
            ]
        return [
            {"optimizer": g_optimizer},
        ]

    def generate_d_prompts(self, batch: dict, preds_texts: list[str]) -> dict[str, Any]:
        """Generate the prompt for the discriminator.

        Args:
            batch: The batch.
            preds_texts: The output of the generator.

        Returns:
            The prompt.
        """
        prompts = []
        sentences = []
        style_accuracy = """Here is sentence S1: {} and sentence S2: {}. 
        How different is sentence S2 compared to S1 on a continuous scale from 0
        (completely different styles) to 1 (completely identical styles)? Result ="""
        content_preservation = """Here is S1: {} and sentence S2: {}.
        How much does S2 preserve the content of S2 on a continuous scale from 0
        (completely different topic) to 1 (identical topic)? Result ="""
        fluency = """How natural is this sentence S1: {} on a scale from 0 to 1
        where 0 (lowest coherent) and 1 (highest coherent)? Result = """
        seg = pysbd.Segmenter(language="en", clean=True)
        for ground_text, pred_text in zip(batch["texts"], preds_texts):
            pred_sents = seg.segment(pred_text.replace("\n", " ").strip())
            ground_sents = seg.segment(ground_text.replace("\n", " ").strip())
            limit = len(pred_sents) if len(pred_sents) < len(ground_sents) else len(ground_sents)
            if len(prompts) == 10:
                break
            for s2, s1 in zip(pred_sents[:limit], ground_sents[:limit]):
                prompts.append(style_accuracy.format(s1, s2))
                prompts.append(content_preservation.format(s1, s2))
                prompts.append(fluency.format(s2))
                sentences.append(s2)
                if len(prompts) == 10:
                    break
        return {
            "prompts": prompts,
            "sentences": " ".join(sentences),
        }
