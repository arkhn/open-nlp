from typing import Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from sklearn.metrics import classification_report
from src import utils
from src.datamodules.ner_camembert_datamodule import NerCamembertDataModule
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torchmetrics import F1Score, MaxMetric, Precision, Recall
from transformers import AutoModelForSeq2SeqLM

log = utils.get_pylogger(__name__)


class NerInstructionsModule(LightningModule):
    """
    implementation of this paper : https://arxiv.org/pdf/2203.03903.pdf
    """

    def __init__(
        self,
        labels2id: dict,
        id2labels: dict,
        optimizer: torch.optim.Optimizer,
        weights: Optional[Tensor],
        num_labels: int = 19,
        ignore_index: int = -100,
        architecture: str = "t5-base",
    ):
        super().__init__()

        self.save_hyperparameters()
        self.transformers = AutoModelForSeq2SeqLM.from_pretrained(self.hparams.architecture)
        self.labels2id = labels2id
        self.id2labels = id2labels
        self.o_id = labels2id["O"]
        self.train_f1 = F1Score(ignore_index=self.o_id, average="weighted", num_classes=num_labels)
        self.val_f1 = F1Score(ignore_index=self.o_id, average="weighted", num_classes=num_labels)
        self.val_recall = Recall(ignore_index=self.o_id, average="weighted", num_classes=num_labels)
        self.val_precision = Precision(
            ignore_index=self.o_id, average="weighted", num_classes=num_labels
        )
        self.val_f1_best = MaxMetric()
        self.val_recall_best = MaxMetric()
        self.val_precision_best = MaxMetric()
        self.criterion = CrossEntropyLoss()

    def training_step(self, batch: dict, batch_idx: int):
        outputs = self.transformers(**batch["encoder_tokens"], labels=batch["generation_targets"])
        self.log(
            "train/loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["targets"].shape[0],
        )
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": outputs.loss}

    def training_epoch_end(self, outputs: list):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: dict, batch_idx: int):
        outputs = self.transformers(
            **batch["eval_encoder_tokens"], labels=batch["eval_generation_targets"]
        )
        self.log(
            "val/loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["targets"].shape[0],
        )

        generated_ids = self.transformers.generate(
            **batch["eval_encoder_tokens"],
            max_length=512,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )
        texts = [
            batch["tokenizer"].decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in generated_ids
        ]
        batch["eval_generation_targets"][batch["eval_generation_targets"] == -100] = 0
        target_texts = [
            batch["tokenizer"].decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in batch["eval_generation_targets"]
        ]
        y_hat = (
            self.transform_text(
                texts,
                batch["fr_types2labels"],
                batch["text"],
                batch["tokens"],
                batch["tokenizer"],
                sentence_mode=batch["sentence_mode"],
            )
            .detach()
            .to(self.device)
        )

        flat_batch = batch["targets"].view(-1).detach().to(self.device)
        flat_batch[flat_batch == -100] = self.train_f1.ignore_index
        self.val_f1(y_hat, flat_batch)
        self.val_recall(y_hat, flat_batch).to(self.device)
        self.val_precision(y_hat, flat_batch).to(self.device)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, batch_size=32)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, batch_size=32)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, batch_size=32)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {
            "val/f1": self.val_f1,
            "val/value_for_metrics": (y_hat, flat_batch),
            "val/texts": texts,
            "val/target_texts": target_texts,
        }

    def validation_epoch_end(self, outputs: list):
        f1 = self.val_f1.compute()
        self.val_f1_best.update(f1)
        self.log(
            "val/f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=True, batch_size=32
        )

        recall = self.val_recall.compute()
        self.val_recall_best.update(recall)
        self.log(
            "val/recall_best",
            self.val_recall_best.compute(),
            on_epoch=True,
            prog_bar=False,
            batch_size=32,
        )
        precision = self.val_precision.compute()
        self.val_precision_best.update(precision)
        self.log(
            "val/precision_best",
            self.val_precision_best.compute(),
            on_epoch=True,
            prog_bar=False,
            batch_size=32,
        )
        # print report
        y_hat = (
            torch.nn.utils.rnn.pad_sequence(
                [example["val/value_for_metrics"][0] for example in outputs],
                padding_value=self.o_id,
            )
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
            self.id2labels[label] for label in set(y.unique().tolist() + y_hat.unique().tolist())
        ]
        labels = [
            idx_target_name
            for idx_target_name, target_name in enumerate(target_names)
            if target_name != "O"
        ]
        for logger in self.trainer.logger:
            if type(logger) == WandbLogger:  # noqa: E721
                logger.log_text(
                    key="val/samples",
                    columns=["texts", "targets texts"],
                    data=[
                        [text, target_text]
                        for (text, target_text) in zip(
                            outputs[0]["val/texts"], outputs[0]["val/target_texts"]
                        )
                    ],
                )
        self.log_dict(
            classification_report(
                y, y_hat, target_names=target_names, labels=labels, output_dict=True
            ),
            batch_size=32,
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

    def transform_text(
        self, pred_texts, types2labels, text_sentences, tokens, tokenizer, sentence_mode
    ):
        all_tags = []
        for pred_text, text_sentence, token in zip(pred_texts, text_sentences, tokens):
            tags = []
            sub_sentences = pred_text.split(", ")
            if sentence_mode == "fr":
                spans = [s.find(" est un(e) ") for s in sub_sentences]
                entities = [
                    [el.strip() for el in sub_sentence.split(" est un(e) ")]
                    for span, sub_sentence in zip(spans, sub_sentences)
                    if span != -1
                ]
            else:
                spans = [s.find(" is a ") for s in sub_sentences]
                entities = [
                    [el.strip() for el in sub_sentence.split(" is a ")]
                    for span, sub_sentence in zip(spans, sub_sentences)
                    if span != -1
                ]

            entities = [
                [
                    [
                        text_sentence.find(entity[0]),
                        text_sentence.find(entity[0]) + len(entity[0]),
                    ],
                    types2labels[entity[1]],
                ]
                for entity in entities
                if entity[1] in types2labels.keys() and text_sentence.find(entity[0]) != -1
            ]
            prev_tag = ""
            for t in token:
                is_inside = False
                label = None
                for entity in entities:
                    label = entity[1]
                    if entity[0][0] <= t["start"] - token[0]["start"] < entity[0][1]:
                        is_inside = True
                        break

                if not is_inside:
                    tag = "O"
                elif prev_tag == f"B-{label}":
                    tag = f"I-{label}"
                elif prev_tag == f"I-{label}":
                    tag = f"I-{label}"
                else:
                    tag = f"B-{label}"

                prev_tag = tag
                tags.append(tag)
            all_tags.append([self.labels2id[tag] for tag in tags])

        labels = NerCamembertDataModule.tokenize_and_align_labels(
            {
                "ner_tag_ids": all_tags,
                "tokens": [[token["text"] for token in tokens_list] for tokens_list in tokens],
            },
            tokenizer,
        )["labels"]
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(label) for label in labels],
            padding_value=self.labels2id["O"],
            batch_first=True,
        )
        labels[labels == -100] = self.labels2id["O"]
        return labels.view(-1)
