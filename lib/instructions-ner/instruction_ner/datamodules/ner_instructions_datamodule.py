from random import choices
from typing import Optional

import torch
from datasets import Dataset
from sklearn.model_selection import KFold
from sklearn.utils import compute_class_weight
from src.datamodules.ner_camembert_datamodule import NerCamembertDataModule
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

CUSTOM_SPAN = "custom.Span"


class NerInstructionsDataModule(NerCamembertDataModule):
    def __init__(
        self,
        num_workers=0,
        data_dir=None,
        batch_size: int = 16,
        pin_memory: bool = False,
        k_fold: int = 5,
        c_fold: int = 0,
        architecture: str = "t5-small",
        datasets: Optional[list] = None,
        sentence_mode: str = "fr",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()
        if datasets is None:
            raise ValueError("datasets_collection must be provided")
        else:
            self.datasets = [data_dir + dataset for dataset in datasets]
        self.batch_size = batch_size
        self.json_full_dataset: list = []
        self.full_dataset = None
        self.train_dataset = None
        self.eval_dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.architecture)

        self.labels2fr_types: dict = {
            "DRUG": "médicaments",
            "DATE": "date",
            "EXAM/PROCEDURE": "procédé cliniques",
            "ANAT/PHYSIO": "anatomie ou physiologie",
            "FREQUENCY": "fréquence",
            "TREATMENT": "traitement",
            "VALUE": "valeur",
            "DURATION": "durée",
            "PROBLEM/OBS": "problème physiologique",
        }
        self.fr_types2labels: dict = {
            fr_type: label for label, fr_type in self.labels2fr_types.items()
        }

        self.labels2en_types: dict = {
            "DRUG": "drug",
            "DATE": "date",
            "EXAM/PROCEDURE": "procedure",
            "ANAT/PHYSIO": "anatomy",
            "FREQUENCY": "frequency",
            "TREATMENT": "treatment",
            "VALUE": "value",
            "DURATION": "duration",
            "PROBLEM/OBS": "problem",
        }
        self.en_types2labels: dict = {
            en_type: label for label, en_type in self.labels2en_types.items()
        }

        self.weights: dict = {}
        self.labels2id: dict = {}
        self.id2labels: dict = {}

    def setup(self, stage: Optional[str] = None):
        all_tokens: list = []
        all_words: list = []
        all_tags: list = []
        all_tag_ids: list = []
        all_main_tasks: list = []
        all_text: list = []
        all_main_tasks_targets: list = []
        all_aux_et_tasks: list = []
        all_aux_et_tasks_targets: list = []
        all_aux_ee_tasks: list = []
        all_aux_ee_tasks_targets: list = []
        # Create label2id
        labels = list(
            set(
                [
                    label["label"]
                    for example in self.json_full_dataset
                    for label in example["annotations"]
                ]
            )
        )
        iob_labels = ["B-" + label for label in labels] + ["I-" + label for label in labels] + ["O"]
        self.labels2id = {label: id_label for id_label, label in enumerate(iob_labels)}
        self.id2labels = {id_label: label for id_label, label in enumerate(iob_labels)}

        for example in self.json_full_dataset:
            tags = []
            words = []
            tokens = []
            annotations = example["annotations"]
            prev_tag = "O"
            for token in example["tokens"]:
                is_inside = False
                label = None
                for annotation in annotations:
                    label = annotation["label"]
                    if annotation["start"] <= token["start"] < annotation["end"]:
                        is_inside = True
                        break

                # Check that the NER annotation has been made (ie is not None) and that the label
                # belongs to the model classes.
                if not is_inside or label is None or label not in labels:
                    tag = "O"
                elif prev_tag == f"B-{label}":
                    tag = f"I-{label}"
                elif prev_tag == f"I-{label}":
                    tag = f"I-{label}"
                else:
                    tag = f"B-{label}"

                prev_tag = tag

                tags.append(tag)
                words.append(token["text"])
                tokens.append(token)
                if len(tags) != len(words):
                    raise ValueError("💔 between len of tokens & labels.")

            all_tokens.append(tokens)
            all_words.append(words)
            sentence = example["text"]
            words_tagged = []
            merged_tags = []
            current_word = ""
            current_tag = ""
            for (idx_tag, tag), word in zip(enumerate(tags), words):
                if tag.split("-")[0] == "I" and tag[2:] == current_tag:
                    current_word += " " + word
                elif tag == "O":
                    if current_word != "":
                        words_tagged.append(current_word)
                        merged_tags.append(current_tag)
                        current_word = ""
                elif tag[0] == "B":
                    if current_word != "":
                        words_tagged.append(current_word)
                        merged_tags.append(current_tag)
                    current_tag = tag[2:]
                    current_word = word

            if self.hparams.sentence_mode == "fr":
                str_fr_labels = ", ".join(([k for k in self.fr_types2labels.keys()]))
                all_main_tasks.append(
                    f"Phrase : {sentence}\n"
                    f"Instruction : S'il te plait extrait les entités "
                    f"et leur types de la phrase d'entrée, "
                    f"tout les type d'entités sont en options\n"
                    f"Options : {str_fr_labels}"
                )
                all_main_tasks_targets.append(
                    ", ".join(
                        f"{word} est un(e) {self.labels2fr_types[tag]}"
                        for word, tag in zip(words_tagged, merged_tags)
                    )
                )

                all_aux_et_tasks.append(
                    f"Phrase : {sentence}\n"
                    f"Instruction : S'il te plait trouve les types d'entités "
                    f"en rapport avec la phrase : "
                    f"{', '.join(words_tagged)}\n"
                    f"Options : {str_fr_labels}"
                )
                all_aux_et_tasks_targets.append(
                    ", ".join(
                        f"{word} est un(e) {self.labels2fr_types[tag]}"
                        for word, tag in zip(words_tagged, merged_tags)
                    )
                )

                all_aux_ee_tasks.append(
                    f"Phrase:{sentence}\n"
                    f"Instruction: S'il te plait extrait les entités de la phrase d'entrée"
                )

            else:
                str_en_labels = ", ".join(([k for k in self.en_types2labels.keys()]))
                all_main_tasks.append(
                    f"Sentence: {sentence}\n"
                    f"Instruction: please extract entities and their types from "
                    f"the input sentence, "
                    f"all entity types are in options"
                    f"Options: {str_en_labels}"
                )
                all_main_tasks_targets.append(
                    ", ".join(
                        f"{word} is a {self.labels2en_types[tag]}"
                        for word, tag in zip(words_tagged, merged_tags)
                    )
                )

                all_aux_et_tasks.append(
                    f"Phrase: {sentence}\n"
                    f"Instruction: please type these entity words according to sentence"
                    f"{', '.join(words_tagged)}\n"
                    f"Options: {str_en_labels}"
                )
                all_aux_et_tasks_targets.append(
                    ", ".join(
                        f"{word} is a {self.labels2en_types[tag]}"
                        for word, tag in zip(words_tagged, merged_tags)
                    )
                )

                all_aux_ee_tasks.append(
                    f"Phrase: {sentence}\n"
                    f"Instruction: please extract entity words from the input sentence"
                )
            all_text.append(sentence)
            all_aux_ee_tasks_targets.append(", ".join(f"{word}" for word in words_tagged))
            all_tags.append(tags)
            all_tag_ids.append([self.labels2id[tag] for tag in tags])
        y = [tag for tags in all_tag_ids for tag in tags]
        self.weights = torch.Tensor(
            compute_class_weight(
                class_weight="balanced",
                classes=list(set(y)),
                y=y,
            )
        )

        dataset = {
            "ids": [i for i in list(range(0, len(all_tokens)))],
            "tokens": all_tokens,
            "words": all_words,
            "text": all_text,
            "ner_tags": all_tags,
            "ner_tag_ids": all_tag_ids,
            "main_tasks": all_main_tasks,
            "main_tasks_targets": all_main_tasks_targets,
            "aux_et_task": all_aux_et_tasks,
            "aux_et_task_targets": all_aux_et_tasks_targets,
            "aux_ee_task": all_aux_ee_tasks,
            "aux_ee_task_targets": all_aux_ee_tasks_targets,
        }

        def tokenize_dataset(examples: Dataset) -> Dataset:
            """Tokenize and align the labels over the entire dataset"""
            return self.tokenize_and_align_labels(examples, self.tokenizer, dict_key="words")

        self.full_dataset = Dataset.from_dict(dataset).map(
            tokenize_dataset,
            batched=True,
            remove_columns=["ner_tags", "ids", "ner_tag_ids"],
        )
        k_fold = KFold(n_splits=self.hparams.k_fold, shuffle=True)
        data_train_ids, data_val_ids = list(
            k_fold.split(self.full_dataset),
        )[self.hparams.c_fold]
        self.train_dataset = Subset(self.full_dataset, data_train_ids.tolist())
        self.eval_dataset = Subset(self.full_dataset, data_val_ids.tolist())

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator_for_seq_2_seq,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=32,
            collate_fn=self.data_collator_for_seq_2_seq,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def data_collator_for_seq_2_seq(self, batch):
        candidates = [
            ("main_tasks", "main_tasks_targets"),
            ("aux_et_task", "aux_et_task_targets"),
            ("aux_ee_task", "aux_ee_task_targets"),
        ]
        targets = []
        targets_input_ids = []
        encoder_text = []
        decoder_text = []
        eval_encoder_text = []
        eval_decoder_text = []
        text = []
        tokens = []

        for b in batch:
            draw = choices(candidates, weights=[0.60, 0.20, 0.20], k=1)  # nosec
            encoder_text.append(b[draw[0][0]])
            decoder_text.append(b[draw[0][1]])
            eval_encoder_text.append(b[candidates[0][0]])
            eval_decoder_text.append(b[candidates[0][1]])
            text.append(b["text"])
            targets.append(b["labels"])
            targets_input_ids.append(b["input_ids"])
            tokens.append(b["tokens"])

        encoder_tokens = self.tokenizer(
            encoder_text, return_tensors="pt", padding=True, truncation=True
        )
        decoder_tokens = self.tokenizer(
            decoder_text, return_tensors="pt", padding=True, truncation=True
        )

        generation_targets = decoder_tokens["input_ids"].clone()
        generation_targets[generation_targets == self.tokenizer.pad_token_id] = -100

        eval_encoder_tokens = self.tokenizer(
            eval_encoder_text, return_tensors="pt", padding=True, truncation=True
        )
        eval_decoder_tokens = self.tokenizer(
            eval_decoder_text, return_tensors="pt", padding=True, truncation=True
        )

        eval_generation_targets = eval_decoder_tokens["input_ids"].clone()
        eval_generation_targets[eval_generation_targets == self.tokenizer.pad_token_id] = -100
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor(target) for target in targets],
            padding_value=self.labels2id["O"],
            batch_first=True,
        )
        return {
            "targets": targets,
            "encoder_tokens": encoder_tokens,
            "decoder_tokens": {
                f"decoder_{d_key}": d_value for d_key, d_value in decoder_tokens.items()
            },
            "generation_targets": generation_targets,
            "eval_encoder_tokens": eval_encoder_tokens,
            "eval_decoder_tokens": {
                f"decoder_{d_key}": d_value for d_key, d_value in eval_decoder_tokens.items()
            },
            "eval_generation_targets": eval_generation_targets,
            "tokenizer": self.tokenizer,
            "text": text,
            "tokens": tokens,
            "fr_types2labels": self.fr_types2labels
            if self.hparams.sentence_mode == "fr"
            else self.en_types2labels,
            "sentence_mode": self.hparams.sentence_mode,
        }
