import os
from typing import Iterator

import datasets
from bs4 import BeautifulSoup, ResultSet
from datasets import DownloadManager
from syntok.tokenizer import Tokenizer

tok = Tokenizer()


_CITATION = """\
@report{Magnini2021,
author = {Bernardo Magnini and Begoña Altuna and Alberto Lavelli and Manuela Speranza
and Roberto Zanoli and Fondazione Bruno Kessler},
keywords = {Clinical data,clinical enti-ties,corpus,multilingual,temporal information},
title = {The E3C Project:
European Clinical Case Corpus El proyecto E3C: European Clinical Case Corpus},
url = {https://uts.nlm.nih.gov/uts/umls/home},
year = {2021},
}

"""

_DESCRIPTION = """\
The European Clinical Case Corpus (E3C) project aims at collecting and \
annotating a large corpus of clinical documents in five European languages (Spanish, \
Basque, English, French and Italian), which will be freely distributed. Annotations \
include temporal information, to allow temporal reasoning on chronologies, and \
information about clinical entities based on medical taxonomies, to be used for semantic reasoning.
"""

_URL = "https://github.com/hltfbk/E3C-Corpus/archive/refs/tags/v2.0.0.zip"


class E3CConfig(datasets.BuilderConfig):
    """BuilderConfig for E3C."""

    def __init__(self, **kwargs):
        """BuilderConfig for E3C.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(E3CConfig, self).__init__(**kwargs)


class E3C(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        E3CConfig(
            name="e3c",
            version=VERSION,
            description="this is an implementation of the E3C dataset",
        ),
    ]

    def _info(self):
        """This method specifies the DatasetInfo which contains information and typings."""
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "tokens_offsets": datasets.Sequence(datasets.Sequence(datasets.Value("int32"))),
                "clinical_entity_tags": datasets.Sequence(
                    datasets.features.ClassLabel(
                        names=[
                            "O",
                            "B-CLINENTITY",
                            "I-CLINENTITY",
                        ],
                    ),
                ),
                "temporal_information_tags": datasets.Sequence(
                    datasets.features.ClassLabel(
                        names=[
                            "O",
                            "B-EVENT",
                            "B-ACTOR",
                            "B-BODYPART",
                            "B-TIMEX3",
                            "B-RML",
                            "I-EVENT",
                            "I-ACTOR",
                            "I-BODYPART",
                            "I-TIMEX3",
                            "I-RML",
                        ],
                    ),
                ),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> list[datasets.SplitGenerator]:
        """Returns SplitGenerators who contains all the difference splits of the dataset.
        Each language has its own split and each split has 3 different layers (sub-split):
            - layer 1: full manual annotation of clinical entities, temporal information and
                factuality, for benchmarking and linguistic analysis.
            - layer 2: semi-automatic annotation of clinical entities
            - layer 3: non-annotated documents
        Args:
            dl_manager: A `datasets.utils.DownloadManager` that can be used to download and
            extract URLs.
        Returns:
            A list of `datasets.SplitGenerator`. Contains all subsets of the dataset depending on
            the language and the layer.
        """
        url = _URL
        data_dir = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name="en.layer1",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_annotation",
                        "English",
                        "layer1",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="en.layer2",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_annotation",
                        "English",
                        "layer2",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="en.layer2.validation",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_validation",
                        "English",
                        "layer2",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="es.layer1",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_annotation",
                        "Spanish",
                        "layer1",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="es.layer2",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_annotation",
                        "Spanish",
                        "layer2",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="es.layer2.validation",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_validation",
                        "Spanish",
                        "layer2",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="eu.layer1",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_annotation",
                        "Basque",
                        "layer1",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="eu.layer2",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_annotation",
                        "Basque",
                        "layer2",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="eu.layer2.validation",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_validation",
                        "Basque",
                        "layer2",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="fr.layer1",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_annotation",
                        "French",
                        "layer1",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="fr.layer2",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_annotation",
                        "French",
                        "layer2",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="fr.layer2.validation",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_validation",
                        "French",
                        "layer2",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="it.layer1",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_annotation",
                        "Italian",
                        "layer1",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="it.layer2",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_annotation",
                        "Italian",
                        "layer2",
                    ),
                },
            ),
            datasets.SplitGenerator(
                name="it.layer2.validation",
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "E3C-Corpus-2.0.0/data_validation",
                        "Italian",
                        "layer2",
                    ),
                },
            ),
        ]

    @staticmethod
    def get_annotations(entities: ResultSet, text: str) -> list:
        """Extract the offset, the text and the type of the entity.

        Args:
            entities: The entities to extract.
            text: The text of the document.
        Returns:
            A list of list containing the offset, the text and the type of the entity.
        """
        return [
            [
                int(entity.get("begin")),
                int(entity.get("end")),
                text[int(entity.get("begin")) : int(entity.get("end"))],
            ]
            for entity in entities
        ]

    def get_parsed_data(self, filepath: str):
        """Parse the data from the E3C dataset and store it in a dictionary.
        Iterate over the files in the dataset and parse for each file the following entities:
            - CLINENTITY
            - EVENT
            - ACTOR
            - BODYPART
            - TIMEX3
            - RML
        for each entity, we extract the offset, the text and the type of the entity.

        Args:
            filepath: The path to the folder containing the files to parse.
        """
        for root, _, files in os.walk(filepath):
            for file in files:
                with open(f"{root}/{file}") as soup_file:
                    soup = BeautifulSoup(soup_file, "xml")
                    text = soup.find("cas:Sofa").get("sofaString")
                    yield {
                        "CLINENTITY": self.get_annotations(
                            soup.find_all("custom:CLINENTITY"), text
                        ),
                        "EVENT": self.get_annotations(soup.find_all("custom:EVENT"), text),
                        "ACTOR": self.get_annotations(soup.find_all("custom:ACTOR"), text),
                        "BODYPART": self.get_annotations(soup.find_all("custom:BODYPART"), text),
                        "TIMEX3": self.get_annotations(soup.find_all("custom:TIMEX3"), text),
                        "RML": self.get_annotations(soup.find_all("custom:RML"), text),
                        "SENTENCE": self.get_annotations(soup.find_all("type4:Sentence"), text),
                        "TOKENS": self.get_annotations(soup.find_all("type4:Token"), text),
                    }

    def _generate_examples(self, filepath) -> Iterator:
        """Yields examples as (key, example) tuples.
        Args:
            filepath: The path to the folder containing the files to parse.
        Yields:
            an example containing four fields: the text, the annotations, the tokens offsets and
            the sentences.
        """
        guid = 0
        for content in self.get_parsed_data(filepath):
            for sentence in content["SENTENCE"]:
                tokens = [
                    (
                        token.offset + sentence[0],
                        token.offset + sentence[0] + len(token.value),
                        token.value,
                    )
                    for token in list(tok.tokenize(sentence[-1]))
                ]

                filtered_tokens = list(
                    filter(
                        lambda token: token[0] >= sentence[0] and token[1] <= sentence[1],
                        tokens,
                    )
                )
                tokens_offsets = [
                    [token[0] - sentence[0], token[1] - sentence[0]] for token in filtered_tokens
                ]
                clinical_labels = ["O"] * len(filtered_tokens)
                temporal_information_labels = ["O"] * len(filtered_tokens)
                for entity_type in [
                    "CLINENTITY",
                    "EVENT",
                    "ACTOR",
                    "BODYPART",
                    "TIMEX3",
                    "RML",
                ]:
                    if len(content[entity_type]) != 0:
                        for entities in list(
                            content[entity_type],
                        ):
                            annotated_tokens = [
                                idx_token
                                for idx_token, token in enumerate(filtered_tokens)
                                if token[0] >= entities[0] and token[1] <= entities[1]
                            ]
                            for idx_token in annotated_tokens:
                                if entity_type == "CLINENTITY":
                                    if idx_token == annotated_tokens[0]:
                                        clinical_labels[idx_token] = f"B-{entity_type}"
                                    else:
                                        clinical_labels[idx_token] = f"I-{entity_type}"
                                else:
                                    if idx_token == annotated_tokens[0]:
                                        temporal_information_labels[idx_token] = f"B-{entity_type}"
                                    else:
                                        temporal_information_labels[idx_token] = f"I-{entity_type}"
                yield guid, {
                    "text": sentence[-1],
                    "tokens": list(map(lambda token: token[2], filtered_tokens)),
                    "clinical_entity_tags": clinical_labels,
                    "temporal_information_tags": temporal_information_labels,
                    "tokens_offsets": tokens_offsets,
                }
                guid += 1
