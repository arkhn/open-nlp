import ast
import os
from pathlib import Path
from typing import Iterator

import datasets
import huggingface_hub
import pandas as pd
from datasets import DownloadManager

_DESCRIPTION = """\
This dataset is an annotated corpus of clinical texts from E3C using Large Language Models (LLM).
"""


class E3cLlmConfig(datasets.BuilderConfig):
    """BuilderConfig for E3C-LLM."""

    def __init__(self, **kwargs):
        """BuilderConfig for E3C.

        Args:
          **kwargs: Keyword arguments forwarded to super.
        """
        super(E3cLlmConfig, self).__init__(**kwargs)


class E3cLmm(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        E3cLlmConfig(
            name="e3c-llm",
            version=VERSION,
            description="This dataset is an annotated corpus of clinical texts from E3C using "
            "Large Language Models (LLM).",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:
        """This method specifies the DatasetInfo which contains information and typings."""
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
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
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: DownloadManager) -> list[datasets.SplitGenerator]:
        """Returns SplitGenerators who contains all the difference splits of the dataset.

        This method download from the huggingface hub the different split of the dataset and
        generates all the dataset splits.

        Args:
            dl_manager: This is an inherited arguments from the parent class
                but useless in this case.

        Returns:
            A list of `datasets.SplitGenerator`. Contains all subsets of the dataset depending on
            the language and the layer.
        """

        data_dir = huggingface_hub.snapshot_download(
            repo_id="bio-datasets/e3c-llm", repo_type="dataset", revision="main"
        )

        return list(
            datasets.SplitGenerator(
                name=f"{layer}".replace("-", "_"),
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir,
                        "data",
                        layer,
                        os.listdir(os.path.join(data_dir, "data", layer))[0],
                    )
                },
            )
            for layer in os.listdir(os.path.join(data_dir, "data"))
            if layer != "vanilla"
        )

    def _generate_examples(self, filepath: Path) -> Iterator:
        """Yields examples as (key, example) tuples.

        Args:
            filepath: The path to the folder containing the files to parse.
        Yields:
            An example containing four fields: the text, the annotations, the tokens offsets.
        """
        guid = 0
        df = pd.read_csv(filepath)
        df = df.dropna()
        for _, row in df.iterrows():
            yield guid, {
                "text": row["text"],
                "tokens_offsets": ast.literal_eval(row["entities_offsets"]),
                "clinical_entity_tags": ast.literal_eval(row["labels"]),
            }
            guid += 1


if __name__ == "__main__":
    E3cLmm().download_and_prepare()
