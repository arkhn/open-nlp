# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This is the huggingface dataset for yelp review style transfer.
"""

import os

import datasets
import pyarrow.parquet as pq

_DESCRIPTION = """\
This dataset is a collection of clinical cases from mimic-iii that have been preprocessed
to be used for style transfer.
"""

_DATASET_NAME = "medical_transcription_for_seed"

_HOMEPAGE = "https://github.com/arkhn/open-nlp"

_LICENSE = "http://www.apache.org/licenses/LICENSE-2.0"

_URLS = {
    "train": "data/train.parquet",
}

_CITATION = """\
[Author(s)], [Year of Publication]. [Dataset Title]. DataFog. Available at: [URL].
Accessed on [Date of Access].
"""


class MimicIiiDataset(datasets.GeneratorBasedBuilder):
    """This is the huggingface dataset for medical transcription for seed."""

    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=_DATASET_NAME,
            version=VERSION,
            description="This is a collection of clinical cases from mimic iii "
            "that have been preprocessed to be used for style transfer.",
        ),
    ]

    DEFAULT_CONFIG_NAME = _DATASET_NAME

    def _info(self):
        features = datasets.Features(
            {
                "text_id": [datasets.Value("int32")],
                "keywords": [datasets.Value("string")],
                "text": [datasets.Value("string")],
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dirs = dl_manager.download_and_extract(
            _URLS,
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dirs["train"]),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        table = pq.read_table(filepath)
        df = table.to_pandas()

        for guid, row in df.iterrows():
            yield guid, {
                "text_id": [row["id"]],
                "keywords": [row["keywords"]],
                "text": [row["text"]],
            }
