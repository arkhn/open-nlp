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

import json
import os

import datasets

_DESCRIPTION = """\
This dataset is a collection of clinical cases from pmc patient that have been preprocessed
to be used for style transfer.
"""

_DATASET_NAME = "pmc_dataset"

_HOMEPAGE = "https://github.com/arkhn/ai-lembic"

_LICENSE = "http://www.apache.org/licenses/LICENSE-2.0"

_URLS = {
    "train": "data/train.jsonl",
    "dev": "data/dev.jsonl",
    "test": "data/test.jsonl",
}

_CITATION = """\
@misc{zhao2023pmcpatients,
      title={PMC-Patients: A Large-scale Dataset of Patient Summaries and
      Relations for Benchmarking Retrieval-based Clinical Decision Support Systems},
      author={Zhengyun Zhao and Qiao Jin and Fangyuan Chen and Tuorui Peng and Sheng Yu},
      year={2023},
      eprint={2202.13876},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


class PmcDataset(datasets.GeneratorBasedBuilder):
    """This is the huggingface dataset for PMC style transfer."""

    VERSION = datasets.Version("0.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=_DATASET_NAME,
            version=VERSION,
            description="This is a collection of clinical cases from pmc patient "
            "that have been preprocessed to be used for style transfer.",
        ),
    ]

    DEFAULT_CONFIG_NAME = _DATASET_NAME

    def _info(self):
        features = datasets.Features(
            {
                "user_id": datasets.Value("string"),
                "text_id": [datasets.Value("int32")],
                "title": [datasets.Value("string")],
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
        data_dirs = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dirs["train"]),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dirs["dev"]),
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dirs["test"]),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            for key, row in enumerate(f):
                data_object = json.loads(row)
                user_id, data = list(data_object.items())[0]
                yield guid, {
                    "user_id": user_id,
                    "text_id": [entry["id"] for entry in data],
                    "keywords": [entry["keywords"] for entry in data],
                    "title": [entry["title"] for entry in data],
                    "text": [entry["text"]["text"] for entry in data],
                }
                guid += 1
