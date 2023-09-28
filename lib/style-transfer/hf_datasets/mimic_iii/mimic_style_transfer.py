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
This dataset is a collection of clinical cases from mimic-iii that have been preprocessed
to be used for style transfer.
"""

_DATASET_NAME = "mimic_iii_dataset"

_HOMEPAGE = "https://github.com/arkhn/ai-lembic"

_LICENSE = "http://www.apache.org/licenses/LICENSE-2.0"

_URLS = {
    "train": "data/train.jsonl",
    "dev": "data/dev.jsonl",
    "test": "data/test.jsonl",
}

_CITATION = """\
    @inproceedings{10.1145/3368555.3384469,
    author = {Wang, Shirly and McDermott, Matthew B. A. and Chauhan,
    Geeticka and Ghassemi, Marzyeh and Hughes, Michael C. and Naumann, Tristan},
    title = {MIMIC-Extract: A Data Extraction, Preprocessing, and Representation
    Pipeline for MIMIC-III},
    year = {2020},
    isbn = {9781450370462},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3368555.3384469},
    doi = {10.1145/3368555.3384469},
    abstract = {Machine learning for healthcare researchers face challenges to
    progress and reproducibility due to a lack of standardized processing
    frameworks for public datasets. We present MIMIC-Extract, an open source pipeline
    for transforming the raw electronic health record (EHR) data of critical care
    patients from the publicly-available MIMIC-III database into data structures that are directly
    usable in common time-series prediction pipelines. MIMIC-Extract addresses three challenges
    in making complex EHR data accessible to the broader machine learning community.
    First, MIMIC-Extract transforms raw vital sign and laboratory measurements into usable hourly
    time series, performing essential steps such as unit conversion, outlier handling,
    and aggregation of semantically similar features to reduce missingness and improve robustness.
    Second, MIMIC-Extract extracts and makes prediction of clinically-relevant targets possible,
    including outcomes such as mortality and length-of-stay
    as well as comprehensive hourly intervention signals for ventilators, vasopressors,
    and fluid therapies. Finally, the pipeline emphasizes reproducibility and
    extensibility to future research questions. We demonstrate the pipeline's effectiveness
    by developing several benchmark tasks for outcome and intervention forecasting and
    assessing the performance of competitive models.},
    booktitle = {Proceedings of the ACM Conference on Health, Inference, and Learning},
    pages = {222–235},
    numpages = {14},
    keywords = {Machine learning, MIMIC-III, Healthcare, Time series data, Reproducibility},
    location = {Toronto, Ontario, Canada},
    series = {CHIL '20}}
"""


class MimicIiiDataset(datasets.GeneratorBasedBuilder):
    """This is the huggingface dataset for mimic III style transfer."""

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
                "user_id": datasets.Value("string"),
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
                    "text": [entry["text"] for entry in data],
                }
                guid += 1
