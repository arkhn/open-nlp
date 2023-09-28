"""This script was adapted from
https://github.com/osainz59/Ask2Transformers/blob/master/scripts/tacred2mnli.py
"""
import json
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

import datasets
import numpy as np

parser = ArgumentParser()
parser.add_argument("cas_archive_path", type=str)
parser.add_argument("--posn", type=int, default=1)
parser.add_argument("--negn", type=int, default=1)
parser.add_argument("--seed", type=int, default=2023)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)


positive_templates = {
    "no_relation": ["{obj} ne s'est pas passé {subj}"],
    "bound": ["{obj} s'est passé {subj}"],
}
negative_templates = {
    "no_relation": ["{obj} s'est passé {subj}"],
    "bound": ["{obj} ne s'est pas passé {subj}"],
}


@dataclass
class REInputFeatures:
    subj: str
    obj: str
    context: str
    pair_type: Optional[str] = None
    label: Optional[str] = None


@dataclass
class MNLIInputFeatures:
    premise: str
    hypothesis: str
    label: int


labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}


def re2mnli(
    instance: REInputFeatures,
    positive_templates,
    negative_templates,
    negn=1,
    posn=1,
):
    mnli_instances = []
    # Generate the positive examples
    positive_template = random.choices(positive_templates[instance.label], k=posn)  # nosec
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["entailment"],
            )
            for t in positive_template
        ]
    )

    # Generate the negative templates
    negative_template = random.choices(negative_templates[instance.label], k=negn)  # nosec
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=instance.context,
                hypothesis=f"{t.format(subj=instance.subj, obj=instance.obj)}.",
                label=labels2id["neutral"]
                if instance.label != "no_relation"
                else labels2id["contradiction"],
            )
            for t in negative_template
        ]
    )

    return mnli_instances


hf_dataset = datasets.load_dataset(
    "bio-datasets/re-medical-annotations",
    labels=["bound"],
    data_dir=args.cas_archive_path,
)["train"]

mnli_data = []
pair_types = set()
for sample in hf_dataset:
    # filter out relations that are not DATE - PROB/OBS | TREATMENT....
    if (
        len(
            [
                ent_type
                for ent_type in [sample["obj_type"], sample["subj_type"]]
                if ent_type in ["DATE", "DURATION"]
            ]
        )
        != 1
    ):
        continue

    # deal with relations that might not be in the right direction
    # we want subj = date and obj = the other entity
    if sample["subj_type"] in ["DATE", "DURATION"]:
        if sample["obj_type"] not in ["PROBLEM/OBS", "TREATMENT", "DRUG", "EXAM/PROCEDURE"]:
            continue
        re_instance = REInputFeatures(
            subj=sample["text"][sample["subj_start"] : sample["subj_end"]],
            obj=sample["text"][sample["obj_start"] : sample["obj_end"]],
            context=sample["text"],
            pair_type=f"{sample['subj_type']}:{sample['obj_type']}",
            label=hf_dataset.features["relation"].int2str(sample["relation"]),
        )
        pair_types.add(f"{sample['subj_type']}:{sample['obj_type']}")
    else:
        if sample["subj_type"] not in ["PROBLEM/OBS", "TREATMENT", "DRUG", "EXAM/PROCEDURE"]:
            continue
        re_instance = REInputFeatures(
            subj=sample["text"][sample["obj_start"] : sample["obj_end"]],
            obj=sample["text"][sample["subj_start"] : sample["subj_end"]],
            context=sample["text"],
            pair_type=f"{sample['obj_type']}:{sample['subj_type']}",
            label=hf_dataset.features["relation"].int2str(sample["relation"]),
        )
        pair_types.add(f"{sample['obj_type']}:{sample['subj_type']}")

    mnli_instance = re2mnli(
        re_instance,
        positive_templates,
        negative_templates,
    )
    mnli_data.extend(mnli_instance)

with open("resources/mnli_dataset.json", "w") as f:
    json.dump([mnli_instance.__dict__ for mnli_instance in mnli_data], f, ensure_ascii=False)
