"""RE Dataset, Arkhn style."""
import itertools
import os
import zipfile
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Optional

import datasets
from cassis import Cas, load_cas_from_xmi, load_typesystem

# You can copy an official description
_DESCRIPTION = (
    "This dataset is designed to solve the great task of Relation Extraction and "
    "is crafted with a lot of care."
)

SENTENCE_CAS = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
CUSTOM_RELATION_CAS = "custom.Relation"
DOCUMENT_METADATA = "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData"
CUSTOM_SPAN = "custom.Span"


@dataclass
class ReMedicalAnnotationsConfig(datasets.BuilderConfig):
    """BuilderConfig for ReMedicalAnnotations dataset."""

    labels: Optional[list[str]] = None


class ReMedicalAnnotations(datasets.GeneratorBasedBuilder):
    """This dataset is designed to solve the great task of Relation Extraction and is crafted
    with a lot of care."""

    BUILDER_CONFIG_CLASS = ReMedicalAnnotationsConfig
    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "subj_start": datasets.Value("int32"),
                    "subj_end": datasets.Value("int32"),
                    "subj_type": datasets.Value("string"),
                    "obj_start": datasets.Value("int32"),
                    "obj_end": datasets.Value("int32"),
                    "obj_type": datasets.Value("string"),
                    "relation": datasets.ClassLabel(names=["no_relation"] + self.config.labels),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.extract(self.config.data_dir)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "all",
                },
            )
        ]

    @staticmethod
    def get_cas_objects(filepath: str):
        cas_objects: list[Cas] = []

        curation_path = os.path.join(filepath, "curation")
        for zip_subset_path in sorted(glob(curation_path + "/**/*.zip")):
            with zipfile.ZipFile(zip_subset_path) as zip_subset:
                subset_folder = str(Path(zip_subset_path).parent)
                zip_subset.extractall(subset_folder)
                with open(glob(subset_folder + "/*.xml")[0], "rb") as f:
                    typesystem = load_typesystem(f)
                with open(glob(subset_folder + "/*.xmi")[0], "rb") as f:
                    cas = load_cas_from_xmi(f, typesystem=typesystem)
                cas_objects.append(cas)

        return cas_objects

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath: str, split: str):
        """Generate RE examples from an unzipped Inception dataset."""
        key = 0
        for cas in self.get_cas_objects(filepath=filepath):
            examples = cas.select(SENTENCE_CAS)
            for example in examples:
                offset = (
                    cas.select(DOCUMENT_METADATA)[0]
                    .get_covered_text()
                    .find(example.get_covered_text())
                )

                # retrieve all the relations as tuples (dependant, governor) with the span ids
                # and the corresponding relation label
                relations = {}
                for relation in cas.select_covered(CUSTOM_RELATION_CAS, example):
                    relations[(relation.Dependent.xmiID, relation.Governor.xmiID)] = relation.label

                # Create all possible combinations of 2 entities (we keep in undirected for now)
                entities = cas.select_covered(CUSTOM_SPAN, example)
                combinations = itertools.combinations(entities, 2)
                for ent1, ent2 in combinations:
                    if (ent1.xmiID, ent2.xmiID) in relations.keys():
                        relation = relations[(ent1.xmiID, ent2.xmiID)]
                    else:
                        relation = "no_relation"
                    yield key, {
                        "text": example.get_covered_text(),
                        "subj_start": ent1.begin - offset,
                        "subj_end": ent1.end - offset,
                        "subj_type": ent1.label,
                        "obj_start": ent2.begin - offset,
                        "obj_end": ent2.end - offset,
                        "obj_type": ent2.label,
                        "relation": relation,
                    }
                    key += 1
