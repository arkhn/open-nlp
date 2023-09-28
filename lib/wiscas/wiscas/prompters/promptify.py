import json
import re

import datasets

from .base import Prompter


class PromptifyPrompter(Prompter):
    def __init__(self, train_dataset: datasets.Dataset, n_examples: int = 0):
        base_prompt: str = (
            "You are a highly intelligent and accurate medical domain Named-entity "
            "recognition (NER) system. You take Passage as input and your task is "
            "to recognize and extract specific types of medical domain named entities "
            "in that given passage and classify into a set of following predefined "
            "entity types: \n['medical exam', 'medical condition', 'treatment', "
            "'date', 'frequency', 'value', 'duration', 'drug', "
            "'anatomy']. Your output format is only [{\"T\": type of entity from predefined entity "
            'types, "E": entity in the input text}},...] form, no other form. \n\n'
            "%examples% \n\n"
            "Input: %text_input%\n"
            "Output:"
        )

        to_dataset_label_map: dict[str, str] = {
            "medical exam": "EXAM/PROCEDURE",
            "medical condition": "PROBLEM/OBS",
            "treatment": "TREATMENT",
            "date": "DATE",
            "frequency": "FREQUENCY",
            "value": "VALUE",
            "duration": "DURATION",
            "drug": "DRUG",
            "anatomy": "ANAT/PHYSIO",
        }

        from_dataset_label_map: dict[str, str] = {
            "EXAM/PROCEDURE": "medical exam",
            "PROBLEM/OBS": "medical condition",
            "TREATMENT": "treatment",
            "DATE": "date",
            "FREQUENCY": "frequency",
            "VALUE": "value",
            "DURATION": "duration",
            "DRUG": "drug",
            "ANAT/PHYSIO": "anatomy",
        }

        super().__init__(
            base_prompt=base_prompt,
            to_dataset_label_map=to_dataset_label_map,
            from_dataset_label_map=from_dataset_label_map,
            train_dataset=train_dataset,
            n_examples=n_examples,
        )

    def dataset_example_to_completion(self, example: dict) -> str:
        entities = []
        for entity in example["entities"]:
            entities.append(
                f'{{"T": "{self.from_dataset_label_map[entity["label"]]}", '
                f'"E": "{example["text"][entity["start_char"]:entity["end_char"]]}"}}'
            )
        return f"[{', '.join(entities)}]"

    def response_to_entities(self, response: str, input_text: str) -> list[dict]:
        entities = []
        response_candidations = re.findall(r"(\[.*?\])", response)
        if not response_candidations:
            return []
        first_response = response_candidations[0]
        try:
            json_response = json.loads(first_response)
        except json.decoder.JSONDecodeError:
            return []

        for annotation in json_response:
            if "E" not in annotation or "T" not in annotation:
                continue
            if annotation["T"] not in self.to_dataset_label_map:
                continue
            entity = annotation["E"]
            label = self.to_dataset_label_map[annotation["T"]]
            start_char = input_text.find(entity)
            end_char = start_char + len(entity)
            entities.append({"start_char": start_char, "end_char": end_char, "label": label})
        return entities
