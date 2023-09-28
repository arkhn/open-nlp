import random
from abc import ABC, abstractmethod

import datasets


class Prompter(ABC):
    """Abstract class to create prompters.

    Args:
        base_prompt: The base prompt to use.
        to_dataset_label_map: Dict to convert the labels of the dataset to the labels of the model.
        from_dataset_label_map: Dict to convert the labels of the model to the labels of the
            dataset.
        n_examples: The number of examples to use during the dataset generation.
        train_dataset: The train dataset.
    """

    def __init__(
        self,
        base_prompt: str,
        to_dataset_label_map: dict[str, str],
        from_dataset_label_map: dict[str, str],
        n_examples: int,
        train_dataset: datasets.Dataset,
    ):
        self.base_prompt = base_prompt
        self.to_dataset_label_map = to_dataset_label_map
        self.from_dataset_label_map = from_dataset_label_map
        self.examples_prompt = self.create_examples_prompt(
            train_dataset=train_dataset, n_examples=n_examples
        )

    def get_prompt(self, text_input: str) -> str:
        """Get the prompt to use for the model."""
        prompt = self.base_prompt
        prompt = prompt.replace("%examples%", self.examples_prompt)
        prompt = prompt.replace("%text_input%", text_input)
        return prompt

    def create_examples_prompt(self, train_dataset: datasets.Dataset, n_examples: int) -> str:
        """Create the examples to use in the prompt from the train dataset."""
        examples = random.choices(train_dataset, k=n_examples)
        examples_prompts = []
        for example in examples:
            example_prompt = (
                f"Input: {example['text']}\nOutput: "
                f"{self.dataset_example_to_completion(example=example)}"
            )
            examples_prompts.append(example_prompt)
        return "\n\n".join(examples_prompts)

    @abstractmethod
    def dataset_example_to_completion(self, example: dict) -> str:
        """Convert an example from the dataset to a completion (i.e a generative model response)."""
        pass

    @abstractmethod
    def response_to_entities(self, response: str, input_text: str) -> list[dict]:
        """Convert the response from the model to a list of entities."""
        pass
