from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract class to create models to use with prompters."""

    @abstractmethod
    def run(self, prompt: str) -> str:
        """Run the model on `prompt`."""
        pass
