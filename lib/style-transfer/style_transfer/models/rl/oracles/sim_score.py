import torch
from sentence_transformers import SentenceTransformer, util
from style_transfer.models.oracles.base import Oracle


class SimScoreOracle(Oracle):
    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def __call__(self, preds: list[str], targets: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the reward for each generated text.
        We compute the reward as the semantic similarity score.

        Args:
            preds: The generated texts.
            targets: The target texts.

        Returns:
            The rewards.
        """
        similarities = [
            util.cos_sim(
                self.model.encode(pred, show_progress_bar=False),
                self.model.encode(target, show_progress_bar=False),
            )
            for target, pred in zip(targets, preds)
        ]
        return torch.stack(similarities).squeeze([1, 2]) * self.alpha
