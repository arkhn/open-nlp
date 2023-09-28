import torch
from style_transfer.models.oracles.base import Oracle
from torchmetrics.text import ROUGEScore


class RougeOracle(Oracle):
    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha)
        self.rouge = ROUGEScore()

    def __call__(self, preds: list[str], targets: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the reward for each generated text.
        We compute the reward as the negative ROUGE-L score.

        Args:
            preds: The generated texts.
            targets: The target texts.

        Returns:
            The rewards.
        """

        return (
            1
            - torch.Tensor(
                [self.rouge(pred, text)["rougeL_fmeasure"] for pred, text in zip(preds, targets)]
            )
            * self.alpha
        )
