from dataclasses import dataclass

import torch
from torch import nn
from transformers.utils import ModelOutput


@dataclass
class DecisionEngineOutput(ModelOutput):
    """Shared output of System One and System Two models.

    Attributes:
        loss: Cross-entropy of the raw option logits, only if `labels` given.
        logits: Raw logits of the options (float32; masked options set to the
            float32 min), shape (batch, num_options).
        probabilities: Softmax of `logits / temperature`, same shape.
        errors: System Two only: whether the output had no valid answer
            block, shape (batch,). None for System One.
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None
    errors: torch.Tensor | None = None

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        temperature: float,
        labels: torch.Tensor | None = None,
        errors: torch.Tensor | None = None,
    ) -> "DecisionEngineOutput":
        """Scores the raw option logits.

        Args:
            logits: Raw option logits, shape (batch, num_options).
            temperature: Divides the logits before softmax (calibration).
            labels: Optional option indices or distributions (soft labels).
            errors: System Two errors, or None.

        Returns:
            The output, with the loss on the raw logits (the temperature is
            fitted after training).
        """
        loss: torch.Tensor | None = (
            nn.functional.cross_entropy(logits, labels) if labels is not None else None
        )
        return cls(
            loss=loss,
            logits=logits,
            probabilities=torch.softmax(logits / temperature, dim=-1),
            errors=errors,
        )
