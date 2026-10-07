from typing import Any

import torch

from src.common.prompt import PromptBuilder


class DecisionEngineCollator:
    """Turns `DecisionEngineDataset` items into a batch for System One or Two.

    The prompt is built and encoded as in `SystemOnePipeline`. Questions with
    fewer options than the largest one in the batch get zero labels and a
    False mask on the missing options.

    Args:
        prompt: Prompt builder (same as inference).
    """

    def __init__(self, prompt: PromptBuilder) -> None:
        self.prompt: PromptBuilder = prompt

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Builds the batch.

        Returns:
            The processor outputs (`input_ids`, `attention_mask`, image
            tensors) plus `labels` (batch, max options), `option_mask` (same
            shape) and `num_options`.
        """
        batch = self.prompt.encode(
            [
                self.prompt.build_messages(
                    item["state"], item["question"], item["images"]
                )
                for item in items
            ]
        )
        num_options: int = max(len(item["target"]) for item in items)
        labels = torch.zeros(len(items), num_options)
        option_mask = torch.zeros(len(items), num_options, dtype=torch.bool)
        for row, item in enumerate(items):
            n: int = len(item["target"])
            labels[row, :n] = torch.tensor(item["target"])
            option_mask[row, :n] = True
        return {
            **batch,
            "labels": labels,
            "option_mask": option_mask,
            "num_options": num_options,
        }
