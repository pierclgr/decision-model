from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForMultimodalLM, PreTrainedModel
from transformers.utils import ModelOutput

from src.model.config import DecisionModelConfig


@dataclass
class SystemOneOutput(ModelOutput):
    """Output of `PreTrainedSystemOneModel`.

    Attributes:
        loss: Cross-entropy of the raw option logits, only if `labels` given.
        logits: Raw logits of the options (float32; masked options set to the
            float32 min), shape (batch, num_options).
        probabilities: Softmax of `logits / temperature`, same shape.
    """

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None


class PreTrainedSystemOneModel(PreTrainedModel):
    """System One decision model: one forward pass, no generation.

    Wraps any HF multimodal causal LM. The probabilities come from the logits
    of the option letters at the first answer position.

    Args:
        config: Model configuration (holds the backbone config).
        backbone: Optional already loaded backbone. If None, it is built from
            `config.backbone_config` (random weights, loaded afterwards by
            `from_pretrained`).
    """

    config: DecisionModelConfig
    base_model_prefix = "backbone"
    input_modalities = ("image", "text")
    # enabled on the backbone layers (`TrainingArguments.gradient_checkpointing`)
    supports_gradient_checkpointing = True
    # the loss is a plain mean: Trainer must scale it for gradient accumulation
    accepts_loss_kwargs = False

    def __init__(
        self, config: DecisionModelConfig, backbone: nn.Module | None = None
    ) -> None:
        super().__init__(config)
        self.backbone: nn.Module = (
            backbone
            if backbone is not None
            else AutoModelForMultimodalLM.from_config(config.backbone_config)
        )
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        # the backbone owns all weights and initializes itself
        pass

    @classmethod
    def from_backbone(
        cls,
        backbone_id: str,
        option_token_ids: list[int],
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> "PreTrainedSystemOneModel":
        """Builds the model around a pretrained backbone.

        Args:
            backbone_id: HF id or local path of the multimodal backbone.
            option_token_ids: Token ids of the option letters (A, B, ...).
            temperature: Divides the option logits before softmax.
            **kwargs: Passed to `AutoModelForMultimodalLM.from_pretrained`
                (e.g. `dtype`, `device_map`).

        Returns:
            The wrapped model.
        """
        backbone = AutoModelForMultimodalLM.from_pretrained(backbone_id, **kwargs)
        attn_implementation = backbone.config._attn_implementation
        config = DecisionModelConfig(
            backbone_config=backbone.config,
            temperature=temperature,
            max_options=len(option_token_ids),
            option_token_ids=option_token_ids,
        )
        model = cls(config, backbone=backbone)
        # building the config resets the backbone attention implementation
        backbone.config._attn_implementation = attn_implementation
        return model

    @staticmethod
    def default_dtype() -> torch.dtype:
        """Returns the load dtype: bf16 on CUDA, float32 elsewhere."""
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_options: int | None = None,
        labels: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        **backbone_kwargs: Any,
    ) -> SystemOneOutput:
        """Scores the options of a question.

        Args:
            input_ids: Prompt token ids, shape (batch, seq). The last position
                is the answer slot (use left padding).
            attention_mask: Attention mask of the prompt.
            num_options: Number of options (default: `config.max_options`).
            labels: Optional targets: option indices, shape (batch,), or
                option distributions (soft labels), shape
                (batch, num_options).
            option_mask: Optional bool mask of the real options, shape
                (batch, num_options), for batches with different option
                counts. Masked options get probability 0 and no loss.
            **backbone_kwargs: Extra inputs for the backbone (e.g.
                `pixel_values`, `image_grid_thw`).

        Returns:
            Loss (if `labels` given), logits and probabilities of the options.

        Raises:
            ValueError: If the letter token ids are not set or there are more
                options than `config.max_options`.
        """
        num_options = num_options or self.config.max_options
        if num_options > self.config.max_options:
            raise ValueError(f"{num_options} options, max is {self.config.max_options}")
        if self.config.option_token_ids is None:
            raise ValueError("config.option_token_ids is not set")

        # lm_head runs on the last position only
        logits: torch.Tensor = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=1,
            **backbone_kwargs,
        ).logits
        letter_ids: list[int] = self.config.option_token_ids[:num_options]
        option_logits: torch.Tensor = logits[:, -1, letter_ids].float()
        if option_mask is not None:
            # finite min, not -inf: soft-label loss would give 0 * -inf = nan
            option_logits = option_logits.masked_fill(
                ~option_mask, torch.finfo(option_logits.dtype).min
            )
        probabilities: torch.Tensor = torch.softmax(
            option_logits / self.config.temperature, dim=-1
        )
        # raw logits: the temperature is fitted after training
        loss: torch.Tensor | None = (
            nn.functional.cross_entropy(option_logits, labels)
            if labels is not None
            else None
        )
        return SystemOneOutput(
            loss=loss, logits=option_logits, probabilities=probabilities
        )
