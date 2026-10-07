from huggingface_hub.dataclasses import strict
from transformers import AutoConfig, PreTrainedConfig
from transformers.models.auto import CONFIG_MAPPING

from src.constants import MAX_LETTER_OPTIONS


@strict
class DecisionEngineConfig(PreTrainedConfig):
    """Configuration of `PreTrainedSystemOneModel`.

    Attributes:
        backbone_config: Config of the multimodal backbone (config or dict).
        temperature: Divides the option logits before softmax.
        max_options: Max options per question (one letter each, A-Z).
        option_token_ids: Token ids of the option letters (A, B, ...), set when
            the model is built so they are saved with it.
    """

    model_type = "system_one"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    temperature: float = 1.0
    max_options: int = MAX_LETTER_OPTIONS
    option_token_ids: list[int] | None = None

    def __post_init__(self, **kwargs) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 2 <= self.max_options <= MAX_LETTER_OPTIONS:
            raise ValueError(f"max_options must be in [2, {MAX_LETTER_OPTIONS}]")
        if isinstance(self.backbone_config, dict):
            model_type: str = self.backbone_config["model_type"]
            self.backbone_config = CONFIG_MAPPING[model_type](**self.backbone_config)
        super().__post_init__(**kwargs)
