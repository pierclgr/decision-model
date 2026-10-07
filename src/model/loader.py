from typing import Any

from transformers import AutoProcessor, PreTrainedConfig

from src.common.prompt import PromptBuilder
from src.constants import MAX_LETTER_OPTIONS
from src.model.config import DecisionEngineConfig
from src.model.system_one import PreTrainedSystemOneModel


class ModelLoader:
    """Loads a System One model and its processor."""

    @staticmethod
    def load(name: str, **kwargs: Any) -> tuple[PreTrainedSystemOneModel, Any]:
        """Loads a trained System One model or wraps a raw HF backbone.

        Only `config.json` is read to tell them apart (`model_type`). A raw
        backbone is untrained (zero-shot) with temperature 1.

        Args:
            name: HF id or local path of a trained model (`runs/<run>/final`)
                or of a multimodal backbone (e.g. `Qwen/Qwen3.5-0.8B`).
            **kwargs: Passed to `from_pretrained` (e.g. `device_map`). `dtype`
                defaults to `PreTrainedSystemOneModel.default_dtype()`.

        Returns:
            The model and the processor.
        """
        kwargs.setdefault("dtype", PreTrainedSystemOneModel.default_dtype())
        processor: Any = AutoProcessor.from_pretrained(name)
        model_type: str | None = PreTrainedConfig.get_config_dict(name)[0].get(
            "model_type"
        )
        if model_type == DecisionEngineConfig.model_type:
            model = PreTrainedSystemOneModel.from_pretrained(name, **kwargs)
        else:
            letter_ids: list[int] = PromptBuilder(
                processor, MAX_LETTER_OPTIONS
            ).letter_ids
            model = PreTrainedSystemOneModel.from_backbone(name, letter_ids, **kwargs)
        return model, processor
