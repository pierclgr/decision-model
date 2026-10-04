"""Shared fixtures: tiny random backbone and fake processor, no downloads."""

import string

import pytest
import torch
from transformers import BatchFeature, CLIPVisionConfig, LlamaConfig, LlavaConfig

from src.model.config import DecisionModelConfig
from src.model.system_one import PreTrainedSystemOneModel

LETTER_IDS: list[int] = [ord(c) for c in string.ascii_uppercase]


class FakeTokenizer:
    """Maps a single character to its code point."""

    padding_side: str = "right"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]


class FakeProcessor:
    """Records chat messages and returns fixed input ids."""

    def __init__(self) -> None:
        self.tokenizer: FakeTokenizer = FakeTokenizer()
        self.messages: list[dict] | None = None

    def apply_chat_template(self, messages: list, **kwargs) -> BatchFeature:
        self.messages = messages
        # a list of conversations is a batch
        batch_size: int = len(messages) if isinstance(messages[0], list) else 1
        input_ids = torch.tensor([[1, 2, 3]] * batch_size)
        return BatchFeature(
            {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        )


@pytest.fixture
def backbone_config() -> LlavaConfig:
    text = LlamaConfig(
        vocab_size=300,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    vision = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
        projection_dim=16,
    )
    return LlavaConfig(
        text_config=text, vision_config=vision, image_token_index=299,
        image_seq_length=4,
    )


@pytest.fixture
def config(backbone_config: LlavaConfig) -> DecisionModelConfig:
    return DecisionModelConfig(
        backbone_config=backbone_config, option_token_ids=LETTER_IDS
    )


@pytest.fixture
def model(config: DecisionModelConfig) -> PreTrainedSystemOneModel:
    torch.manual_seed(0)
    return PreTrainedSystemOneModel(config).eval()


@pytest.fixture
def processor() -> FakeProcessor:
    return FakeProcessor()
