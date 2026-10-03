import pytest
import torch
from transformers import AutoModelForMultimodalLM

from src.config import DecisionModelConfig
from src.model import PreTrainedSystemOneModel

INPUT_IDS: torch.Tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])


def test_forward_shapes_and_softmax(model: PreTrainedSystemOneModel) -> None:
    out = model(input_ids=INPUT_IDS, num_options=3)
    assert out.logits.shape == (2, 3)
    assert out.probabilities.shape == (2, 3)
    assert torch.allclose(out.probabilities.sum(-1), torch.ones(2))


def test_forward_reads_only_letter_logits(model: PreTrainedSystemOneModel) -> None:
    full = model.backbone(input_ids=INPUT_IDS).logits[:, -1]
    out = model(input_ids=INPUT_IDS, num_options=4)
    letter_ids = model.config.option_token_ids[:4]
    assert torch.allclose(out.logits, full[:, letter_ids], atol=1e-5)


def test_temperature_flattens_distribution(config: DecisionModelConfig) -> None:
    torch.manual_seed(0)
    sharp = PreTrainedSystemOneModel(config).eval()
    config_hot = DecisionModelConfig(
        backbone_config=config.backbone_config,
        option_token_ids=config.option_token_ids,
        temperature=100.0,
    )
    hot = PreTrainedSystemOneModel(config_hot, backbone=sharp.backbone).eval()
    p_sharp = sharp(input_ids=INPUT_IDS, num_options=3).probabilities
    p_hot = hot(input_ids=INPUT_IDS, num_options=3).probabilities
    assert p_hot.max() < p_sharp.max()


def test_too_many_options_raises(model: PreTrainedSystemOneModel) -> None:
    model.config.max_options = 3
    with pytest.raises(ValueError):
        model(input_ids=INPUT_IDS, num_options=4)


def test_missing_option_token_ids_raises(backbone_config) -> None:
    config = DecisionModelConfig(backbone_config=backbone_config)
    model = PreTrainedSystemOneModel(config).eval()
    with pytest.raises(ValueError):
        model(input_ids=INPUT_IDS, num_options=2)


def test_from_backbone_matches_plain_backbone(
    backbone_config, config: DecisionModelConfig, tmp_path
) -> None:
    AutoModelForMultimodalLM.from_config(backbone_config).save_pretrained(tmp_path)
    plain = AutoModelForMultimodalLM.from_pretrained(tmp_path).eval()
    wrapped = PreTrainedSystemOneModel.from_backbone(
        str(tmp_path), config.option_token_ids
    ).eval()
    assert (
        wrapped.backbone.config._attn_implementation
        == plain.config._attn_implementation
    )
    a = plain(input_ids=INPUT_IDS).logits
    b = wrapped.backbone(input_ids=INPUT_IDS).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_save_load_round_trip(model: PreTrainedSystemOneModel, tmp_path) -> None:
    model.config.temperature = 2.5
    model.save_pretrained(tmp_path)
    loaded = PreTrainedSystemOneModel.from_pretrained(tmp_path).eval()
    assert loaded.config.temperature == 2.5
    assert loaded.config.option_token_ids == model.config.option_token_ids
    a = model(input_ids=INPUT_IDS, num_options=5).logits
    b = loaded(input_ids=INPUT_IDS, num_options=5).logits
    assert torch.allclose(a, b, atol=1e-5)
