from pathlib import Path

import pytest

from src.testing.config import TestConfig

YAML: str = """
model: runs/x/final
data:
  dataset: Jacqkues/kev-vision-decisions-full
  split: test
testing:
  output_dir: runs/x/test
  per_device_eval_batch_size: 8
  report_to: none
"""


def write(tmp_path: Path, text: str = YAML) -> Path:
    path = tmp_path / "test.yml"
    path.write_text(text)
    return path


def test_loads_typed_sections(tmp_path: Path) -> None:
    config = TestConfig.from_yaml(write(tmp_path))
    assert config.model == "runs/x/final"
    assert config.data.dataset == "Jacqkues/kev-vision-decisions-full"
    assert config.data.split == "test"
    assert config.testing.output_dir == "runs/x/test"
    assert config.testing.per_device_eval_batch_size == 8
    assert config.testing.report_to == []


def test_cli_overrides(tmp_path: Path) -> None:
    overrides = [
        "--model", "runs/y/final",
        "--data.split", "test_ood",
        "--testing.per_device_eval_batch_size", "2",
    ]
    config = TestConfig.from_yaml(write(tmp_path), overrides)
    assert config.model == "runs/y/final"
    assert config.data.split == "test_ood"
    assert config.testing.per_device_eval_batch_size == 2


@pytest.mark.parametrize(
    "overrides",
    [["--backbone", "x"], ["--training.seed", "1"], ["--data.train", "x"]],
)
def test_bad_overrides_raise(tmp_path: Path, overrides: list[str]) -> None:
    with pytest.raises(ValueError):
        TestConfig.from_yaml(write(tmp_path), overrides)


def test_unknown_section_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TestConfig.from_yaml(write(tmp_path, YAML + "lora:\n  r: 8\n"))


def test_missing_split_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TestConfig.from_yaml(write(tmp_path, YAML.replace("  split: test\n", "")))


@pytest.mark.parametrize("path", sorted(Path("configs/test").glob("*.yml")))
def test_shipped_configs_load(path: Path) -> None:
    TestConfig.from_yaml(path)


def test_modal_section_is_skipped(tmp_path: Path) -> None:
    config = TestConfig.from_yaml(write(tmp_path, YAML + "modal:\n  gpu: H100\n"))
    assert config.model == "runs/x/final"


def test_temperature_override(tmp_path: Path) -> None:
    assert TestConfig.from_yaml(write(tmp_path)).temperature is None
    config = TestConfig.from_yaml(write(tmp_path, YAML + "temperature: 1.5\n"))
    assert config.temperature == 1.5
    overridden = TestConfig.from_yaml(write(tmp_path), ["--temperature", "1"])
    assert overridden.temperature == 1.0


@pytest.mark.parametrize("value", ["0", "-1"])
def test_bad_temperature_raises(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError):
        TestConfig.from_yaml(write(tmp_path), ["--temperature", value])
