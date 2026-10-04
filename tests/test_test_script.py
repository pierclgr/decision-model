import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.testing import test as test_script
from tests.test_evaluation import RECORDS

YAML: str = """
model: runs/x/final
data:
  dataset: org/kev
  split: test
testing:
  output_dir: {output_dir}
  per_device_eval_batch_size: 2
  use_cpu: true
  report_to: none
"""


def run(
    model, processor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    overrides: list[str],
) -> None:
    config = tmp_path / "test.yml"
    config.write_text(YAML.format(output_dir=tmp_path / "out"))
    pipeline = SimpleNamespace(model=model, processor=processor)
    monkeypatch.setattr(
        test_script.SystemOnePipeline, "from_pretrained", lambda *a, **k: pipeline
    )
    monkeypatch.setattr(test_script.HubRecordLoader, "load", lambda self, s: RECORDS)
    test_script.main([str(config), *overrides])


def test_reports_seconds_per_question(
    model, processor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(model, processor, tmp_path, monkeypatch, [])
    metrics = json.loads((tmp_path / "out" / "test_metrics.json").read_text())
    # 3 records x 2 questions
    assert metrics["test_seconds_per_question"] == pytest.approx(
        metrics["test_runtime"] / 6
    )


@pytest.mark.parametrize(
    ("overrides", "expected"), [([], 2.0), (["--temperature", "1.5"], 1.5)]
)
def test_temperature_override(
    model, processor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    overrides: list[str], expected: float,
) -> None:
    # stands for the fitted temperature saved in config.json
    model.config.temperature = 2.0
    run(model, processor, tmp_path, monkeypatch, overrides)
    assert model.config.temperature == expected
