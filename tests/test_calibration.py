import json

import datasets
import numpy as np
import pytest
import torch
from transformers import TrainingArguments

from src.data.hub import HubRecordLoader
from src.model.system_one import PreTrainedSystemOneModel
from src.training.calibration import TemperatureCalibrator
from src.training.config import CalibrationSettings

FLOAT32_MIN: float = float(np.finfo(np.float32).min)


def overconfident(scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Logits `scale` times too sharp, with the true probabilities as labels."""
    torch.manual_seed(0)
    true_logits = torch.randn(200, 4)
    labels = torch.softmax(true_logits, -1)
    return (true_logits * scale).numpy(), labels.numpy()


@pytest.mark.parametrize("scale", [3.0, 0.5])
def test_fit_recovers_temperature(scale: float) -> None:
    logits, labels = overconfident(scale)
    assert TemperatureCalibrator.fit(logits, labels) == pytest.approx(scale, rel=1e-3)


def test_fit_ignores_masked_and_padded_options() -> None:
    logits, labels = overconfident(2.0)
    # masked options (model) and Trainer padding must not change the result
    padded_logits = np.concatenate(
        [logits, np.full((200, 1), FLOAT32_MIN), np.full((200, 1), -100.0)], axis=1
    )
    padded_labels = np.concatenate(
        [labels, np.zeros((200, 1)), np.full((200, 1), -100.0)], axis=1
    )
    fitted = TemperatureCalibrator.fit(
        padded_logits.astype(np.float32), padded_labels.astype(np.float32)
    )
    assert fitted == pytest.approx(2.0, rel=1e-3)


def hub_records(n: int) -> "datasets.Dataset":
    raw = datasets.Dataset.from_dict(
        {
            "state": [json.dumps(f"s{i}") for i in range(n)],
            "questions": [json.dumps({"q": {"type": "noul", "instructions": "i"}})]
            * n,
            "image": [None] * n,
        }
    )
    return HubRecordLoader.convert(raw)


def test_split_carves_out_a_fraction() -> None:
    train, calibration = TemperatureCalibrator.split(hub_records(50), 0.1, seed=0)
    assert (len(train), len(calibration)) == (45, 5)
    train_states = {record["state"] for record in train}
    assert not train_states & {record["state"] for record in calibration}
    assert "questions" in calibration[0]


def test_split_is_deterministic() -> None:
    first = TemperatureCalibrator.split(hub_records(50), 0.2, seed=7)[1]
    second = TemperatureCalibrator.split(hub_records(50), 0.2, seed=7)[1]
    assert [r["state"] for r in first] == [r["state"] for r in second]


RECORDS: list[dict] = [
    {
        "state": "s",
        "questions": {
            "q": {"type": "noul", "instructions": "Billing?", "label": True},
            "c": {
                "type": "choice",
                "instructions": "Tone?",
                "criteria": {"calm": None, "angry": None, "sad": None},
                "label": "angry",
            },
        },
    }
] * 3


def args(tmp_path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(tmp_path), per_device_eval_batch_size=2, report_to=[],
        use_cpu=True,
    )


def test_fixed_temperature_is_set(model, processor, tmp_path) -> None:
    calibrator = TemperatureCalibrator(model, processor, args(tmp_path))
    temperature = calibrator.calibrate(CalibrationSettings(temperature=1.5), RECORDS)
    assert temperature == 1.5
    assert model.config.temperature == 1.5


def test_temperature_survives_config_only_save(model, processor, tmp_path) -> None:
    model.save_pretrained(tmp_path / "model")
    calibrator = TemperatureCalibrator(model, processor, args(tmp_path))
    calibrator.calibrate(CalibrationSettings(temperature=2.0), RECORDS)
    model.config.save_pretrained(tmp_path / "model")
    loaded = PreTrainedSystemOneModel.from_pretrained(tmp_path / "model")
    assert loaded.config.temperature == 2.0


def test_fit_sets_positive_temperature(model, processor, tmp_path) -> None:
    calibrator = TemperatureCalibrator(model, processor, args(tmp_path))
    temperature = calibrator.calibrate(CalibrationSettings(temperature="fit"), RECORDS)
    assert temperature > 0
    assert model.config.temperature == temperature
