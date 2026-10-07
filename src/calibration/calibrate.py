"""Calibration of the global temperature of a System One model.

Run standalone to calibrate a trained model, e.g. after quantization, with a
calibration config (`configs/calibration/`). To fit, it rebuilds the
calibration set carved out of `data.train` at training time: the config's
`seed`, `data` and `calibration.split` must match the training config, or the
fit sees records the model was trained on. Only `config.json` of the model is
rewritten, not the weights.

Usage:
    uv run python -m src.calibration.calibrate configs/calibration/<run>.yml \\
        [--section.key value]
"""

import sys
from typing import Any

import datasets
import numpy as np
import torch
from transformers import TrainingArguments

from src.calibration.config import CalibrationConfig, CalibrationSettings
from src.common.prompt import PromptBuilder
from src.data.hub import HubRecordLoader
from src.evaluation.evaluator import DecisionEvaluator
from src.model.loader import ModelLoader
from src.model.system_one import PreTrainedSystemOneModel


class TemperatureCalibrator:
    """Sets the global temperature of a System One model (temperature scaling).

    Args:
        model: The trained System One model (its config is updated).
        processor: Processor matching the backbone.
        training_args: `Trainer` arguments, used for the prediction run
            (device, eval batch size, precision).
    """

    def __init__(
        self,
        model: PreTrainedSystemOneModel,
        processor: Any,
        training_args: TrainingArguments,
    ) -> None:
        self.model: PreTrainedSystemOneModel = model
        self.evaluator: DecisionEvaluator = DecisionEvaluator(
            model, PromptBuilder(processor, model.config.max_options), training_args
        )

    def calibrate(self, settings: CalibrationSettings, records: Any) -> float:
        """Sets `model.config.temperature` and returns it.

        Args:
            settings: `"fit"` or a fixed temperature.
            records: The calibration records from `split` (used only to fit).

        Returns:
            The temperature.
        """
        if settings.temperature == "fit":
            output = self.evaluator.predict(records)
            # predictions are (logits, probabilities): the fit uses the raw logits
            temperature: float = self.fit(output.predictions[0], output.label_ids)
        else:
            temperature = float(settings.temperature)
        self.model.config.temperature = temperature
        return temperature

    @staticmethod
    def split(
        records: datasets.Dataset, fraction: float, seed: int
    ) -> tuple[datasets.Dataset, datasets.Dataset]:
        """Carves a calibration set out of the training records.

        Whole records are split, so questions about the same state never end
        up on both sides. The same seed gives the same split, so a later
        calibration run finds the same calibration set.

        Args:
            records: The training records.
            fraction: Fraction of records for calibration, in (0, 1).
            seed: Random seed (use the config's `seed`).

        Returns:
            The remaining training records and the calibration records.
        """
        parts = records.train_test_split(test_size=fraction, seed=seed)
        return parts["train"], parts["test"]

    @staticmethod
    def fit(logits: np.ndarray, labels: np.ndarray) -> float:
        """Returns the temperature that minimizes the NLL of the labels.

        Args:
            logits: Raw option logits, shape (questions, options). Masked
                options hold the float32 min, `Trainer` padding holds -100.
            labels: Option distributions, same shape (`Trainer` padding -100).

        Returns:
            The fitted temperature.
        """
        logits_t = torch.from_numpy(logits).double()
        targets = torch.from_numpy(labels).double()
        # real options: not Trainer padding, not masked by the model
        mask = (targets >= 0) & (logits_t > torch.finfo(torch.float32).min / 2)
        logits_t = logits_t.masked_fill(~mask, 0.0)
        targets = targets.masked_fill(~mask, 0.0)
        log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_temperature], max_iter=100, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = (logits_t / log_temperature.exp()).masked_fill(~mask, -torch.inf)
            log_probs = torch.log_softmax(scaled, dim=-1).masked_fill(~mask, 0.0)
            loss = -(targets * log_probs).sum(-1).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(log_temperature.detach().exp())


def main(argv: list[str] | None = None) -> None:
    """Fits or sets the temperature and saves it in `<model>/config.json`.

    Args:
        argv: Command-line arguments without the program name (default:
            `sys.argv[1:]`).
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(__doc__)
    config = CalibrationConfig.from_yaml(argv[0], argv[1:])
    model, processor = ModelLoader.load(config.model)
    records = None
    if config.calibration.temperature == "fit":
        _, records = TemperatureCalibrator.split(
            HubRecordLoader(config.data.dataset).load(config.data.train),
            config.calibration.split,
            config.seed,
        )
    calibrator = TemperatureCalibrator(model, processor, config.calibrating)
    temperature: float = calibrator.calibrate(config.calibration, records)
    model.config.save_pretrained(config.model)
    print(f"temperature: {temperature:.4f}")


if __name__ == "__main__":
    main()
