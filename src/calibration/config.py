from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import TrainingArguments

from src.common.config_parser import ConfigParser
from src.data.config import DataSettings


@dataclass
class CalibrationSettings:
    """Calibration of the global temperature.

    Attributes:
        temperature: `"fit"` to fit it on a calibration set carved out of the
            training split, or a fixed value (> 0).
        split: Fraction of the training records carved out for the fit
            (0 < split < 1). Not used with a fixed temperature.

    Raises:
        ValueError: If the temperature is neither `"fit"` nor a positive
            number, or the split is not in (0, 1).
    """

    temperature: float | str = "fit"
    split: float = 0.1

    def __post_init__(self) -> None:
        self.split = float(self.split)
        if not 0 < self.split < 1:
            raise ValueError("calibration split must be in (0, 1)")
        if self.temperature == "fit":
            return
        try:
            self.temperature = float(self.temperature)
        except ValueError as error:
            raise ValueError("temperature must be 'fit' or a number") from error
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")


@dataclass
class CalibrationConfig:
    """Standalone calibration configuration, loaded from a YAML file.

    Fits or sets the temperature of a trained model. To fit, the calibration
    set carved out of `data.train` at training time is rebuilt, so `seed`,
    `data` and `calibration.split` must match the training config.

    The YAML has a `model` key, an optional `seed` key and the sections
    `data`, `calibration` (optional, defaults as in `CalibrationSettings`) and
    `calibrating` (any HF `TrainingArguments` key, for the prediction run).

    Attributes:
        model: Path to the trained model (weights + processor), e.g.
            `runs/<run>/final`. Its `config.json` gets the temperature.
        data: The training data (`dataset`, `train`; `validation` unused).
        calibration: Calibration settings.
        calibrating: `Trainer` arguments for the prediction run.
        seed: Seed of the training run (rebuilds the calibration split).
    """

    model: str
    data: DataSettings
    calibration: CalibrationSettings
    calibrating: TrainingArguments
    seed: int = 42

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: list[str] | None = None
    ) -> "CalibrationConfig":
        """Loads the configuration from a YAML file.

        Args:
            path: The YAML file.
            overrides: CLI overrides as `--key value` pairs, with dotted keys
                for sections, e.g. `["--calibration.temperature", "1.5"]`.

        Returns:
            The configuration.

        Raises:
            ValueError: If an override is malformed, or a section or key is
                unknown or missing.
        """
        values: dict[str, Any] = ConfigParser.load(
            path,
            overrides,
            keys=["model", "seed"],
            sections=["data", "calibration", "calibrating"],
        )
        model: str = str(values.pop("model"))
        seed: int = int(values.pop("seed", cls.seed))
        calibration: CalibrationSettings = ConfigParser.settings(
            values, "calibration", CalibrationSettings
        ) or CalibrationSettings()
        data = ConfigParser.section(values, "data", DataSettings)
        calibrating = ConfigParser.section(values, "calibrating", TrainingArguments)
        ConfigParser.check_empty(values)
        return cls(
            model=model,
            data=data,
            calibration=calibration,
            calibrating=calibrating,
            seed=seed,
        )
