from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import TrainingArguments

from src.common.config_parser import ConfigParser


@dataclass
class LoraSettings:
    """LoRA settings.

    Attributes:
        r: LoRA rank.
        alpha: LoRA scaling.
    """

    r: int = 16
    alpha: int = 32


@dataclass
class DataSettings:
    """Training data on the HF Hub (kev-vision layout).

    Attributes:
        dataset: HF dataset id.
        train: Training split (slices like `train[:1000]` work).
        validation: Validation split, or None for no validation.
    """

    dataset: str
    train: str
    validation: str | None = None


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
class TrainConfig:
    """Training run configuration, loaded from a YAML file.

    The YAML has a `backbone` key, optional `seed` and `checkpoint` keys and
    the sections `lora`, `data`, `training` (any HF `TrainingArguments` key,
    except `num_train_epochs`, which is named `num_epochs`, and `seed`, which
    is top-level; `logging_steps` defaults to 1, i.e. every weight update) and
    the optional `calibration`. Values are converted to the field types, so
    `1e-4` works.

    Attributes:
        backbone: HF id or local path of the multimodal backbone.
        lora: LoRA settings.
        data: Training data.
        training: `Trainer` arguments.
        calibration: Calibration settings, or None to skip calibration.
        seed: Random seed of the run (LoRA init, data order, calibration
            split); also set as `training.seed`.
        checkpoint: `Trainer` checkpoint to resume training from (e.g.
            `runs/<run>/checkpoint-500`), or None to start fresh.
    """

    backbone: str
    lora: LoraSettings
    data: DataSettings
    training: TrainingArguments
    calibration: CalibrationSettings | None = None
    seed: int = 42
    checkpoint: str | None = None

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: list[str] | None = None
    ) -> "TrainConfig":
        """Loads the configuration from a YAML file.

        Args:
            path: The YAML file.
            overrides: CLI overrides as `--key value` pairs, with dotted keys
                for sections, e.g. `["--lora.r", "8", "--backbone", "x"]`.

        Returns:
            The configuration.

        Raises:
            ValueError: If an override is malformed, or a section or key is
                unknown or missing.
        """
        values: dict[str, Any] = ConfigParser.load(
            path,
            overrides,
            keys=["backbone", "seed", "checkpoint"],
            sections=[*SECTIONS, "calibration"],
        )
        backbone: str = str(values.pop("backbone"))
        seed: int = int(values.pop("seed", 42))
        checkpoint: Any = values.pop("checkpoint", None)
        training: dict[str, Any] = values.get("training") or {}
        values["training"] = training
        if "num_train_epochs" in training:
            raise ValueError("use training.num_epochs, not num_train_epochs")
        if "seed" in training:
            raise ValueError("use the top-level seed, not training.seed")
        training["seed"] = seed
        if "num_epochs" in training:
            training["num_train_epochs"] = training.pop("num_epochs")
        training.setdefault("logging_steps", 1)
        calibration: CalibrationSettings | None = None
        if "calibration" in values:
            try:
                calibration = CalibrationSettings(**(values.pop("calibration") or {}))
            except TypeError as error:
                raise ValueError("unknown calibration keys") from error
        sections = {
            name: ConfigParser.parse(dataclass_type, values.pop(name, None) or {})
            for name, dataclass_type in SECTIONS.items()
        }
        if values:
            raise ValueError(f"unknown keys: {sorted(values)}")
        return cls(
            backbone=backbone,
            calibration=calibration,
            seed=seed,
            checkpoint=None if checkpoint is None else str(checkpoint),
            **sections,
        )


SECTIONS: dict[str, type] = {
    "lora": LoraSettings,
    "data": DataSettings,
    "training": TrainingArguments,
}
