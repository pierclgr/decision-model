from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import TrainingArguments

from src.common.config_parser import ConfigParser


@dataclass
class TestDataSettings:
    """Test data on the HF Hub (kev-vision layout).

    Attributes:
        dataset: HF dataset id.
        split: Test split (slices like `test[:500]` work).
    """

    # not a pytest test class
    __test__ = False

    dataset: str
    split: str


@dataclass
class TestConfig:
    """Test run configuration, loaded from a YAML file.

    The YAML has a `model` key, an optional `temperature` key and the sections `data` and `testing` (any HF
    `TrainingArguments` key, e.g. `per_device_eval_batch_size`, `bf16`;
    `output_dir` is where `test_metrics.json` is written). Values are
    converted to the field types.

    Attributes:
        model: Path to the trained model (weights + processor), e.g.
            `runs/<run>/final`.
        data: Test data.
        testing: `Trainer` arguments for the prediction run.
        temperature: Temperature instead of the model's saved one (e.g. 1.0
            for uncalibrated scores), or None to use the saved one.

    Raises:
        ValueError: If the temperature is not > 0.
    """

    # not a pytest test class
    __test__ = False

    model: str
    data: TestDataSettings
    testing: TrainingArguments
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and self.temperature <= 0:
            raise ValueError("temperature must be > 0")

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: list[str] | None = None
    ) -> "TestConfig":
        """Loads the configuration from a YAML file.

        Args:
            path: The YAML file.
            overrides: CLI overrides as `--key value` pairs, with dotted keys
                for sections, e.g. `["--data.split", "test_ood"]`.

        Returns:
            The configuration.

        Raises:
            ValueError: If an override is malformed, or a section or key is
                unknown or missing.
        """
        values: dict[str, Any] = ConfigParser.load(
            path, overrides, keys=["model", "temperature"], sections=["data", "testing"]
        )
        model: str = str(values.pop("model"))
        temperature: Any = values.pop("temperature", None)
        data = ConfigParser.parse(TestDataSettings, values.pop("data", None) or {})
        testing = ConfigParser.parse(
            TrainingArguments, values.pop("testing", None) or {}
        )
        if values:
            raise ValueError(f"unknown keys: {sorted(values)}")
        return cls(
            model=model,
            data=data,
            testing=testing,
            temperature=None if temperature is None else float(temperature),
        )
