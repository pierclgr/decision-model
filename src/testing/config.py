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


# System Two thinking: off, or the reasoning effort (Qwen3.8 template levels)
THINKING: tuple[str, ...] = ("off", "low", "medium", "xhigh")


@dataclass
class SystemTwoSettings:
    """System Two mode: the model generates its answer (see `SystemTwoModel`).

    Attributes:
        thinking: `off`, or a reasoning effort `low`, `medium`, `xhigh`
            (thinking on; Qwen3.5 has no levels, any of them is just on).
            YAML reads a bare `off` as false, which is accepted as `off`.
        max_new_tokens: Cap on generated tokens, thinking included. Each
            question stops at `</answer>` or the model's end token; the cap
            only stops outputs that never do. Default: 256 with thinking off
            (an answer block is about 10 tokens), 32768 with thinking (Qwen's
            suggested output length for thinking).

    Raises:
        ValueError: If `thinking` is not one of the levels.
    """

    thinking: str = "off"
    max_new_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.thinking is False:
            self.thinking = "off"
        if self.thinking not in THINKING:
            raise ValueError(f"thinking must be one of {THINKING}")
        if self.max_new_tokens is None:
            self.max_new_tokens = 256 if self.thinking == "off" else 32768
        self.max_new_tokens = int(self.max_new_tokens)


@dataclass
class TestConfig:
    """Test run configuration, loaded from a YAML file.

    The YAML has a `model` key, an optional `temperature` key, the sections
    `data` and `testing` (any HF
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
        system_two: System Two settings (optional `system_two` section, even
            empty), or None for System One.

    Raises:
        ValueError: If the temperature is not > 0.
    """

    # not a pytest test class
    __test__ = False

    model: str
    data: TestDataSettings
    testing: TrainingArguments
    temperature: float | None = None
    system_two: SystemTwoSettings | None = None

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
            path,
            overrides,
            keys=["model", "temperature"],
            sections=["data", "testing", "system_two"],
        )
        model: str = str(values.pop("model"))
        temperature: Any = values.pop("temperature", None)
        system_two: SystemTwoSettings | None = ConfigParser.settings(
            values, "system_two", SystemTwoSettings
        )
        data = ConfigParser.section(values, "data", TestDataSettings)
        testing = ConfigParser.section(values, "testing", TrainingArguments)
        ConfigParser.check_empty(values)
        return cls(
            model=model,
            data=data,
            testing=testing,
            temperature=None if temperature is None else float(temperature),
            system_two=system_two,
        )
