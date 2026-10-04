from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from transformers import HfArgumentParser


class ConfigParser:
    """Loads YAML run configs with CLI overrides into typed dataclasses.

    Shared by the training and the test configs.
    """

    @staticmethod
    def load(
        path: str | Path,
        overrides: list[str] | None,
        keys: Iterable[str],
        sections: Iterable[str],
    ) -> dict[str, Any]:
        """Reads a YAML file and applies CLI overrides.

        The `modal` section is dropped: it belongs to the Modal launcher.

        Args:
            path: The YAML file.
            overrides: `--key value` pairs: `--<key>` for a top-level key or
                `--<section>.<key>` for a section key.
            keys: Allowed top-level keys for overrides.
            sections: Allowed sections for overrides.

        Returns:
            The raw values.

        Raises:
            ValueError: If an override is malformed or unknown.
        """
        values: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        # read only by the modal launcher (`src/modal/app.py`)
        values.pop("modal", None)
        overrides = overrides or []
        if len(overrides) % 2:
            raise ValueError("overrides must be --key value pairs")
        for flag, value in zip(overrides[::2], overrides[1::2]):
            if not flag.startswith("--"):
                raise ValueError(f"override {flag!r} must start with --")
            section, _, key = flag[2:].partition(".")
            if not key and section in keys:
                values[section] = value
            elif key and section in sections:
                values[section] = values.get(section) or {}
                values[section][key] = value
            else:
                raise ValueError(f"unknown override {flag!r}")
        return values

    @staticmethod
    def parse(dataclass_type: type, values: dict[str, Any]) -> Any:
        """Builds a dataclass with HF's argument parser, which converts types.

        Raises:
            ValueError: If a key is unknown or a required one is missing.
        """
        argv: list[str] = []
        for key, value in values.items():
            # null keeps the field default
            if value is None:
                continue
            argv.append(f"--{key}")
            if isinstance(value, list):
                argv.extend(str(item) for item in value)
            else:
                argv.append(str(value))
        parser = HfArgumentParser(dataclass_type)
        try:
            parsed, remaining = parser.parse_args_into_dataclasses(
                args=argv, return_remaining_strings=True
            )
        except SystemExit as error:
            raise ValueError(f"invalid {dataclass_type.__name__}: {values}") from error
        if remaining:
            raise ValueError(f"unknown {dataclass_type.__name__} keys: {remaining}")
        return parsed
