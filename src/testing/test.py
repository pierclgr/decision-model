"""Tests a trained System One model on a dataset split.

Runs the model of a test config on its `data.split` (with the model's saved
temperature, or the config's `temperature` override), prints the metrics
(`test_loss`, `test_accuracy`, `test_brier`, `test_ece`,
`test_seconds_per_question`, plus Trainer's `test_runtime` and speed) as JSON
and saves them to `<testing.output_dir>/test_metrics.json`.

Usage:
    uv run python -m src.testing.test configs/test/<run>.yml \\
        [--section.key value]
"""

import json
import sys
from pathlib import Path

from src.data.hub import HubRecordLoader
from src.model.system_one import PreTrainedSystemOneModel
from src.pipeline.system_one import SystemOnePipeline
from src.testing.config import TestConfig
from src.training.evaluation import SystemOneEvaluator


def main(argv: list[str] | None = None) -> None:
    """Runs the model on the test split and reports the metrics.

    Args:
        argv: Command-line arguments without the program name (default:
            `sys.argv[1:]`).
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(__doc__)
    config = TestConfig.from_yaml(argv[0], argv[1:])
    pipeline = SystemOnePipeline.from_pretrained(
        config.model, dtype=PreTrainedSystemOneModel.default_dtype()
    )
    if config.temperature is not None:
        pipeline.model.config.temperature = config.temperature
    records = HubRecordLoader(config.data.dataset).load(config.data.split)
    evaluator = SystemOneEvaluator(pipeline.model, pipeline.processor, config.testing)
    output = evaluator.predict(records)
    metrics: dict[str, float] = output.metrics
    # whole run time (data loading, prompts, forward) over the questions
    metrics["test_seconds_per_question"] = (
        metrics["test_runtime"] / len(output.label_ids)
    )
    report: str = json.dumps(metrics, indent=2)
    output_dir = Path(config.testing.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
