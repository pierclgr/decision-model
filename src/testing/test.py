"""Tests a System One model on a dataset split.

`model` is a trained System One model (`runs/<run>/final`), or a raw HF
backbone (e.g. `Qwen/Qwen3.5-0.8B`) for zero-shot scores; `model_type` in its
`config.json` tells them apart. With a `system_two` section, the model
generates its answer instead (System Two: `<answer>` block, optional
thinking; adds `test_errors`). Runs it on the config's `data.split` (with
the model's saved temperature, 1 for a backbone, or the config's
`temperature` override), prints the metrics
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

from src.common.prompt import PromptBuilder
from src.data.hub import HubRecordLoader
from src.evaluation.evaluator import DecisionEvaluator
from src.model.loader import ModelLoader
from src.model.system_one import PreTrainedSystemOneModel
from src.model.system_two import SystemTwoModel
from src.testing.config import TestConfig


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
    system_one, processor = ModelLoader.load(config.model)
    if config.temperature is not None:
        system_one.config.temperature = config.temperature
    records = HubRecordLoader(config.data.dataset).load(config.data.split)
    model: PreTrainedSystemOneModel | SystemTwoModel = system_one
    prompt = PromptBuilder(processor, system_one.config.max_options)
    if config.system_two is not None:
        model = SystemTwoModel.from_system_one(
            system_one,
            processor,
            config.system_two.thinking,
            config.system_two.max_new_tokens,
        )
        prompt = model.prompt
    output = DecisionEvaluator(model, prompt, config.testing).predict(records)
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
