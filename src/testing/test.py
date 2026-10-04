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
from typing import Any

from transformers import PreTrainedConfig

from src.common.prompt import SYSTEM_TWO_INSTRUCTION, PromptBuilder
from src.data.collator import SystemOneCollator
from src.data.hub import HubRecordLoader
from src.model.config import DecisionModelConfig
from src.model.system_one import PreTrainedSystemOneModel
from src.model.system_two import SystemTwoModel
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
    # only config.json is read: a trained System One model, or a raw backbone
    # (untrained, zero-shot, temperature 1)
    model_type: str | None = PreTrainedConfig.get_config_dict(config.model)[0].get(
        "model_type"
    )
    load = (
        SystemOnePipeline.from_pretrained
        if model_type == DecisionModelConfig.model_type
        else SystemOnePipeline.from_backbone
    )
    pipeline = load(config.model, dtype=PreTrainedSystemOneModel.default_dtype())
    if config.temperature is not None:
        pipeline.model.config.temperature = config.temperature
    records = HubRecordLoader(config.data.dataset).load(config.data.split)
    model: PreTrainedSystemOneModel | SystemTwoModel = pipeline.model
    collator: SystemOneCollator | None = None
    if config.system_two is not None:
        thinking: str = config.system_two.thinking
        tokenizer: Any = pipeline.processor.tokenizer
        model = SystemTwoModel(
            pipeline.model,
            tokenizer,
            config.system_two.max_new_tokens,
            thinking != "off",
        )
        template_kwargs: dict[str, Any] = {"enable_thinking": thinking != "off"}
        # only if the template has levels (Qwen3.8, not Qwen3.5): transformers
        # turns a variable the template does not use into processor kwargs,
        # which replace `processor_kwargs` (padding)
        if thinking != "off" and "reasoning_effort" in (
            pipeline.processor.chat_template or ""
        ):
            template_kwargs["reasoning_effort"] = thinking
        collator = SystemOneCollator(
            pipeline.processor,
            PromptBuilder(
                tokenizer, pipeline.model.config.max_options, SYSTEM_TWO_INSTRUCTION
            ),
            template_kwargs,
        )
    evaluator = SystemOneEvaluator(
        model, pipeline.processor, config.testing, collator
    )
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
