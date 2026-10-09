"""Tests a System One model on a dataset split.

`model` is a trained System One model (`runs/<run>/final`), or a raw HF
backbone (e.g. `Qwen/Qwen3.5-0.8B`) for zero-shot scores; `model_type` in its
`config.json` tells them apart. With a `system_two` section, the model
generates its answer instead (System Two: `<answer>` block, optional
thinking; adds `test_errors`). Runs it on the config's `data.split` (with
the model's saved temperature, 1 for a backbone, or the config's
`temperature` override), prints the metrics
(`test_loss`, `test_accuracy`, `test_brier`, `test_ece`,
`test_seconds_per_question`, `test_gpu`, plus Trainer's `test_runtime` and
speed) as JSON
and saves them to `<testing.output_dir>/test_metrics.json`.
`test_seconds_per_question` skips the first `WARMUP_STEPS` batches (GPU
start-up), so it needs more batches than that.

Usage:
    uv run python -m src.testing.test configs/test/<run>.yml \\
        [--section.key value]
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

from src.common.prompt import PromptBuilder
from src.data.hub import HubRecordLoader
from src.evaluation.evaluator import DecisionEvaluator
from src.model.loader import ModelLoader
from src.model.system_one import PreTrainedSystemOneModel
from src.model.system_two import SystemTwoModel
from src.testing.config import SAMPLE_SEED, TestConfig

# batches left out of the timing
WARMUP_STEPS: int = 10


class SteadyTimer(TrainerCallback):
    """Times the prediction batches after the first `warmup` ones.

    Args:
        warmup: Batches left out of the timing.

    Attributes:
        steps: Batches seen.
        start: Time at the end of the last warm-up batch (None before).
        end: Time at the end of the last batch.
    """

    def __init__(self, warmup: int) -> None:
        self.warmup: int = warmup
        self.steps: int = 0
        self.start: float | None = None
        self.end: float = 0.0

    def on_prediction_step(self, *args: object, **kwargs: object) -> None:
        """Records the time after each batch, once the GPU work is done."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.steps += 1
        self.end = time.perf_counter()
        if self.steps == self.warmup:
            self.start = self.end


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
    if config.data.sample is not None:
        records = records.shuffle(seed=SAMPLE_SEED).select(range(config.data.sample))
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
    evaluator = DecisionEvaluator(model, prompt, config.testing)
    timer = SteadyTimer(WARMUP_STEPS)
    evaluator.trainer.add_callback(timer)
    output = evaluator.predict(records)
    metrics: dict[str, float | str] = output.metrics
    # the hardware of the times (a cloud may give another gpu than asked)
    if torch.cuda.is_available():
        metrics["test_gpu"] = torch.cuda.get_device_name()
    # time after warm-up (data loading, prompts, forward) over its questions
    warmup_questions: int = WARMUP_STEPS * config.testing.per_device_eval_batch_size
    metrics["test_seconds_per_question"] = (timer.end - timer.start) / (
        len(output.label_ids) - warmup_questions
    )
    report: str = json.dumps(metrics, indent=2)
    output_dir = Path(config.testing.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
