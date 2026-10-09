"""Tests a System One model on a dataset split.

`model` is a trained System One model (`runs/<run>/final`), or a raw HF
backbone (e.g. `Qwen/Qwen3.5-0.8B`) for zero-shot scores; `model_type` in its
`config.json` tells them apart. With a `system_two` section, the model
generates its answer instead (System Two: `<answer>` block, optional
thinking; adds `test_errors`). Runs it on the config's `data.split` (with
the model's saved temperature, 1 for a backbone, or the config's
`temperature` override), prints the metrics
(`test_loss`, `test_accuracy`, `test_brier`, `test_ece`,
`test_seconds_per_question`, `test_gpu`) as JSON
and saves them to `<testing.output_dir>/test_metrics.json`.
`test_seconds_per_question` is the model's time only (System One: forward
pass; System Two: generation up to the final answer), without data loading,
and skips the first `WARMUP_STEPS` batches (GPU start-up), so it needs more
batches than that.

Usage:
    uv run python -m src.testing.test configs/test/<run>.yml \\
        [--section.key value]
"""

import json
import sys
import time
from pathlib import Path

import torch

from src.common.prompt import PromptBuilder
from src.data.hub import HubRecordLoader
from src.evaluation.evaluator import DecisionEvaluator
from src.model.loader import ModelLoader
from src.model.system_one import PreTrainedSystemOneModel
from src.model.system_two import SystemTwoModel
from src.testing.config import SAMPLE_SEED, TestConfig

# model calls (batches) left out of the timing
WARMUP_STEPS: int = 10


class ModelTimer:
    """Times each call of a model (System One: the forward pass; System Two:
    the generation up to the final answer), from its GPU inputs to its output.

    Args:
        model: The model the Trainer calls.

    Attributes:
        times: Seconds of each call, in order.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.times: list[float] = []
        self.start: float = 0.0
        model.register_forward_pre_hook(self.before)
        model.register_forward_hook(self.after)

    @staticmethod
    def now() -> float:
        """Returns the time once the queued GPU work is done."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def before(self, *args: object) -> None:
        """Starts the clock as the inputs enter the model."""
        self.start = self.now()

    def after(self, *args: object) -> None:
        """Stops the clock when the model's output is ready."""
        self.times.append(self.now() - self.start)


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
    timer = ModelTimer(model)
    output = DecisionEvaluator(model, prompt, config.testing).predict(records)
    metrics: dict[str, float | str] = output.metrics
    # trainer's whole-run times (data loading included): only the model's is kept
    for key in ("test_runtime", "test_samples_per_second", "test_steps_per_second"):
        metrics.pop(key, None)
    # the hardware of the times (a cloud may give another gpu than asked)
    if torch.cuda.is_available():
        metrics["test_gpu"] = torch.cuda.get_device_name()
    # model time after warm-up over its questions
    warmup_questions: int = WARMUP_STEPS * config.testing.per_device_eval_batch_size
    metrics["test_seconds_per_question"] = sum(timer.times[WARMUP_STEPS:]) / (
        len(output.label_ids) - warmup_questions
    )
    report: str = json.dumps(metrics, indent=2)
    output_dir = Path(config.testing.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_metrics.json").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
