"""LoRA training of the System One model with HF `Trainer`.

Example:
    uv run python -m src.training.train configs/train/qwen3_5_0_8b.yml \\
        --training.learning_rate 1e-4 --data.train "train[:1000]"
"""

import sys
from pathlib import Path
from typing import Any

from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import Trainer, TrainingArguments, set_seed

from src.common.prompt import PromptBuilder
from src.data.collator import SystemOneCollator
from src.data.dataset import SystemOneDataset
from src.data.hub import HubRecordLoader
from src.model.system_one import PreTrainedSystemOneModel
from src.pipeline.system_one import SystemOnePipeline
from src.training.calibration import TemperatureCalibrator
from src.training.config import LoraSettings, TrainConfig
from src.training.metrics import DecisionMetrics


def lora_target_modules(model: nn.Module) -> list[str]:
    """Returns the linear layers of the language model (LoRA targets).

    The vision tower, the projector and `lm_head` are left out, so they stay
    frozen.

    Raises:
        ValueError: If the model has no `language_model` linear layers.
    """
    names: list[str] = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and ".language_model." in name
    ]
    if not names:
        raise ValueError("no language_model linear layers found")
    return names


def build_trainer(
    model: PreTrainedSystemOneModel,
    processor: Any,
    train_records: Any,
    eval_records: Any | None,
    lora: LoraSettings,
    training_args: TrainingArguments,
) -> Trainer:
    """Adds LoRA to the model and builds the `Trainer`.

    Args:
        model: The System One model (LoRA layers are added in place).
        processor: Processor matching the backbone.
        train_records: Labelled training records.
        eval_records: Labelled evaluation records, or None.
        lora: LoRA settings.
        training_args: `Trainer` arguments. `remove_unused_columns` is set to
            False, since the collator needs the raw items.

    Returns:
        The trainer.
    """
    lora_config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        target_modules=lora_target_modules(model),
    )
    max_options: int = model.config.max_options
    train_dataset = SystemOneDataset(train_records, max_options=max_options)
    eval_dataset = (
        SystemOneDataset(eval_records, shuffle_options=False, max_options=max_options)
        if eval_records is not None
        else None
    )
    print(f"train questions: {len(train_dataset)}, skipped: {train_dataset.skipped}")
    training_args.remove_unused_columns = False
    # the lora weights are created before `Trainer` sets the seed
    set_seed(training_args.seed)
    return Trainer(
        model=get_peft_model(model, lora_config),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SystemOneCollator(
            processor, PromptBuilder(processor.tokenizer, max_options)
        ),
        compute_metrics=DecisionMetrics(),
    )


def main(argv: list[str] | None = None) -> None:
    """Trains a LoRA and saves the merged model to `<output_dir>/final`.

    If the config has a `calibration` section, the merged model is calibrated
    before it is saved. To fit the temperature, `calibration.split` of the
    training records is carved out before training (seeded with
    `seed`) and used only for the fit.

    Usage: `python -m src.training.train <config.yml> [--section.key value]`.

    Args:
        argv: Command-line arguments without the program name (default:
            `sys.argv[1:]`).
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(__doc__)
    config = TrainConfig.from_yaml(argv[0], argv[1:])
    pipeline = SystemOnePipeline.from_backbone(
        config.backbone, dtype=PreTrainedSystemOneModel.default_dtype()
    )
    loader = HubRecordLoader(config.data.dataset)
    train_records = loader.load(config.data.train)
    calibration_records = None
    if config.calibration is not None and config.calibration.temperature == "fit":
        train_records, calibration_records = TemperatureCalibrator.split(
            train_records, config.calibration.split, config.seed
        )
    eval_records = (
        loader.load(config.data.validation) if config.data.validation else None
    )
    trainer = build_trainer(
        pipeline.model,
        pipeline.processor,
        train_records,
        eval_records,
        config.lora,
        config.training,
    )
    trainer.train(resume_from_checkpoint=config.checkpoint)
    merged: PreTrainedSystemOneModel = trainer.model.merge_and_unload()
    if config.calibration is not None:
        calibrator = TemperatureCalibrator(merged, pipeline.processor, config.training)
        temperature: float = calibrator.calibrate(
            config.calibration, calibration_records
        )
        print(f"temperature: {temperature:.4f}")
    final: Path = Path(config.training.output_dir) / "final"
    merged.save_pretrained(final)
    pipeline.processor.save_pretrained(final)


if __name__ == "__main__":
    main()
