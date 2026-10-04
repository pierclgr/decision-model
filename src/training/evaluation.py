from dataclasses import replace
from typing import Any

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PredictionOutput

from src.common.prompt import PromptBuilder
from src.data.collator import SystemOneCollator
from src.data.dataset import SystemOneDataset
from src.model.system_one import PreTrainedSystemOneModel
from src.model.system_two import SystemTwoModel
from src.training.metrics import DecisionMetrics


class SystemOneEvaluator:
    """Runs a decision model on labelled records, without training.

    Used to test a model (System One or Two) and to fit its temperature.

    Args:
        model: The System One model, or a `SystemTwoModel`.
        processor: Processor matching the backbone.
        training_args: `Trainer` arguments (device, eval batch size,
            precision). A copy is used, with `remove_unused_columns` False
            (the collator needs the raw items) and `eval_strategy` "no" (no
            eval dataset; the training's args may have evaluation on).
        collator: Batch collator (default: the System One prompt, thinking
            off).
    """

    def __init__(
        self,
        model: PreTrainedSystemOneModel | SystemTwoModel,
        processor: Any,
        training_args: TrainingArguments,
        collator: SystemOneCollator | None = None,
    ) -> None:
        self.max_options: int = model.config.max_options
        self.trainer: Trainer = Trainer(
            model=model,
            args=replace(
                training_args, remove_unused_columns=False, eval_strategy="no"
            ),
            data_collator=collator
            or SystemOneCollator(
                processor, PromptBuilder(processor.tokenizer, self.max_options)
            ),
            compute_metrics=DecisionMetrics(),
        )

    def predict(self, records: Any) -> PredictionOutput:
        """Runs the model on the records (options in their original order).

        Returns:
            `predictions` (option logits, probabilities), `label_ids` and
            `metrics` (`test_loss`, `test_accuracy`, `test_brier`,
            `test_ece`).
        """
        dataset = SystemOneDataset(
            records, shuffle_options=False, max_options=self.max_options
        )
        return self.trainer.predict(dataset)
