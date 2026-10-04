from typing import Any

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PredictionOutput

from src.common.prompt import PromptBuilder
from src.data.collator import SystemOneCollator
from src.data.dataset import SystemOneDataset
from src.model.system_one import PreTrainedSystemOneModel
from src.training.metrics import DecisionMetrics


class SystemOneEvaluator:
    """Runs a System One model on labelled records, without training.

    Used to test a model and to fit its temperature.

    Args:
        model: The System One model.
        processor: Processor matching the backbone.
        training_args: `Trainer` arguments (device, eval batch size,
            precision). `remove_unused_columns` is set to False, since the
            collator needs the raw items.
    """

    def __init__(
        self,
        model: PreTrainedSystemOneModel,
        processor: Any,
        training_args: TrainingArguments,
    ) -> None:
        self.max_options: int = model.config.max_options
        training_args.remove_unused_columns = False
        self.trainer: Trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=SystemOneCollator(
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
