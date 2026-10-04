import math

from transformers import TrainingArguments

from src.training.evaluation import SystemOneEvaluator

RECORDS: list[dict] = [
    {
        "state": "s",
        "questions": {
            "q": {"type": "noul", "instructions": "Billing?", "label": True},
            "c": {
                "type": "choice",
                "instructions": "Tone?",
                "criteria": {"calm": None, "angry": None, "sad": None},
                "label": "angry",
            },
        },
    }
] * 3


def evaluator(model, processor, tmp_path) -> SystemOneEvaluator:
    args = TrainingArguments(
        output_dir=str(tmp_path), per_device_eval_batch_size=2, report_to=[],
        use_cpu=True,
    )
    return SystemOneEvaluator(model, processor, args)


def test_predict_returns_metrics(model, processor, tmp_path) -> None:
    output = evaluator(model, processor, tmp_path).predict(RECORDS)
    for name in ("test_loss", "test_accuracy", "test_brier", "test_ece"):
        assert math.isfinite(output.metrics[name])


def test_predict_returns_logits_and_labels(model, processor, tmp_path) -> None:
    output = evaluator(model, processor, tmp_path).predict(RECORDS)
    logits, probabilities = output.predictions
    assert logits.shape[0] == probabilities.shape[0] == 6
    assert output.label_ids.shape[0] == 6


def test_options_are_not_shuffled(model, processor, tmp_path) -> None:
    first = evaluator(model, processor, tmp_path).predict(RECORDS)
    second = evaluator(model, processor, tmp_path).predict(RECORDS)
    assert (first.label_ids == second.label_ids).all()
