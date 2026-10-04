import numpy as np
from transformers import EvalPrediction


class DecisionMetrics:
    """`Trainer` `compute_metrics` for decision models.

    Computed on the option probabilities (with the model temperature):
    - `accuracy`: the most likely option is the most likely label option.
    - `brier`: mean over questions of the sum over options of
      `(p_k - y_k)^2` (lower is better).
    - `ece`: expected calibration error over `NUM_BINS` equal-width
      confidence bins (confidence = max probability; 0 is perfect).
    """

    NUM_BINS: int = 10

    def __call__(self, prediction: EvalPrediction) -> dict[str, float]:
        """Returns the metrics of an evaluation run."""
        # predictions are (logits, probabilities); Trainer pads with -100 the
        # batches with fewer options
        probabilities: np.ndarray = np.clip(prediction.predictions[1], 0.0, None)
        labels: np.ndarray = np.clip(prediction.label_ids, 0.0, None)
        confidence: np.ndarray = probabilities.max(-1)
        correct: np.ndarray = probabilities.argmax(-1) == labels.argmax(-1)
        bins: np.ndarray = np.minimum(
            (confidence * self.NUM_BINS).astype(int), self.NUM_BINS - 1
        )
        ece: float = 0.0
        for b in np.unique(bins):
            in_bin: np.ndarray = bins == b
            gap: float = abs(correct[in_bin].mean() - confidence[in_bin].mean())
            ece += in_bin.mean() * gap
        return {
            "accuracy": float(correct.mean()),
            "brier": float(((probabilities - labels) ** 2).sum(-1).mean()),
            "ece": float(ece),
        }
