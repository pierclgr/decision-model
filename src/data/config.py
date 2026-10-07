from dataclasses import dataclass


@dataclass
class DataSettings:
    """Training data on the HF Hub (kev-vision layout).

    Attributes:
        dataset: HF dataset id.
        train: Training split (slices like `train[:1000]` work).
        validation: Validation split, or None for no validation.
    """

    dataset: str
    train: str
    validation: str | None = None
