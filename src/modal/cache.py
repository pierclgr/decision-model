"""Caches the backbone and the dataset of a training config.

Downloads the backbone into the HF Hub cache and the dataset into the
datasets cache, and prepares its splits, so later runs start without
downloads. On Modal the caches are the `models` and `datasets` Volumes.

Usage:
    uv run python -m src.modal.cache <train_config.yml> \\
        [--section.key value]
"""

import sys

from huggingface_hub import snapshot_download

from src.data.hub import HubRecordLoader
from src.training.config import TrainConfig


def main(argv: list[str] | None = None) -> None:
    """Downloads the config's backbone and dataset.

    Args:
        argv: Command-line arguments without the program name (default:
            `sys.argv[1:]`).
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(__doc__)
    # no GPU needed, skips the bf16 hardware check of `TrainingArguments`
    config = TrainConfig.from_yaml(argv[0], [*argv[1:], "--training.use_cpu", "true"])
    snapshot_download(repo_id=config.backbone)
    # loading one split downloads the dataset and prepares all its splits
    HubRecordLoader(config.data.dataset).load(config.data.train)


if __name__ == "__main__":
    main()
