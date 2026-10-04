from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from src.common.request import RequestParser
from src.common.types import Question
from src.constants import MAX_LETTER_OPTIONS


class SystemOneDataset(Dataset):
    """Training items from labelled `/v1/systemone` records.

    A record is a request whose questions also hold a `label` (choice: option
    name, noul: true/false, score: level index) and optionally a soft `target`
    (weights keyed like the label, e.g. vote counts; normalized here). Each
    question of each record is one item. Questions with more options than
    `max_options` are skipped and counted in `skipped`.

    Args:
        records: The labelled records.
        shuffle_options: Shuffle the options of `choice` questions at each
            access (the target follows its option).
        max_options: Max options per question (one letter each).
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        shuffle_options: bool = True,
        max_options: int = MAX_LETTER_OPTIONS,
    ) -> None:
        self.records: Sequence[dict[str, Any]] = records
        self.shuffle_options: bool = shuffle_options
        self.index: list[tuple[int, str]] = []
        self.skipped: int = 0
        for i, record in enumerate(records):
            for question_id, spec in record["questions"].items():
                num_options: int = (
                    2 if spec["type"] == "noul" else len(spec["criteria"])
                )
                if num_options > max_options:
                    self.skipped += 1
                else:
                    self.index.append((i, question_id))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        """Returns one question as an item.

        Returns:
            `type`, `state` (text), `images`, `question` (`Question`) and
            `target` (one probability per option).

        Raises:
            ValueError: If the label or a target key is unknown, or the target
                sums to 0.
        """
        record_index, question_id = self.index[i]
        record: dict[str, Any] = self.records[record_index]
        spec: dict[str, Any] = record["questions"][question_id]
        question: Question = RequestParser.build_question(spec)
        target: list[float] = self._target(spec, question)
        if self.shuffle_options and spec["type"] == "choice":
            question, target = self._shuffle(question, target)
        return {
            "type": spec["type"],
            "state": RequestParser.render_text(record["state"]),
            "images": RequestParser.load_media(record.get("media", [])),
            "question": question,
            "target": target,
        }

    @staticmethod
    def _target(spec: dict[str, Any], question: Question) -> list[float]:
        kind: str = spec["type"]
        weights: list[float] = [0.0] * len(question.options)
        if "target" in spec:
            for key, weight in spec["target"].items():
                index: int = SystemOneDataset._option_index(kind, key, question)
                weights[index] = float(weight)
        else:
            index = SystemOneDataset._option_index(kind, spec["label"], question)
            weights[index] = 1.0
        total: float = sum(weights)
        if total <= 0:
            raise ValueError("target sums to 0")
        return [weight / total for weight in weights]

    @staticmethod
    def _option_index(kind: str, key: Any, question: Question) -> int:
        """Returns the option index of a label or target key."""
        if kind == "choice":
            if key not in question.options:
                raise ValueError(f"unknown option {key!r}")
            return question.options.index(key)
        if kind == "noul":
            if key not in (True, False, "true", "false"):
                raise ValueError(f"unknown noul label {key!r}")
            return 0 if key in (True, "true") else 1
        index: int = int(key)
        if not 0 <= index < len(question.options):
            raise ValueError(f"level {index} out of range")
        return index

    @staticmethod
    def _shuffle(
        question: Question, target: list[float]
    ) -> tuple[Question, list[float]]:
        order: list[int] = torch.randperm(len(question.options)).tolist()
        descriptions = question.descriptions or [None] * len(question.options)
        shuffled = Question(
            question.text,
            [question.options[j] for j in order],
            [descriptions[j] for j in order],
        )
        return shuffled, [target[j] for j in order]
