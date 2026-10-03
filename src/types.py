from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    """A lettered choice question (the internal form of every question type).

    Attributes:
        text: The question instructions.
        options: The answer options (at least 2).
        descriptions: Optional description of each option, aligned with
            `options` (entries can be None).
    """

    text: str
    options: list[str]
    descriptions: list[str | None] | None = None

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError("a question needs at least 2 options")
        if self.descriptions is not None and len(self.descriptions) != len(
            self.options
        ):
            raise ValueError("descriptions must match options")
