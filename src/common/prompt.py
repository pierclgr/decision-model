import string
from typing import Any

from PIL import Image

from src.common.types import Question

# final line of the prompt: System One reads the letter logits right after it
SYSTEM_ONE_INSTRUCTION: str = "Answer with the letter of the correct option only."
# System Two generates its answer in a fixed block, parsed by `SystemTwoModel`
SYSTEM_TWO_INSTRUCTION: str = (
    "Answer exactly in this format:\n<answer>\n[letter of the correct option]"
    "\n</answer>\nFor example, if the correct option is A:\n<answer>\nA\n</answer>"
)


class PromptBuilder:
    """Builds chat messages and maps option letters to token ids.

    Args:
        tokenizer: Tokenizer of the backbone.
        max_options: Number of letters to resolve (A, B, C, ...).
        instruction: Final line of the prompt (`SYSTEM_ONE_INSTRUCTION` or
            `SYSTEM_TWO_INSTRUCTION`).

    Raises:
        ValueError: If a letter is not a single token.
    """

    def __init__(
        self,
        tokenizer: Any,
        max_options: int,
        instruction: str = SYSTEM_ONE_INSTRUCTION,
    ) -> None:
        self.instruction: str = instruction
        self.letters: list[str] = list(string.ascii_uppercase[:max_options])
        self.letter_ids: list[int] = []
        for letter in self.letters:
            ids: list[int] = tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"letter {letter!r} is not a single token")
            self.letter_ids.append(ids[0])

    def build_messages(
        self, state: str, question: Question, images: list[Image.Image]
    ) -> list[dict[str, Any]]:
        """Builds the chat messages: images first, then state and question."""
        descriptions: list[str | None] = question.descriptions or [None] * len(
            question.options
        )
        options: str = "\n".join(
            f"{letter}. {option}" + (f": {description}" if description else "")
            for letter, option, description in zip(
                self.letters, question.options, descriptions
            )
        )
        text: str = (
            f"{state}\n\nQuestion: {question.text}\n\nOptions:\n{options}\n\n"
            f"{self.instruction}"
        )
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image} for image in images
        ]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]
