import pytest
from PIL import Image

from src.prompt import PromptBuilder
from src.types import Question

from conftest import FakeTokenizer


class MultiTokenTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1, 2]


def test_letter_ids() -> None:
    builder = PromptBuilder(FakeTokenizer(), max_options=26)
    assert builder.letter_ids[:3] == [ord("A"), ord("B"), ord("C")]
    assert len(builder.letter_ids) == 26


def test_multi_token_letter_raises() -> None:
    with pytest.raises(ValueError):
        PromptBuilder(MultiTokenTokenizer(), max_options=26)


def test_build_messages_text_only() -> None:
    builder = PromptBuilder(FakeTokenizer(), max_options=26)
    question = Question(text="Is it safe?", options=["yes", "no"])
    messages = builder.build_messages("some state", question, [])
    content = messages[0]["content"]
    assert messages[0]["role"] == "user"
    assert [part["type"] for part in content] == ["text"]
    text = content[0]["text"]
    assert "some state" in text
    assert "Is it safe?" in text
    assert "A. yes" in text and "B. no" in text


def test_build_messages_with_descriptions() -> None:
    builder = PromptBuilder(FakeTokenizer(), max_options=26)
    question = Question(
        text="q", options=["x", "y"], descriptions=["first one", None]
    )
    text = builder.build_messages("s", question, [])[0]["content"][0]["text"]
    assert "A. x: first one" in text
    assert "B. y\n" in text


def test_build_messages_images_first() -> None:
    builder = PromptBuilder(FakeTokenizer(), max_options=26)
    question = Question(text="q", options=["x", "y"])
    image = Image.new("RGB", (4, 4))
    content = builder.build_messages("s", question, [image, image])[0]["content"]
    assert [part["type"] for part in content] == ["image", "image", "text"]
    assert content[0]["image"] is image
