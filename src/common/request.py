import json
from typing import Any

from PIL import Image
from transformers.image_utils import load_image

from src.common.types import Question


class RequestParser:
    """Parses `/v1/systemone` requests, shared by inference and training.

    The pipeline and the dataset both use it, so training and inference build
    the same prompt.
    """

    @staticmethod
    def render_text(value: Any) -> str:
        """Returns a state or instructions as text.

        A string is returned as is, anything else as indented JSON.
        """
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, ensure_ascii=False)

    @staticmethod
    def load_media(media: list[dict[str, Any]]) -> list[Image.Image]:
        """Loads the media items of a request.

        Args:
            media: Items `{"type": "image", "data": ...}`, where `data` is a
                data URI, a URL, a path or a PIL image.

        Returns:
            The images, in request order.

        Raises:
            ValueError: If an item is not an image.
        """
        images: list[Image.Image] = []
        for item in media:
            if item["type"] != "image":
                raise ValueError(f"unsupported media type {item['type']!r}")
            images.append(load_image(item["data"]))
        return images

    @staticmethod
    def build_question(spec: dict[str, Any]) -> Question:
        """Turns a typed question spec into a lettered `Question`.

        `choice` options are the criteria keys, `noul` options are "Yes" and
        "No", `score` options are the levels.

        Raises:
            ValueError: If the question type is unknown.
        """
        kind: str = spec["type"]
        text: str = RequestParser.render_text(spec["instructions"])
        criteria: Any = spec.get("criteria")
        if kind == "choice":
            return Question(text, list(criteria), list(criteria.values()))
        if kind == "noul":
            criteria = criteria or {}
            return Question(
                text, ["Yes", "No"], [criteria.get("true"), criteria.get("false")]
            )
        if kind == "score":
            return Question(text, list(criteria))
        raise ValueError(f"unknown question type {kind!r}")
