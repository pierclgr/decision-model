import json
from collections.abc import Iterator
from os import PathLike
from typing import Any

from transformers import AutoProcessor
from transformers.image_utils import load_image
from transformers.pipelines.base import ChunkPipeline

from src.constants import MAX_LETTER_OPTIONS
from src.model import PreTrainedSystemOneModel
from src.prompt import PromptBuilder
from src.types import Question


class SystemOnePipeline(ChunkPipeline):
    """System One pipeline with the Jev `/v1/systemone` request and response.

    Each question of a request is one chunk: one forward pass, no generation.
    See `docs/MODEL_SPEC.md` for the request and response format.

    Args:
        model: The System One model.
        processor: Processor matching the backbone.
        **kwargs: Passed to `Pipeline` (e.g. `device`).

    Raises:
        ValueError: If the processor's letter token ids differ from the ones
            saved in the model config.
    """

    def __init__(self, model: PreTrainedSystemOneModel, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        n: int = model.config.max_options
        self.prompt: PromptBuilder = PromptBuilder(self.tokenizer, n)
        if self.prompt.letter_ids != (model.config.option_token_ids or [])[:n]:
            raise ValueError("processor and model use different letter token ids")

    @classmethod
    def from_backbone(
        cls, backbone_id: str, temperature: float = 1.0, **kwargs: Any
    ) -> "SystemOnePipeline":
        """Loads a pretrained backbone and its processor.

        Args:
            backbone_id: HF id or local path of the multimodal backbone.
            temperature: Divides the option logits before softmax.
            **kwargs: Passed to the backbone `from_pretrained` (e.g. `dtype`,
                `device_map`).

        Returns:
            The pipeline.
        """
        processor = AutoProcessor.from_pretrained(backbone_id)
        # left padding keeps the answer slot at the last position in batches
        processor.tokenizer.padding_side = "left"
        letter_ids: list[int] = PromptBuilder(
            processor.tokenizer, MAX_LETTER_OPTIONS
        ).letter_ids
        model = PreTrainedSystemOneModel.from_backbone(
            backbone_id, letter_ids, temperature, **kwargs
        )
        return cls(model=model, processor=processor)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "SystemOnePipeline":
        """Loads a pipeline saved with `save_pretrained`."""
        processor = AutoProcessor.from_pretrained(path)
        processor.tokenizer.padding_side = "left"
        model = PreTrainedSystemOneModel.from_pretrained(path, **kwargs)
        return cls(model=model, processor=processor)

    def save_pretrained(self, save_directory: str | PathLike, **kwargs: Any) -> None:
        """Saves the model and the processor to `save_directory`."""
        super().save_pretrained(save_directory, **kwargs)
        self.processor.save_pretrained(save_directory)

    def _sanitize_parameters(self, **kwargs: Any) -> tuple[dict, dict, dict]:
        return {}, {}, {}

    def preprocess(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yields the model inputs of each question of the request."""
        state: Any = request["state"]
        state_text: str = (
            state
            if isinstance(state, str)
            else json.dumps(state, indent=2, ensure_ascii=False)
        )
        images = [self._load_image(item) for item in request.get("media", [])]
        questions: dict[str, dict[str, Any]] = request["questions"]
        for i, (question_id, spec) in enumerate(questions.items()):
            question = self._build_question(spec)
            messages = self.prompt.build_messages(state_text, question, images)
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            yield {
                "is_last": i == len(questions) - 1,
                "id": question_id,
                "type": spec["type"],
                "question": question,
                "model_inputs": inputs,
            }

    def _forward(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        inputs = model_inputs["model_inputs"]
        question: Question = model_inputs["question"]
        out = self.model(**inputs, num_options=len(question.options))
        meta = {k: v for k, v in model_inputs.items() if k != "model_inputs"}
        return {
            **meta,
            "probabilities": out.probabilities[0],
            "input_tokens": inputs["input_ids"].shape[-1],
        }

    def postprocess(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Builds the `/v1/systemone` response from the chunks."""
        answers = {
            chunk["id"]: self._format_answer(
                chunk["type"], chunk["question"], chunk["probabilities"].tolist()
            )
            for chunk in chunks
        }
        return {
            "model": self.model.config.backbone_config.name_or_path,
            "answers": answers,
            "usage": {
                "input_tokens": sum(chunk["input_tokens"] for chunk in chunks),
                "output_tokens": 0,
            },
        }

    @staticmethod
    def _load_image(item: dict[str, Any]) -> Any:
        if item["type"] != "image":
            raise ValueError(f"unsupported media type {item['type']!r}")
        return load_image(item["data"])

    @staticmethod
    def _build_question(spec: dict[str, Any]) -> Question:
        kind: str = spec["type"]
        text: str = spec["instructions"]
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

    @staticmethod
    def _format_answer(
        kind: str, question: Question, probabilities: list[float]
    ) -> dict[str, Any]:
        if kind == "noul":
            return {"type": "noul", "noul": probabilities[0]}
        k: int = len(probabilities)
        best: int = probabilities.index(max(probabilities))
        if kind == "choice":
            return {
                "type": "choice",
                "choice": question.options[best],
                "confidence": (probabilities[best] - 1 / k) / (1 - 1 / k),
                "probabilities": dict(zip(question.options, probabilities)),
            }
        # score: spread around the mode, relative to the uniform distribution
        spread: float = sum(p * abs(i - best) for i, p in enumerate(probabilities))
        uniform: float = sum(abs(i - best) for i in range(k)) / k
        return {
            "type": "score",
            "score": sum(i * p for i, p in enumerate(probabilities)),
            "confidence": max(0.0, 1 - spread / uniform),
            "legend": {str(i): level for i, level in enumerate(question.options)},
            "probabilities": {str(i): p for i, p in enumerate(probabilities)},
        }
