import re
from bisect import bisect_right
from typing import Any

import torch
from torch import nn
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from src.common.prompt import PromptBuilder
from src.constants import SYSTEM_TWO_INSTRUCTION
from src.model.config import DecisionEngineConfig
from src.model.output import DecisionEngineOutput
from src.model.system_one import PreTrainedSystemOneModel


class LetterLogitsRecorder(LogitsProcessor):
    """Keeps the letter logits of each generation step, scores unchanged.

    `output_logits` would keep the full-vocabulary logits of every step on
    the GPU, which thinking (thousands of steps) runs out of memory with.
    Custom processors run after `generate`'s own ones, which greedy decoding
    without a generation config (Qwen) does not add, so these are the raw
    logits.

    Args:
        letter_ids: Token ids of the letters (A, B, ...).
    """

    def __init__(self, letter_ids: list[int]) -> None:
        self.letter_ids: list[int] = letter_ids
        self.steps: list[torch.Tensor] = []

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.steps.append(scores[:, self.letter_ids].float())
        return scores

    def stacked(self) -> torch.Tensor:
        """Letter logits of all steps, shape (batch, steps, letters)."""
        return torch.stack(self.steps, dim=1)


# the System Two answer format (`SYSTEM_TWO_INSTRUCTION`): the option letter
# in an <answer> block
ANSWER: re.Pattern[str] = re.compile(r"<answer>\s*([A-Z])\s*</answer>")
END_THINK: str = "</think>"
END_ANSWER: str = "</answer>"
# last tokens searched for `</answer>`: enough even if it is split into
# single characters (9 tokens)
STOP_WINDOW: int = 16


class AnswerStop(StoppingCriteria):
    """Stops each output once its answer block is closed (`</answer>`).

    Generation goes on only for the questions still writing, instead of
    until the slowest output ends. With thinking, a `</answer>` written
    before `</think>` does not stop it.

    Args:
        tokenizer: Tokenizer of the backbone.
        prompt_length: Length of the (padded) prompts: the prompt's own
            example block is not searched.
        thinking: Whether thinking is on.
    """

    def __init__(self, tokenizer: Any, prompt_length: int, thinking: bool) -> None:
        self.tokenizer: Any = tokenizer
        self.prompt_length: int = prompt_length
        self.thinking: bool = thinking

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any
    ) -> torch.BoolTensor:
        done = torch.zeros(len(input_ids), dtype=torch.bool, device=input_ids.device)
        new_tokens: torch.Tensor = input_ids[:, self.prompt_length:]
        for row, tokens in enumerate(new_tokens.tolist()):
            # only the last tokens: decoding the whole output every step is slow
            if END_ANSWER not in self.tokenizer.decode(tokens[-STOP_WINDOW:]):
                continue
            # rare: decode the whole output once to see if thinking ended
            text: str = self.tokenizer.decode(tokens) if self.thinking else ""
            done[row] = not self.thinking or END_THINK in text[: text.rfind(END_ANSWER)]
        return done


class SystemTwoModel(nn.Module):
    """System Two decision model: the backbone generates its answer.

    Wraps a System One model (trained or built from a backbone) and uses its
    backbone, letter token ids and temperature. The answer is the first
    `<answer>\nX\n</answer>` block of the output (after `</think>` when
    thinking); the option probabilities come from the raw letter logits of
    the step that generated X (kept by `LetterLogitsRecorder`). An output
    without a valid block is an error: uniform probabilities, counted wrong.

    Args:
        model: The System One model to wrap.
        prompt: Prompt builder with the System Two instruction and the
            thinking template variables (its tokenizer decodes the outputs).
        max_new_tokens: Cap on generated tokens, thinking included.
        thinking: Whether thinking is on (the answer then comes after
            `</think>`; without it, the prompt holds an empty thinking block).
    """

    def __init__(
        self,
        model: PreTrainedSystemOneModel,
        prompt: PromptBuilder,
        max_new_tokens: int,
        thinking: bool = False,
    ) -> None:
        super().__init__()
        self.model: PreTrainedSystemOneModel = model
        self.prompt: PromptBuilder = prompt
        self.tokenizer: Any = prompt.processor.tokenizer
        self.max_new_tokens: int = max_new_tokens
        self.thinking: bool = thinking

    @classmethod
    def from_system_one(
        cls,
        model: PreTrainedSystemOneModel,
        processor: Any,
        thinking: str,
        max_new_tokens: int,
    ) -> "SystemTwoModel":
        """Builds System Two and its prompt around a System One model.

        Args:
            model: The System One model to wrap.
            processor: Processor matching the backbone.
            thinking: `off`, or a reasoning effort (e.g. `low`); sent to the
                chat template only if it has levels (Qwen3.8, not Qwen3.5).
            max_new_tokens: Cap on generated tokens, thinking included.

        Returns:
            The System Two model (its prompt in `prompt`).
        """
        on: bool = thinking != "off"
        template_kwargs: dict[str, Any] = {"enable_thinking": on}
        # transformers turns a variable the template does not use into
        # processor kwargs, which replace `processor_kwargs` (padding)
        if on and "reasoning_effort" in (processor.chat_template or ""):
            template_kwargs["reasoning_effort"] = thinking
        prompt = PromptBuilder(
            processor,
            model.config.max_options,
            SYSTEM_TWO_INSTRUCTION,
            template_kwargs,
        )
        return cls(model, prompt, max_new_tokens, on)

    @property
    def config(self) -> DecisionEngineConfig:
        """The wrapped model's config (`Trainer` reads it)."""
        return self.model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_options: int | None = None,
        labels: torch.Tensor | None = None,
        option_mask: torch.Tensor | None = None,
        **backbone_kwargs: Any,
    ) -> DecisionEngineOutput:
        """Generates the answers and scores the options.

        Args: as `PreTrainedSystemOneModel.forward`.

        Returns:
            Loss (if `labels` given), logits and probabilities of the options
            and `errors` (bool, shape (batch,): no valid answer block).
        """
        num_options = num_options or self.config.max_options
        if option_mask is None:
            option_mask = torch.ones(
                len(input_ids), num_options, dtype=torch.bool, device=input_ids.device
            )
        recorder = LetterLogitsRecorder(self.config.option_token_ids)
        sequences: torch.Tensor = self.model.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            logits_processor=LogitsProcessorList([recorder]),
            stopping_criteria=StoppingCriteriaList(
                [AnswerStop(self.tokenizer, input_ids.shape[1], self.thinking)]
            ),
            **backbone_kwargs,
        )
        new_tokens: torch.Tensor = sequences[:, input_ids.shape[1]:]
        logits, errors = self.find_answers(
            new_tokens, recorder.stacked(), self.tokenizer, option_mask, self.thinking
        )
        return DecisionEngineOutput.from_logits(
            logits, self.config.temperature, labels, errors
        )

    @staticmethod
    def find_answers(
        new_tokens: torch.Tensor,
        step_logits: torch.Tensor,
        tokenizer: Any,
        option_mask: torch.Tensor,
        thinking: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finds the answer block of each output and its option logits.

        Each token is decoded on its own and the pieces are joined, so the
        position of the answer letter in the text gives its generation step.

        Args:
            new_tokens: Generated token ids, shape (batch, steps).
            step_logits: Raw letter logits of each step, shape
                (batch, steps, letters), letters in A, B, ... order.
            tokenizer: Tokenizer of the backbone.
            option_mask: Bool mask of the real options, shape
                (batch, num_options).
            thinking: Search after the last `</think>` (none is an error).

        Returns:
            Option logits (letter logits at the step of the answer letter;
            zeros, i.e. uniform, for errors; masked options at the float32
            min) and `errors`, shape (batch,): no block, or a letter beyond
            the question's options.
        """
        num_options: int = option_mask.shape[1]
        logits = torch.zeros(option_mask.shape, device=step_logits.device)
        errors = torch.ones(len(new_tokens), dtype=torch.bool)
        for row, tokens in enumerate(new_tokens.tolist()):
            pieces: list[str] = [
                tokenizer.decode([token], skip_special_tokens=False)
                for token in tokens
            ]
            # character offset where each token starts
            starts: list[int] = [0]
            for piece in pieces[:-1]:
                starts.append(starts[-1] + len(piece))
            text: str = "".join(pieces)
            start: int = 0
            if thinking:
                end_think: int = text.rfind(END_THINK)
                if end_think < 0:
                    continue
                start = end_think + len(END_THINK)
            match: re.Match[str] | None = ANSWER.search(text, start)
            if match is None:
                continue
            option: int = ord(match.group(1)) - ord("A")
            if option >= num_options or not option_mask[row, option]:
                continue
            step: int = bisect_right(starts, match.start(1)) - 1
            logits[row] = step_logits[row, step, :num_options]
            errors[row] = False
        return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min), errors
