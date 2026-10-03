# AGENTS.md

## Project
Decoder-based System One decision model. A pretrained multimodal HF LLM
answers typed questions in one forward pass (no generation). Probabilities are
read from the logits of the option letters. See `docs/MODEL_SPEC.md`.

## Layout
- `src/`: source code (package `src`, imports as `from src.x import ...`)
  - `constants.py`: shared constants
  - `config.py`: `DecisionModelConfig` (`PreTrainedConfig`)
  - `types.py`: `Question`
  - `prompt.py`: `PromptBuilder` (chat messages, letter token ids)
  - `model.py`: `PreTrainedSystemOneModel` (`PreTrainedModel` wrapping the
    backbone, option logits and probabilities)
  - `pipeline.py`: `SystemOnePipeline` (HF `ChunkPipeline`, Jev
    `/v1/systemone` request and response)
- `tests/`: unit tests (tiny random backbone, no downloads)
- `docs/`: documentation

## Commands
- `uv sync`: install the environment
- `uv run pytest`: run the tests

## Conventions
- Python: PEP8, Google docstrings, type hints everywhere (PEP 484)
- Comments: start lowercase, no ending period
- File names: `snake_case.py`; Markdown: `UPPER_SNAKE_CASE.md`
- Markdown: max 80 chars per line
- Structure: `src/` code, `tests/` tests, `docs/` documentation
- Git: branches `feature/snake_case`, `fix/snake_case`; commit messages in
  past tense, one short sentence, name the files affected
