# AGENTS.md

## Project
Decoder-based System One decision model. A pretrained multimodal HF LLM
answers typed questions in one forward pass (no generation). Probabilities are
read from the logits of the option letters. See `docs/MODEL_SPEC.md`.

## Layout
- `src/`: source code (package `src`, imports as `from src.x import ...`)
  - `constants.py`: shared constants
  - `model/`: the model
    - `config.py`: `DecisionModelConfig` (`PreTrainedConfig`)
    - `system_one.py`: `PreTrainedSystemOneModel` (`PreTrainedModel`
      wrapping the backbone, option logits, probabilities and loss)
    - `system_two.py`: `SystemTwoModel` (wraps a System One model, the
      backbone generates its answer; System Two test mode)
  - `common/`: shared code
    - `types.py`: `Question`
    - `request.py`: `RequestParser` (request parsing)
    - `prompt.py`: `PromptBuilder` (chat messages, letter token ids)
    - `config_parser.py`: `ConfigParser` (YAML configs with CLI overrides,
      used by `TrainConfig`, `CalibrationConfig` and `TestConfig`)
  - `data/`: training data
    - `dataset.py`: `SystemOneDataset` (labelled records, one item per
      question, choice option shuffle)
    - `collator.py`: `SystemOneCollator` (batch with `labels`,
      `option_mask`)
    - `hub.py`: `HubRecordLoader` (HF Hub dataset in the kev-vision layout
      to records)
  - `pipeline/`: inference
    - `system_one.py`: `SystemOnePipeline` (HF `ChunkPipeline`, Jev
      `/v1/systemone` request and response)
  - `training/`: LoRA training and calibration
    - `config.py`: `TrainConfig` (YAML run config: `backbone`, `seed`,
      `lora`, `data`, `training`, optional `calibration`) and
      `CalibrationConfig` (standalone calibration: `model`, `seed`, `data`,
      `calibration`, `calibrating`)
    - `train.py`: training script (`Trainer`, LoRA on the language model;
      calibrates at the end if configured)
    - `metrics.py`: `DecisionMetrics` (accuracy, Brier score, ECE)
    - `evaluation.py`: `SystemOneEvaluator` (runs a model on records)
    - `calibration.py`: `TemperatureCalibrator` (global temperature) and
      the standalone calibration script
  - `testing/`: standalone testing of a trained model
    - `config.py`: `TestConfig` (YAML: `model`, optional `temperature`,
      `data`, `testing`)
    - `test.py`: test script (metrics and seconds per question on a
      dataset split)
  - `modal/`: Modal cloud runs
    - `app.py`: Modal app with `cache`, `train`, `calibrate`, `test`
      (GPUs, Volumes `models`, `datasets`, `runs` as persistent caches)
    - `tensorboard_app.py`: separate Modal app serving TensorBoard (web
      app on the `runs` Volume), so runs of `app.py` can go in parallel
    - `cache.py`: downloads a config's backbone and dataset (cache)
- `configs/train/`: training run configs (YAML)
- `configs/calibration/`: standalone calibration configs (YAML)
- `configs/test/`: test run configs (YAML)
- `tests/`: unit tests (tiny random backbone, no downloads)
- `docs/`: documentation

## Commands
- `uv sync`: install the environment (includes `flash-linear-attention`,
  fast Qwen3.5/3.8 linear attention on CUDA)
- `uv run pytest`: run the tests
- `uv run python -m src.training.train configs/train/<run>.yml
  [--section.key value]`: train (the gated dataset needs `hf auth login`)
- `uv run python -m src.training.calibration configs/calibration/<run>.yml
  [--section.key value]`: calibrate a saved model
- `uv run python -m src.testing.test configs/test/<run>.yml
  [--section.key value]`: test a trained model (or a raw HF backbone in
  `model`, zero-shot; a `system_two` section for System Two), prints the
  metrics
- `uv run python -m src.modal.cache configs/train/<run>.yml`: download
  the backbone and dataset of a config
- `modal run [--detach] src/modal/app.py --task cache|train|calibrate|test
  --config <config.yml> [--overrides "..."] [--gpu X]`:
  same tasks on Modal (needs `modal setup` and Secret `huggingface` with
  `HF_TOKEN`); GPU from the config's `modal.gpu` (default H200), `--gpu`
  wins
- `modal deploy src/modal/tensorboard_app.py`: serve TensorBoard on the
  training logs (URL printed)

## Conventions
- Python: PEP8, Google docstrings, type hints everywhere (PEP 484)
- Comments: start lowercase, no ending period
- File names: `snake_case.py`; Markdown: `UPPER_SNAKE_CASE.md`
- Markdown: max 80 chars per line
- Structure: `src/` code, `tests/` tests, `docs/` documentation
- Git: branches `feature/snake_case`, `fix/snake_case`; commit messages in
  past tense, one short sentence, name the files affected
