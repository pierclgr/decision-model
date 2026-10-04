"""Modal app: caching, training, calibration and testing on cloud GPUs.

Also serves TensorBoard on the training logs.

Persistent Volumes (created on first use), so big data is downloaded once:
- `models`: HF Hub cache (backbones), mounted at `/root/models`
- `datasets`: HF datasets cache (raw files and prepared splits), mounted at
  `/root/datasets`
- `runs`: run outputs (checkpoints, `final/`, test metrics), mounted at
  `/root/runs`, so the configs' `runs/...` paths land on it

Needs `modal setup` and a Secret `huggingface` with `HF_TOKEN` for the gated
dataset (`modal secret create huggingface HF_TOKEN=<token>`). Run from the
repository root:

    modal run src/modal/app.py --task cache \\
        --config configs/train/qwen3_8_27b.yml
    modal run --detach src/modal/app.py --task train \\
        --config configs/train/qwen3_8_27b.yml
    modal run src/modal/app.py --task calibrate \\
        --config configs/train/qwen3_8_27b.yml \\
        --model-dir runs/qwen3_8_27b/final
    modal run src/modal/app.py --task test --config configs/test/qwen3_8_27b.yml

GPU: the config's `modal.gpu` (default H200), e.g.

    modal:
      gpu: H100

Options: `--overrides "--training.learning_rate 5e-5"` (config overrides),
`--gpu H100` (instead of the config's GPU). `--detach` keeps the run going
if the local client disconnects (Ctrl+C stops only the log streaming).

TensorBoard on the `runs` Volume (needs `report_to: tensorboard` in the
training config), at the URL printed by:

    modal deploy src/modal/app.py
"""

import shlex
from collections.abc import Callable, Iterable
from pathlib import Path

import modal
import yaml

ROOT: str = "/root"
# modal's maximum function timeout
DAY: int = 24 * 60 * 60
# gpu of the tasks when the config's `modal` section names none
GPU: str = "H200"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync()
    .env({"HF_HUB_CACHE": f"{ROOT}/models", "HF_DATASETS_CACHE": f"{ROOT}/datasets"})
    .workdir(ROOT)
    .add_local_python_source("src")
    .add_local_dir("configs", remote_path=f"{ROOT}/configs")
)

VOLUMES: dict[str, modal.Volume] = {
    name: modal.Volume.from_name(name, create_if_missing=True)
    for name in ("models", "datasets", "runs")
}

app = modal.App(
    "decision-engine",
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={f"{ROOT}/{name}": volume for name, volume in VOLUMES.items()},
)


@app.function(cpu=4, memory=16384, timeout=DAY)
def cache(argv: list[str]) -> None:
    """Runs `src.modal.cache` with `argv` (no GPU)."""
    from src.modal.cache import main

    main(argv)


@app.function(gpu=GPU, timeout=DAY)
def train(argv: list[str]) -> None:
    """Runs `src.training.train` with `argv`."""
    from src.training.train import main

    main(argv)


@app.function(gpu=GPU, timeout=DAY)
def calibrate(argv: list[str]) -> None:
    """Runs `src.training.calibration` with `argv`."""
    from src.training.calibration import main

    main(argv)


@app.function(gpu=GPU, timeout=DAY)
def test(argv: list[str]) -> None:
    """Runs `src.testing.test` with `argv`."""
    from src.testing.test import main

    main(argv)


class RunsReloadMiddleware:
    """WSGI middleware: reloads the `runs` Volume on each page load.

    A running container sees only the Volume state from its start, so
    TensorBoard shows new training logs only after a reload.
    """

    def __init__(self, app: Callable) -> None:
        """Wraps a WSGI app.

        Args:
            app: The TensorBoard WSGI app.
        """
        self.app: Callable = app

    def __call__(self, environ: dict, start_response: Callable) -> Iterable[bytes]:
        """Reloads the Volume on `/`, then runs the wrapped app."""
        if environ.get("PATH_INFO") == "/":
            try:
                VOLUMES["runs"].reload()
            # reload fails while TensorBoard has files open, keep the old state
            except Exception as error:
                print(f"runs Volume reload failed: {error}")
        return self.app(environ, start_response)


@app.function(max_containers=1, scaledown_window=5 * 60)
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def tensorboard() -> Callable:
    """Serves TensorBoard on the `runs` Volume (training logs)."""
    from tensorboard import program
    from tensorboard.backend import application

    board = program.TensorBoard()
    board.configure(logdir=f"{ROOT}/runs")
    data_provider, multiplexer = board._make_data_provider()
    return application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        multiplexer,
        experimental_middlewares=[RunsReloadMiddleware],
    )._create_wsgi_app()


def config_gpu(config: str, gpu: str) -> str:
    """Returns the GPU of a run.

    Args:
        config: Config path, with an optional `modal` section.
        gpu: GPU from the command line, empty if not given.

    Returns:
        `gpu` if given, else the config's `modal.gpu`, else `GPU`.
    """
    values: dict = yaml.safe_load(Path(config).read_text()) or {}
    return gpu or (values.get("modal") or {}).get("gpu") or GPU


TASKS: dict[str, modal.Function] = {
    "cache": cache,
    "train": train,
    "calibrate": calibrate,
    "test": test,
}


@app.local_entrypoint()
def main(
    task: str, config: str, model_dir: str = "", overrides: str = "", gpu: str = ""
) -> None:
    """Runs a task on Modal.

    Args:
        task: `cache`, `train`, `calibrate` or `test`.
        config: Config path, relative to the repository root (a training
            config for `cache`, `train` and `calibrate`, a test config for
            `test`).
        model_dir: Model to calibrate, e.g. `runs/<run>/final` (`calibrate`
            only).
        overrides: Config overrides, e.g. `"--lora.r 8 --checkpoint x"`.
        gpu: GPU instead of the config's `modal.gpu` (e.g. `H100`,
            `A100-80GB`). Ignored by `cache` (CPU only).

    Raises:
        ValueError: If the task is unknown, or `calibrate` has no model dir.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {sorted(TASKS)}")
    if task == "calibrate" and not model_dir:
        raise ValueError("calibrate needs --model-dir")
    argv: list[str] = [config, *([model_dir] if task == "calibrate" else [])]
    function = (
        TASKS[task]
        if task == "cache"
        else TASKS[task].with_options(gpu=config_gpu(config, gpu))
    )
    # a spawned call survives ctrl+c with --detach, a `.remote()` one is cancelled
    function.spawn([*argv, *shlex.split(overrides)]).get()
