"""Modal app serving TensorBoard on the training logs of the `runs` Volume.

Separate from `src/modal/app.py`: a web function has a fixed URL, so it
would stop runs of that app from running in parallel. Needs
`report_to: tensorboard` in the training config. Deploy from the
repository root (the URL is printed):

    modal deploy src/modal/tensorboard_app.py
"""

from collections.abc import Callable, Iterable

import modal

from src.modal.app import ROOT, VOLUMES, image

app = modal.App("decision-engine-tensorboard", image=image)


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


@app.function(
    max_containers=1,
    scaledown_window=5 * 60,
    volumes={f"{ROOT}/runs": VOLUMES["runs"]},
)
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
