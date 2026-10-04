from pathlib import Path

import pytest

import src.modal.app as modal_app


def test_app_defines_the_tasks() -> None:
    assert modal_app.app.name == "decision-engine"
    assert sorted(modal_app.TASKS) == ["cache", "calibrate", "test", "train"]


class FakeVolume:
    """Counts `reload` calls."""

    def __init__(self) -> None:
        self.reloads: int = 0

    def reload(self) -> None:
        self.reloads += 1


def test_app_defines_tensorboard() -> None:
    assert modal_app.tensorboard is not None


def test_runs_reload_middleware_reloads_on_page_load(monkeypatch) -> None:
    volume: FakeVolume = FakeVolume()
    monkeypatch.setitem(modal_app.VOLUMES, "runs", volume)
    middleware = modal_app.RunsReloadMiddleware(lambda environ, start: [b"ok"])
    assert middleware({"PATH_INFO": "/"}, None) == [b"ok"]
    assert middleware({"PATH_INFO": "/data/runs"}, None) == [b"ok"]
    assert volume.reloads == 1


@pytest.mark.parametrize(
    ("text", "flag", "expected"),
    [
        ("modal:\n  gpu: H100\n", "", "H100"),
        ("modal:\n  gpu: H100\n", "A100-80GB", "A100-80GB"),
        ("backbone: x\n", "", "H200"),
        ("modal:\n", "", "H200"),
    ],
)
def test_gpu_from_config(tmp_path: Path, text: str, flag: str, expected: str) -> None:
    config = tmp_path / "config.yml"
    config.write_text(text)
    assert modal_app.config_gpu(str(config), flag) == expected
