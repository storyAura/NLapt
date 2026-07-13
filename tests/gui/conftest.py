"""Shared GUI test fixtures (owned by the foundation agent).

Sets the offscreen Qt platform BEFORE any Qt import and isolates the
per-user data directory so tests never touch real settings. Panel agents
reuse ``demo_dataset`` and ``controller``.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path  # noqa: E402
from typing import Iterator  # noqa: E402

import pytest  # noqa: E402

# Captions used by the demo dataset (keys are relative POSIX paths).
DEMO_CAPTIONS: dict[str, str] = {
    "0001.png": "1girl, solo, long hair, 少女站在樱花树下。masterpiece",
    "0002.png": "1girl, school uniform, short hair, best quality",
    "10_concept/0003.png": "1girl, yukata, fireworks",
    # 10_concept/0004.png intentionally has NO txt file (unlabeled).
}
DEMO_IMAGE_SIZE = (4, 3)
DEMO_KEYS: tuple[str, ...] = (
    "0001.png",
    "0002.png",
    "10_concept/0003.png",
    "10_concept/0004.png",
)


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point app_data_dir() at a per-test directory (never the real one)."""
    monkeypatch.setenv("NLAPT_DATA_DIR", str(tmp_path / "appdata"))


@pytest.fixture(autouse=True)
def _animations_disabled() -> Iterator[None]:
    """Disable UI animations for every test (anim.py's documented contract).

    Open/close fades and other transitions collapse to synchronous end-states
    so tests never wait on wall-clock animations; tests that exercise the
    animation helpers re-enable them explicitly.
    """
    from nlapt_gui import anim

    previous = anim.animations_enabled()
    anim.set_animations_enabled(False)
    yield
    anim.set_animations_enabled(previous)


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", DEMO_IMAGE_SIZE, (128, 64, 32)).save(path, format="PNG")


@pytest.fixture()
def demo_dataset(tmp_path: Path) -> Path:
    """Small real dataset: root PNGs + kohya-style subfolder + one missing txt."""
    root = tmp_path / "dataset"
    for rel, caption in DEMO_CAPTIONS.items():
        image = root / rel
        _write_png(image)
        image.with_suffix(".txt").write_text(caption, encoding="utf-8", newline="\n")
    _write_png(root / "10_concept" / "0004.png")
    return root


@pytest.fixture()
def controller(qtbot, demo_dataset: Path) -> Iterator["AppController"]:  # noqa: F821
    """AppController with the demo dataset opened (and waited for)."""
    from nlapt.app import NLaptApp

    from nlapt_gui.controller import AppController
    from nlapt_gui.settings import UISettings

    ctrl = AppController(NLaptApp(), settings=UISettings())
    with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
        ctrl.open_dataset(demo_dataset)
    yield ctrl


@pytest.fixture()
def toasts(controller) -> list[tuple[str, str]]:
    """Collects (text, kind) pairs emitted via toast_requested."""
    collected: list[tuple[str, str]] = []
    controller.toast_requested.connect(lambda text, kind: collected.append((text, kind)))
    return collected
