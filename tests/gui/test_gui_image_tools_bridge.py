"""Tests for ImageToolsBridge (flatten / scan / restore on a tmp dataset)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image

from nlapt.images.alpha import FlattenReport, FlattenSpec

from nlapt_gui.image_tools_bridge import (
    TOAST_FLATTEN_DONE,
    TOAST_NO_BACKUP,
    ImageToolsBridge,
)


def _rgba(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (6, 6), (10, 20, 30, 0)).save(path, format="PNG")


@pytest.fixture()
def alpha_controller(qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Controller over a tiny dataset that actually has transparency."""
    from nlapt.app import NLaptApp

    from nlapt_gui.controller import AppController
    from nlapt_gui.settings import UISettings

    root = tmp_path / "ds"
    _rgba(root / "clear.png")
    (root / "clear.txt").write_text("tag", encoding="utf-8")
    Image.new("RGB", (6, 6), (1, 1, 1)).save(root / "solid.png", format="PNG")
    (root / "solid.txt").write_text("solid", encoding="utf-8")
    ctrl = AppController(NLaptApp(), settings=UISettings())
    with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
        ctrl.open_dataset(root)
    return ctrl


def test_flatten_changes_transparent_and_skips_opaque(qtbot, alpha_controller) -> None:
    bridge = ImageToolsBridge(alpha_controller)
    keys = alpha_controller.keys()
    with qtbot.waitSignal(bridge.flatten_finished, timeout=4000) as blocker:
        assert bridge.flatten(keys, FlattenSpec())
    report = blocker.args[0]
    assert isinstance(report, FlattenReport)
    assert "clear.png" in report.changed
    assert "solid.png" in report.skipped
    path = alpha_controller.root / "clear.png" if alpha_controller.root else None
    assert path is not None
    with Image.open(path) as image:
        assert image.mode == "RGB"


def test_restore_without_backup_toasts(qtbot, alpha_controller) -> None:
    collected: list[str] = []
    alpha_controller.toast_requested.connect(lambda text, _kind: collected.append(text))
    bridge = ImageToolsBridge(alpha_controller)
    with qtbot.waitSignal(bridge.restore_finished, timeout=2000):
        assert bridge.restore_last_backup()
    assert TOAST_NO_BACKUP in collected


def test_flatten_toast_mentions_count(qtbot, alpha_controller) -> None:
    collected: list[tuple[str, str]] = []
    alpha_controller.toast_requested.connect(lambda text, kind: collected.append((text, kind)))
    bridge = ImageToolsBridge(alpha_controller)
    with qtbot.waitSignal(bridge.flatten_finished, timeout=4000):
        bridge.flatten(("clear.png",), FlattenSpec())
    assert any(TOAST_FLATTEN_DONE.split("{")[0] in text for text, _k in collected)
