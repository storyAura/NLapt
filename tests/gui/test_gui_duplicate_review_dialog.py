"""Tests for DuplicateReviewDialog (UI wiring, fake bridge)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from nlapt.images.hashing import DuplicateGroup, ImageFingerprint

from nlapt_gui.widgets.duplicate_review_dialog import (
    BUTTON_SCAN,
    DEFAULT_DISTANCE,
    DuplicateReviewDialog,
)
from nlapt_gui.widgets.thumbnails import ThumbnailLoader


class _FakeBridge(QObject):
    progress = Signal(str, int, int)
    flatten_finished = Signal(object)
    groups_ready = Signal(object)
    quarantine_finished = Signal(object)
    restore_finished = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.scanned: list[tuple[tuple[str, ...], int]] = []
        self.quarantined: list[tuple[Path, ...]] = []

    def scan_duplicates(self, keys: tuple[str, ...], max_distance: int) -> bool:
        self.scanned.append((keys, max_distance))
        return True

    def quarantine(self, paths: tuple[Path, ...]) -> bool:
        self.quarantined.append(paths)
        return True


def _fp(key: str, path: Path, *, width: int = 4, height: int = 3) -> ImageFingerprint:
    return ImageFingerprint(
        key=key,
        path=path,
        dhash=0,
        sha256="abc",
        width=width,
        height=height,
        size=12,
    )


def test_scan_uses_scope_and_slider(qtbot, controller, demo_dataset: Path) -> None:
    bridge = _FakeBridge()
    loader = ThumbnailLoader()
    dialog = DuplicateReviewDialog(controller, bridge, loader)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    assert dialog.scan_button.text() == BUTTON_SCAN
    assert dialog.slider.value() == DEFAULT_DISTANCE
    dialog.scope_row.selector.set_current("all")
    dialog.slider.setValue(0)
    dialog._scan()
    assert bridge.scanned == [(controller.keys(), 0)]


def test_rebuild_groups_and_drop_paths(qtbot, controller, demo_dataset: Path) -> None:
    bridge = _FakeBridge()
    loader = ThumbnailLoader()
    dialog = DuplicateReviewDialog(controller, bridge, loader)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    p1 = controller.image_path("0001.png")
    p2 = controller.image_path("0002.png")
    group = DuplicateGroup(
        members=(_fp("0001.png", p1, width=10, height=10), _fp("0002.png", p2, width=4, height=4)),
        exact=True,
    )
    dialog._on_groups((group,))
    assert "0001.png" in dialog._keep
    assert dialog._keep["0001.png"].isChecked()
    assert not dialog._keep["0002.png"].isChecked()
    dropped = dialog._drop_paths()
    assert p2 in dropped
    assert p1 not in dropped
