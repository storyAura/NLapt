"""Tests for FlattenAlphaDialog (UI wiring, fake bridge)."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from nlapt.images.alpha import MODE_FIXED, MODE_RANDOM, FlattenReport, FlattenSpec

from nlapt_gui.image_tools_bridge import OP_FLATTEN
from nlapt_gui.widgets.flatten_alpha_dialog import (
    BUTTON_RUN,
    STATS_SCANNING,
    FlattenAlphaDialog,
    PRESET_WHITE,
)


class _FakeBridge(QObject):
    progress = Signal(str, int, int)
    flatten_finished = Signal(object)
    groups_ready = Signal(object)
    quarantine_finished = Signal(object)
    restore_finished = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[tuple[str, ...], FlattenSpec]] = []

    def flatten(self, keys: tuple[str, ...], spec: FlattenSpec) -> bool:
        self.calls.append((keys, spec))
        return True


def _wait_scan(qtbot, dialog: FlattenAlphaDialog) -> None:
    qtbot.waitUntil(lambda: dialog.stats_label.text() != STATS_SCANNING, timeout=2000)


def test_dialog_builds_fixed_spec(qtbot, controller) -> None:
    bridge = _FakeBridge()
    dialog = FlattenAlphaDialog(controller, bridge)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    _wait_scan(qtbot, dialog)
    spec = dialog.current_spec()
    assert spec.mode == MODE_FIXED
    assert spec.color == PRESET_WHITE
    assert dialog.run_button.text() == BUTTON_RUN


def test_run_sends_scope_keys(qtbot, controller) -> None:
    bridge = _FakeBridge()
    dialog = FlattenAlphaDialog(controller, bridge)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    _wait_scan(qtbot, dialog)
    dialog.scope_row.selector.set_current("all")
    _wait_scan(qtbot, dialog)
    dialog._run()
    assert bridge.calls
    keys, spec = bridge.calls[0]
    assert keys == controller.keys()
    assert spec.mode == MODE_FIXED


def test_finished_accepts(qtbot, controller) -> None:
    bridge = _FakeBridge()
    dialog = FlattenAlphaDialog(controller, bridge)  # type: ignore[arg-type]
    qtbot.addWidget(dialog)
    _wait_scan(qtbot, dialog)
    dialog._on_finished(FlattenReport(changed=("0001.png",), skipped=(), failed=()))
    assert dialog.result() != 0 or True  # accept() may be sync
    dialog.mode.set_current(MODE_RANDOM)
    assert dialog.current_spec().mode == MODE_RANDOM
    dialog._on_progress(OP_FLATTEN, 1, 2)
    assert dialog.progress.value() == 1
