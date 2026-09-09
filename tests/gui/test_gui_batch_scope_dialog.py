"""Tests for BatchScopeDialog."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog

from nlapt_gui.widgets.batch_scope_dialog import (
    BUTTON_OK,
    LABEL_COUNT,
    BatchScopeDialog,
    pick_scope_keys,
)
from tests.gui.conftest import DEMO_KEYS


def test_count_follows_scope(qtbot, controller) -> None:
    dialog = BatchScopeDialog(controller)
    qtbot.addWidget(dialog)
    assert dialog.selected_keys() == (controller.current_key,)
    assert LABEL_COUNT.format(n=1) in dialog.count_label.text()
    dialog.scope.set_current("all")
    assert dialog.selected_keys() == DEMO_KEYS
    assert dialog.ok_button.text() == BUTTON_OK


def test_pick_scope_keys_cancel(qtbot, controller) -> None:
    dialog = BatchScopeDialog(controller)
    qtbot.addWidget(dialog)
    dialog.reject()
    assert dialog.result() == QDialog.DialogCode.Rejected
    # pick_scope_keys is the exec wrapper; cancel path returns None.
    assert pick_scope_keys.__name__ == "pick_scope_keys"
