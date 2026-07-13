"""Tests for nlapt_gui.widgets.dialogs (centered, fading sub-windows)."""

from __future__ import annotations

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QDialog, QMessageBox, QWidget

from nlapt_gui import anim
from nlapt_gui.widgets.dialogs import (
    CenteredDialog,
    FadingMessageBox,
    ask_confirm,
    center_on_host,
    show_message,
)


class TestCenterOnHost:
    def test_centers_on_visible_parent_window(self, qtbot) -> None:
        host = QWidget()
        qtbot.addWidget(host)
        host.setGeometry(200, 200, 800, 600)
        host.show()
        qtbot.waitExposed(host)
        dialog = QDialog(host)
        dialog.resize(200, 100)
        center_on_host(dialog)
        host_center = host.frameGeometry().center()
        dialog_center = dialog.frameGeometry().center()
        assert (dialog_center - host_center).manhattanLength() <= 4

    def test_no_parent_falls_back_to_screen(self, qtbot) -> None:
        dialog = QDialog()
        qtbot.addWidget(dialog)
        dialog.resize(200, 100)
        center_on_host(dialog)  # must not raise


class TestCenteredDialog:
    def test_show_centers_on_parent(self, qtbot) -> None:
        host = QWidget()
        qtbot.addWidget(host)
        host.setGeometry(150, 150, 900, 700)
        host.show()
        qtbot.waitExposed(host)
        dialog = CenteredDialog(host)
        dialog.resize(300, 200)
        dialog.show()
        qtbot.waitExposed(dialog)
        assert (
            dialog.frameGeometry().center() - host.frameGeometry().center()
        ).manhattanLength() <= 4
        # Animations disabled (conftest): fully opaque, no fade artifacts.
        assert dialog.windowOpacity() == 1.0
        dialog.close()

    def test_done_is_synchronous_when_animations_disabled(self, qtbot) -> None:
        dialog = CenteredDialog()
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.waitExposed(dialog)
        dialog.accept()
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert not dialog.isVisible()

    def test_done_fades_then_closes_when_animations_enabled(self, qtbot) -> None:
        anim.set_animations_enabled(True)
        try:
            dialog = CenteredDialog()
            qtbot.addWidget(dialog)
            dialog.show()
            qtbot.waitExposed(dialog)
            dialog.accept()
            # The close is deferred behind the fade-out animation...
            assert dialog.isVisible()
            qtbot.waitUntil(lambda: not dialog.isVisible(), timeout=2000)
            assert dialog.result() == QDialog.DialogCode.Accepted
        finally:
            anim.set_animations_enabled(False)


class TestMessageHelpers:
    def test_show_message_builds_fading_box(self, qtbot, monkeypatch) -> None:
        captured: list[FadingMessageBox] = []

        def fake_exec(self: FadingMessageBox) -> int:
            captured.append(self)
            return 0

        monkeypatch.setattr(FadingMessageBox, "exec", fake_exec)
        show_message(None, "标题", "正文")
        assert captured
        box = captured[0]
        assert box.windowTitle() == "标题"
        assert box.text() == "正文"

    def test_ask_confirm_true_on_accept(self, qtbot, monkeypatch) -> None:
        def fake_exec(self: FadingMessageBox) -> int:
            for button in self.buttons():
                if self.buttonRole(button) == QMessageBox.ButtonRole.AcceptRole:
                    button.click()
            return 0

        monkeypatch.setattr(FadingMessageBox, "exec", fake_exec)
        assert ask_confirm(None, "删除", "确认?") is True

    def test_ask_confirm_false_on_cancel(self, qtbot, monkeypatch) -> None:
        def fake_exec(self: FadingMessageBox) -> int:
            for button in self.buttons():
                if self.buttonRole(button) == QMessageBox.ButtonRole.RejectRole:
                    button.click()
            return 0

        monkeypatch.setattr(FadingMessageBox, "exec", fake_exec)
        assert ask_confirm(None, "删除", "确认?") is False


class TestNoGraphicsOpacityEffect:
    def test_module_never_uses_graphics_effects(self) -> None:
        # Top-level fades must use windowOpacity: a QGraphicsOpacityEffect on
        # a window crashes Qt during native resizes (v1.3.2 lesson).
        import inspect

        import nlapt_gui.widgets.dialogs as dialogs_module

        source = inspect.getsource(dialogs_module)
        assert "QGraphicsOpacityEffect(" not in source


def test_center_on_host_tolerates_hidden_parent(qtbot) -> None:
    host = QWidget()
    qtbot.addWidget(host)
    dialog = QDialog(host)
    dialog.resize(120, 80)
    center_on_host(dialog)  # hidden parent -> screen fallback; must not raise


def test_show_positions_before_paint(qtbot) -> None:
    """Centering happens inside showEvent, before the first frame is drawn."""
    host = QWidget()
    qtbot.addWidget(host)
    host.setGeometry(100, 100, 600, 400)
    host.show()
    qtbot.waitExposed(host)
    dialog = CenteredDialog(host)
    dialog.resize(200, 120)
    seen_positions: list[QPoint] = []
    original_show_event = CenteredDialog.showEvent

    def spy(self, event):  # noqa: ANN001
        original_show_event(self, event)
        seen_positions.append(self.pos())

    CenteredDialog.showEvent = spy  # type: ignore[method-assign]
    try:
        dialog.show()
        qtbot.waitExposed(dialog)
    finally:
        CenteredDialog.showEvent = original_show_event  # type: ignore[method-assign]
    assert seen_positions
    assert (
        dialog.frameGeometry().center() - host.frameGeometry().center()
    ).manhattanLength() <= 4
