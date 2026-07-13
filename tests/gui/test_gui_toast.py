"""Tests for nlapt_gui.widgets.toast (bottom-center toast overlay)."""

from __future__ import annotations

from PySide6.QtWidgets import QWidget

from nlapt_gui.theme.tokens import THEMES
from nlapt_gui.widgets.toast import TOAST_DURATION_MS, ToastOverlay, _dot_color


class TestDotColors:
    def test_kind_mapping_uses_tokens(self) -> None:
        tokens = THEMES["雾灰"]
        assert _dot_color(tokens, "ok") == tokens.ok
        assert _dot_color(tokens, "warn") == tokens.warn
        assert _dot_color(tokens, "err") == tokens.danger
        assert _dot_color(tokens, "info") == tokens.accent
        # Unknown kinds fall back to the prototype default (ok green).
        assert _dot_color(tokens, "whatever") == tokens.ok


class TestOverlay:
    def test_default_duration_matches_design(self, qtbot) -> None:
        overlay = ToastOverlay()
        qtbot.addWidget(overlay)
        assert TOAST_DURATION_MS == 2600
        assert overlay._duration_ms == TOAST_DURATION_MS

    def test_show_toast_stacks_pills(self, qtbot) -> None:
        overlay = ToastOverlay(duration_ms=60_000)
        qtbot.addWidget(overlay)
        overlay.show_toast("已保存 0001.txt", "ok")
        overlay.show_toast("已撤销", "info")
        assert overlay.active_texts() == ("已保存 0001.txt", "已撤销")

    def test_empty_text_ignored(self, qtbot) -> None:
        overlay = ToastOverlay(duration_ms=60_000)
        qtbot.addWidget(overlay)
        overlay.show_toast("", "ok")
        assert overlay.active_texts() == ()

    def test_auto_dismiss_after_interval(self, qtbot) -> None:
        overlay = ToastOverlay(duration_ms=80)
        qtbot.addWidget(overlay)
        overlay.show_toast("已复制标注文本", "ok")
        assert overlay.active_texts() == ("已复制标注文本",)
        qtbot.waitUntil(lambda: overlay.active_texts() == (), timeout=1500)

    def test_dot_color_from_active_tokens(self, qtbot) -> None:
        tokens = THEMES["石墨"]
        overlay = ToastOverlay(duration_ms=60_000, tokens=tokens)
        qtbot.addWidget(overlay)
        overlay.show_toast("hello", "warn")
        dot = overlay._pills[0].layout().itemAt(0).widget()
        assert tokens.warn in dot.styleSheet()

    def test_set_tokens_affects_new_toasts(self, qtbot) -> None:
        overlay = ToastOverlay(duration_ms=60_000)
        qtbot.addWidget(overlay)
        dark = THEMES["墨黑"]
        overlay.set_tokens(dark)
        overlay.show_toast("switched", "err")
        dot = overlay._pills[0].layout().itemAt(0).widget()
        assert dark.danger in dot.styleSheet()

    def test_overlay_tracks_parent_geometry(self, qtbot) -> None:
        parent = QWidget()
        qtbot.addWidget(parent)
        overlay = ToastOverlay(parent, duration_ms=60_000)
        parent.resize(700, 500)
        parent.show()
        qtbot.waitUntil(lambda: overlay.geometry() == parent.rect(), timeout=1500)


class TestControllerIntegration:
    def test_connected_to_toast_requested(self, qtbot, controller) -> None:
        overlay = ToastOverlay(duration_ms=60_000)
        qtbot.addWidget(overlay)
        controller.toast_requested.connect(overlay.show_toast)
        controller.toast_requested.emit("尚未选择任何图片", "warn")
        assert overlay.active_texts() == ("尚未选择任何图片",)
