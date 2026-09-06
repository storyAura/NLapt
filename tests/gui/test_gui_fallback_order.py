"""Tests for the 设置 ▸ 翻译服务 fallback-order editor."""

from __future__ import annotations

from nlapt_gui.translate_config import SUGGESTED_FALLBACK_ORDER
from nlapt_gui.widgets.fallback_order import (
    BUTTON_SUGGEST,
    HINT_FALLBACK,
    FallbackOrderEditor,
)


def _editor(qtbot) -> FallbackOrderEditor:
    editor = FallbackOrderEditor()
    qtbot.addWidget(editor)
    editor.set_primary("deeplx")
    return editor


class TestFallbackOrderEditor:
    def test_hint_mentions_example_chain(self, qtbot) -> None:
        editor = _editor(qtbot)
        assert editor.hint.text() == HINT_FALLBACK
        assert "DeepLX" in HINT_FALLBACK
        assert editor.suggest_button.text() == BUTTON_SUGGEST

    def test_set_order_and_move(self, qtbot) -> None:
        editor = _editor(qtbot)
        editor.set_order(("google", "baidu", "local_mt"))
        assert editor.order() == ("google", "baidu", "local_mt")
        editor.list.setCurrentRow(2)
        editor.move_up()
        assert editor.order() == ("google", "local_mt", "baidu")
        editor.move_down()
        assert editor.order() == ("google", "baidu", "local_mt")

    def test_remove_and_add(self, qtbot) -> None:
        editor = _editor(qtbot)
        editor.set_order(("google", "baidu"))
        editor.list.setCurrentRow(0)
        editor.remove_selected()
        assert editor.order() == ("baidu",)
        index = editor.add_combo.findData("google")
        editor.add_combo.setCurrentIndex(index)
        editor.add_selected()
        assert editor.order() == ("baidu", "google")

    def test_primary_cannot_stay_in_list(self, qtbot) -> None:
        editor = _editor(qtbot)
        editor.set_order(("google", "deeplx", "baidu"))
        assert editor.order() == ("google", "baidu")
        editor.set_primary("google")
        assert editor.order() == ("baidu",)
        assert editor.add_combo.findData("google") < 0

    def test_suggest_fills_recommended_minus_primary(self, qtbot) -> None:
        editor = _editor(qtbot)
        editor.apply_suggested()
        assert editor.order() == SUGGESTED_FALLBACK_ORDER
        editor.set_primary("google")
        editor.apply_suggested()
        assert editor.order() == ("baidu", "local_mt")
