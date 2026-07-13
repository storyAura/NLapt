"""Tests for nlapt_gui.widgets.editor_panel (tabs, blocks, floating toolbar)."""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal

import nlapt_gui.widgets as widgets_pkg
from nlapt_gui.widgets.chips_editor import ChipsEditor, TOAST_SPLIT_DONE
from nlapt_gui.widgets.editor_panel import (
    MODE_HINTS,
    TOAST_TRANSLATE_UNCONFIGURED,
    EditorPanel,
    _BlockHeader,
)
from nlapt_gui.widgets.sents_editor import SentsEditor
from nlapt_gui.widgets.text_editor import TextEditor

KEY1 = "0001.png"
KEY2 = "0002.png"

OWNED_MODULES = (
    "editor_panel.py",
    "chips_editor.py",
    "sents_editor.py",
    "text_editor.py",
    "flow_layout.py",
)


class StubBridge(QObject):
    """Duck-typed TranslateBridge stand-in (request() + segment_ready)."""

    segment_ready = Signal(str, str, str, bool)
    all_done = Signal(str, int, int)

    def __init__(self, *, configured: bool = True) -> None:
        super().__init__()
        self._configured = configured
        self.requests: list[tuple[str, str, bool]] = []

    def configured(self) -> bool:
        return self._configured

    def request(self, key: str, text: str, *, fresh: bool = False) -> None:
        self.requests.append((key, text, fresh))


def make_panel(qtbot, controller, bridge: object | None = None) -> EditorPanel:
    panel = EditorPanel(controller, bridge)
    qtbot.addWidget(panel)
    panel.resize(760, 480)
    panel.show()
    return panel


class TestLayout:
    def test_single_mode_has_one_block_for_current(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        assert [block.key for block in panel.blocks()] == [KEY1]
        assert isinstance(panel.blocks()[0].editor, ChipsEditor)

    def test_mode_tabs_have_design_labels(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        labels = [panel._tab_buttons[m].text() for m in ("chips", "sents", "text")]
        assert labels == ["胶囊", "分句", "文本"]
        assert panel._tab_buttons["chips"].property("segActive") == "true"

    def test_hint_and_char_info(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        assert panel.hint_text() == MODE_HINTS["chips"]
        assert panel.char_info_text() == controller.char_seg_info(KEY1)

    def test_tab_click_switches_mode_and_editor_type(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        qtbot.mouseClick(panel._tab_buttons["sents"], Qt.MouseButton.LeftButton)
        assert controller.mode == "sents"
        assert isinstance(panel.blocks()[0].editor, SentsEditor)
        assert panel.hint_text() == MODE_HINTS["sents"]
        assert panel._tab_buttons["sents"].property("segActive") == "true"
        qtbot.mouseClick(panel._tab_buttons["text"], Qt.MouseButton.LeftButton)
        assert isinstance(panel.blocks()[0].editor, TextEditor)
        assert panel.hint_text() == MODE_HINTS["text"]

    def test_caption_change_refreshes_info_and_block(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        controller.set_caption(KEY1, "a, b", "测试")
        assert panel.char_info_text() == "4 字符 · 2 段"
        chips = panel.blocks()[0].editor.chips()
        assert tuple(chip.chip_text for chip in chips) == ("a", "b")


class TestMultiMode:
    def _select_two(self, controller) -> None:
        controller.toggle_selected(KEY1)
        controller.toggle_selected(KEY2)

    def test_two_blocks_with_headers_and_badge(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        self._select_two(controller)
        assert [block.key for block in panel.blocks()] == [KEY1, KEY2]
        first, second = panel.blocks()
        # isHidden reflects the explicit show/hide state without needing
        # pending show events to be delivered (offscreen determinism).
        assert first._badge is not None and not first._badge.isHidden()  # 编辑中 on current
        assert second._badge is not None and second._badge.isHidden()
        assert first.styleSheet() != ""  # accent border on the current block
        assert second.styleSheet() == ""
        assert first._stat.text() == controller.char_seg_info(KEY1)

    def test_header_click_focuses_block(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        self._select_two(controller)
        second = panel.blocks()[1]
        header = second.findChild(_BlockHeader)
        qtbot.waitUntil(lambda: header.width() > 0, timeout=2000)
        qtbot.mouseClick(header, Qt.MouseButton.LeftButton)
        assert controller.current_key == KEY2
        assert not second._badge.isHidden()
        assert panel.blocks()[0]._badge.isHidden()

    def test_toolbar_label_prefixed_with_name(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        self._select_two(controller)
        panel.blocks()[1].editor.start_edit(0)
        assert panel.toolbar.is_active()
        assert panel.toolbar.label_text() == f"{KEY2} · 第 1 段"
        assert controller.current_key == KEY2  # start_edit focused the block


class TestFloatingToolbar:
    def test_hidden_until_edit_starts(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        assert not panel.toolbar.is_active()
        panel.blocks()[0].editor.start_edit(0)
        assert panel.toolbar.is_active()
        assert panel.toolbar.label_text() == "第 1 段"

    def test_dismisses_after_commit(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        editor = panel.blocks()[0].editor
        editor.start_edit(0)
        editor.commit_active()
        assert not panel.toolbar.is_active()

    def test_buttons_never_take_focus(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        toolbar = panel.toolbar
        for button in (
            toolbar.split_btn,
            toolbar.insert_btn,
            toolbar.translate_btn,
            toolbar.retranslate_btn,
            toolbar.delete_btn,
        ):
            assert button.focusPolicy() == Qt.FocusPolicy.NoFocus

    def test_split_button_press_splits_segment(self, qtbot, controller, toasts) -> None:
        panel = make_panel(qtbot, controller)
        editor = panel.blocks()[0].editor
        editor.start_edit(2)  # "long hair"
        editor._field.setCursorPosition(4)
        qtbot.mousePress(panel.toolbar.split_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseRelease(panel.toolbar.split_btn, Qt.MouseButton.LeftButton)
        assert controller.segments(KEY1)[2:4] == ("long", "hair")
        assert (TOAST_SPLIT_DONE, "ok") in toasts
        assert not panel.toolbar.is_active()

    def test_insert_button_opens_insert_editor(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        editor = panel.blocks()[0].editor
        editor.start_edit(0)
        qtbot.mousePress(panel.toolbar.insert_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseRelease(panel.toolbar.insert_btn, Qt.MouseButton.LeftButton)
        assert editor.is_inserting()
        assert panel.toolbar.label_text() == "插入新段 · 第 2 位"

    def test_delete_button_removes_segment(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        editor = panel.blocks()[0].editor
        editor.start_edit(1)
        qtbot.mousePress(panel.toolbar.delete_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseRelease(panel.toolbar.delete_btn, Qt.MouseButton.LeftButton)
        assert "solo" not in controller.segments(KEY1)
        assert not panel.toolbar.is_active()

    def test_delete_button_cancels_insert(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        editor = panel.blocks()[0].editor
        before = controller.record(KEY1).text
        editor.start_insert_end()
        editor.set_editor_text("pending")
        qtbot.mousePress(panel.toolbar.delete_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseRelease(panel.toolbar.delete_btn, Qt.MouseButton.LeftButton)
        assert controller.record(KEY1).text == before
        assert not panel.toolbar.is_active()

    def test_no_toolbar_in_text_mode(self, qtbot, controller) -> None:
        panel = make_panel(qtbot, controller)
        controller.set_mode("text")
        assert panel.active_editor() is None
        assert not panel.toolbar.is_active()


class TestTranslate:
    def test_none_bridge_warns(self, qtbot, controller, toasts) -> None:
        panel = make_panel(qtbot, controller, None)
        panel.blocks()[0].editor.start_edit(0)
        qtbot.mousePress(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseRelease(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        assert (TOAST_TRANSLATE_UNCONFIGURED, "warn") in toasts

    def test_unconfigured_bridge_warns(self, qtbot, controller, toasts) -> None:
        bridge = StubBridge(configured=False)
        panel = make_panel(qtbot, controller, bridge)
        panel.blocks()[0].editor.start_edit(0)
        qtbot.mousePress(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseRelease(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        assert (TOAST_TRANSLATE_UNCONFIGURED, "warn") in toasts
        assert bridge.requests == []

    def test_translate_previews_first_then_applies(self, qtbot, controller) -> None:
        """翻译 shows the result in a preview first; 替换 applies it on demand."""
        bridge = StubBridge()
        panel = make_panel(qtbot, controller, bridge)
        editor = panel.blocks()[0].editor
        editor.start_edit(2)  # "long hair"
        qtbot.mousePress(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseRelease(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        assert bridge.requests == [(KEY1, "long hair", False)]
        bridge.segment_ready.emit(KEY1, "long hair", "长发", True)
        # View first: nothing replaced yet, the preview holds the translation.
        assert editor.editor_text() == "long hair"
        assert panel.translation_preview.is_active()
        assert panel.translation_preview.current_text() == "长发"
        # 替换 applies into the still-open inline editor.
        panel.translation_preview.apply_btn.pressed.emit()
        assert editor.editor_text() == "长发"
        assert editor.has_active()  # edit stays open for further tweaks
        assert not panel.translation_preview.is_active()

    def test_translate_preview_view_only_dismiss(self, qtbot, controller) -> None:
        """关闭 keeps the original text (纯查看)."""
        bridge = StubBridge()
        panel = make_panel(qtbot, controller, bridge)
        editor = panel.blocks()[0].editor
        editor.start_edit(2)
        qtbot.mousePress(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseRelease(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        bridge.segment_ready.emit(KEY1, "long hair", "长发", True)
        assert panel.translation_preview.is_active()
        panel.translation_preview.close_btn.pressed.emit()
        assert editor.editor_text() == "long hair"
        assert not panel.translation_preview.is_active()

    def test_retranslate_bypasses_cache(self, qtbot, controller) -> None:
        bridge = StubBridge()
        panel = make_panel(qtbot, controller, bridge)
        panel.blocks()[0].editor.start_edit(0)
        qtbot.mousePress(panel.toolbar.retranslate_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseRelease(panel.toolbar.retranslate_btn, Qt.MouseButton.LeftButton)
        assert bridge.requests == [(KEY1, "1girl", True)]

    def test_mismatched_result_is_ignored(self, qtbot, controller) -> None:
        bridge = StubBridge()
        panel = make_panel(qtbot, controller, bridge)
        editor = panel.blocks()[0].editor
        editor.start_edit(0)
        qtbot.mousePress(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseRelease(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        bridge.segment_ready.emit(KEY1, "other source", "wrong", True)
        assert editor.editor_text() == "1girl"

    def test_failed_result_is_ignored(self, qtbot, controller) -> None:
        bridge = StubBridge()
        panel = make_panel(qtbot, controller, bridge)
        editor = panel.blocks()[0].editor
        editor.start_edit(0)
        qtbot.mousePress(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        qtbot.mouseRelease(panel.toolbar.translate_btn, Qt.MouseButton.LeftButton)
        bridge.segment_ready.emit(KEY1, "1girl", "boom", False)
        assert editor.editor_text() == "1girl"


class TestNoHardcodedColors:
    def test_owned_widget_modules_have_no_hex_colors(self) -> None:
        """Design mandate: every color comes from ThemeTokens / QSS / palette."""
        hex_color = re.compile(r"#[0-9A-Fa-f]{6}\b")
        widgets_dir = Path(widgets_pkg.__file__).parent
        for name in OWNED_MODULES:
            source = (widgets_dir / name).read_text(encoding="utf-8")
            assert not hex_color.search(source), f"hardcoded color in {name}"
