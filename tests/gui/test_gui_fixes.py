"""Regression tests for the reviewed & confirmed GUI findings.

Each test pins a behavior that a confirmed review finding got wrong; the
letter in the docstring maps to the fix in the review round (controller data
loss, async saves, translate flood, etc.).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, QObject, QPointF, Qt, Signal
from PySide6.QtGui import QMouseEvent

from nlapt_gui.controller import AppController
from nlapt_gui.translate_bridge import has_cjk
from nlapt_gui.widgets.chips_editor import ChipsEditor
from nlapt_gui.widgets.editor_panel import EditorPanel
from nlapt_gui.widgets.preview_panel import PreviewPanel
from nlapt_gui.widgets.sections.translate import TranslateSection

K1 = "0001.png"
K3 = "10_concept/0003.png"


# --------------------------------------------------------------------------- controller

class TestRefreshDataSafety:
    def test_refresh_preserves_unsaved_edits_and_history(self, qtbot, controller) -> None:
        """A: 刷新 must not silently discard unsaved edits or wipe history."""
        controller.set_caption(K1, "edited but unsaved", "编辑分段")
        assert controller.record(K1).dirty
        history_len = len(controller.history.entries(K1))
        assert history_len == 2
        with qtbot.waitSignal(controller.dataset_opened, timeout=2000):
            controller.refresh()
        assert controller.record(K1).text == "edited but unsaved"
        assert controller.record(K1).dirty
        assert len(controller.history.entries(K1)) == history_len


class TestUndo:
    def test_undo_steps_cursor_without_deleting(self, controller) -> None:
        """E (revised): Ctrl+Z moves the history cursor one entry older; the
        newer snapshot stays listed (jump-forward remains possible) and no
        撤销 row is appended."""
        controller.set_current(K1)
        original = controller.record(K1).text
        controller.set_caption(K1, original + ", extra", "编辑分段")
        assert len(controller.history.entries(K1)) == 2
        controller.undo_current()
        assert controller.record(K1).text == original
        entries = controller.history.entries(K1)
        assert len(entries) == 2  # nothing deleted
        assert controller.history.current_index(K1) == 1
        assert all(entry.label != "撤销" for entry in entries)


class TestBatchReentrancy:
    def test_overlapping_batch_rejected(self, controller, toasts) -> None:
        """A batch requested while one is in flight is rejected with a toast."""
        controller._batch_in_flight = True  # simulate an in-flight batch
        controller.replace_all("girl", "woman", False, "all")
        assert (AppController.TOAST_BUSY, "warn") in toasts


class TestReloadConfig:
    def test_controller_reload_config_applies_and_invalidates(self, controller) -> None:
        """N: config changes flow through the public seam, not a private poke."""
        from nlapt.core.config import AppConfig, LLMProfile

        profile = LLMProfile(
            name="default", api_type="openai", base_url="http://x", text_model="m"
        )
        config = AppConfig(profiles=(profile,), active_profile="default")
        controller.reload_config(config)
        assert controller.app.config is config
        assert controller.make_translator_or_none() is not None

    def test_app_reload_config_validates(self) -> None:
        from nlapt.app import NLaptApp
        from nlapt.core.config import AppConfig
        from nlapt.core.errors import ValidationError

        app = NLaptApp()
        config = AppConfig()
        app.reload_config(config)
        assert app.config is config
        with pytest.raises(ValidationError):
            app.reload_config("not a config")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- chips editor

class TestChipIndexShift:
    def test_start_edit_reresolves_index_after_delete(self, qtbot, controller) -> None:
        """I: committing an empty edit shifts indices; the clicked chip still opens."""
        controller.set_caption(K1, "alpha, bravo, charlie", "seed")
        editor = ChipsEditor(controller, K1)
        qtbot.addWidget(editor)
        editor.start_edit(0)  # edit 'alpha'
        editor.set_editor_text("")  # empty -> deletes on commit
        editor.start_edit(1)  # click 'bravo' (build-time index 1)
        assert editor.editor_text() == "bravo"
        assert editor.edit_index() == 0  # bravo moved to index 0 after alpha's delete


# --------------------------------------------------------------------------- translate section

class _CountingBridge(QObject):
    """Stub bridge counting request() calls; always 'configured'."""

    segment_ready = Signal(str, str, str, bool)
    all_done = Signal(str, int, int)

    def __init__(self) -> None:
        super().__init__()
        self.requests = 0
        self.batch: list[tuple[str, str]] = []

    def configured(self) -> bool:
        return True

    def request(self, key: str, text: str, *, fresh: bool = False) -> None:
        self.requests += 1

    def translate_all_cjk(self, key: str, texts) -> None:
        self.batch = [(key, t) for t in texts if has_cjk(t)]


class TestTranslateGating:
    def test_collapsed_section_issues_no_requests(self, qtbot, controller) -> None:
        """J: a collapsed (not-live) 翻译对照 card makes zero LLM requests."""
        bridge = _CountingBridge()
        section = TranslateSection(controller, bridge=bridge)
        qtbot.addWidget(section)
        assert bridge.requests == 0  # __init__ refresh is inert while not live
        section.refresh()
        assert bridge.requests == 0
        section.set_live(True)  # expanding fetches lazily
        assert bridge.requests > 0

    def test_translate_all_commits_after_navigation(self, qtbot, controller) -> None:
        """K: 中文全部转为英文 commits to its own file even after navigating away."""
        bridge = _CountingBridge()
        section = TranslateSection(controller, bridge=bridge)
        qtbot.addWidget(section)
        section.set_live(True)
        controller.set_current(K1)
        section.translate_all_to_english()
        assert bridge.batch, "expected a CJK segment queued for translation"
        source = bridge.batch[0][1]
        controller.set_current("0002.png")  # navigate away mid-flight
        bridge.segment_ready.emit(K1, source, "translated tag", True)
        bridge.all_done.emit(K1, 1, 0)
        assert "translated tag" in controller.record(K1).text
        assert controller.history.entries(K1)[0].label == "中文全部转为英文"

    def test_translate_scope_batch_processes_all_keys(self, qtbot, controller) -> None:
        """指定范围翻译: scope=全部 queues every CJK file and commits each one."""
        controller.set_caption("0002.png", "1girl, 黑色短发, best quality", "seed")
        bridge = _CountingBridge()
        section = TranslateSection(controller, bridge=bridge)
        qtbot.addWidget(section)
        section.scope.set_current("all")
        section.translate_all_to_english()
        # First CJK file dispatched; button shows batch progress and is locked.
        assert not section.all_en_button.isEnabled()
        first_key = bridge.batch[0][0]
        first_src = next(s for _k, s in bridge.batch)
        bridge.segment_ready.emit(first_key, first_src, "en-one", True)
        bridge.all_done.emit(first_key, 1, 0)
        # Second CJK file dispatched automatically.
        second_key = bridge.batch[0][0]
        assert second_key != first_key
        second_src = bridge.batch[0][1]
        bridge.segment_ready.emit(second_key, second_src, "en-two", True)
        bridge.all_done.emit(second_key, 1, 0)
        assert "en-one" in controller.record(first_key).text
        assert "en-two" in controller.record(second_key).text
        assert section.all_en_button.isEnabled()
        assert section.all_en_button.text() == "中文全部转为英文"


# --------------------------------------------------------------------------- editor translate targeting

class _NoopBridge(QObject):
    segment_ready = Signal(str, str, str, bool)

    def configured(self) -> bool:
        return True

    def request(self, key: str, text: str, *, fresh: bool = False) -> None:
        pass


class TestEditorTranslateTargeting:
    def test_reply_does_not_corrupt_a_different_segment(self, qtbot, controller) -> None:
        """H: a slow 翻译 reply only lands in the segment it was requested for."""
        controller.set_caption(K1, "one, two, three", "seed")
        bridge = _NoopBridge()
        panel = EditorPanel(controller, translate_bridge=bridge)
        qtbot.addWidget(panel)
        editor = panel.block_for(K1).editor
        editor.start_edit(0)  # editing 'one'
        panel._on_translate(fresh=False)  # request translation for 'one'
        editor.start_edit(2)  # user moves to 'three' before the reply
        bridge.segment_ready.emit(K1, "one", "XLATED", True)
        assert editor.editor_text() == "three"  # NOT overwritten by 'one's reply


# --------------------------------------------------------------------------- preview zoom

class TestPreviewZoom:
    def test_zoom_resets_to_fit_on_navigation(self, qtbot, controller) -> None:
        """O + P: from 适应, + steps to 120%; switching image resets to 适应."""
        panel = PreviewPanel(controller)
        qtbot.addWidget(panel)
        panel.zoom_in()
        assert panel.zoom_label() == "120%"
        controller.nav(1)
        assert panel.zoom_label() == "适应"


# --------------------------------------------------------------------------- editor rebuild on rescan

class TestEditorRebuild:
    def test_editor_refreshes_after_rescan(self, qtbot, controller, demo_dataset: Path) -> None:
        """G: dataset_opened forces a full editor rebuild (no stale segments)."""
        panel = EditorPanel(controller)
        qtbot.addWidget(panel)
        controller.set_current(K1)
        (demo_dataset / "0001.txt").write_text(
            "NEWTAG_A, NEWTAG_B", encoding="utf-8", newline="\n"
        )
        with qtbot.waitSignal(controller.dataset_opened, timeout=2000):
            controller.refresh()
        editor = panel.block_for(K1).editor
        assert [chip.chip_text for chip in editor.chips()] == ["NEWTAG_A", "NEWTAG_B"]


# --------------------------------------------------------------------------- settings dialog

class TestSettingsSafety:
    def test_save_refuses_to_overwrite_unreadable_config(self, controller, toasts) -> None:
        """M: a corrupt config is not clobbered with defaults on 保存."""
        from nlapt_gui.widgets.settings_dialog import SettingsDialog, config_path

        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        corrupt = "{ this is not valid json"
        path.write_text(corrupt, encoding="utf-8")
        dialog = SettingsDialog(controller)
        dialog.llm_tab.add_profile()
        dialog.llm_tab.base_url.setText("http://localhost:1")
        dialog.llm_tab.new_model_edit.setText("model")
        dialog.llm_tab.add_model()
        dialog.llm_tab.text_combo.setCurrentIndex(1)
        dialog.save()
        assert path.read_text(encoding="utf-8") == corrupt  # untouched
        assert any("无法读取现有配置" in text for text, _ in toasts)


# --------------------------------------------------------------------------- fidelity string / style

class TestFidelity:
    def test_view_tips_match_design(self) -> None:
        """Q: view-mode tooltips use the exact prototype strings."""
        from nlapt_gui.widgets.file_panel import VIEW_TIPS

        assert VIEW_TIPS == {"list": "详细列表", "mid": "中图网格", "big": "大图网格"}

    def test_toggle_chip_radius_is_six(self) -> None:
        """R: toggle chips use the design's 6px radius, not the full pill."""
        from nlapt_gui.theme.qss import TOGGLE_CHIP_RADIUS_PX, build_qss
        from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES

        assert TOGGLE_CHIP_RADIUS_PX == 6
        qss = build_qss(THEMES[DEFAULT_THEME])
        marker = 'QPushButton[toggleChip="true"] {'
        block = qss[qss.index(marker) : qss.index("}", qss.index(marker))]
        assert "border-radius: 6px" in block


# --------------------------------------------------------------------------- title bar

class TestTitleBarButtons:
    def test_window_button_accepts_press(self, qtbot) -> None:
        """S: window buttons consume the press so it can't start a system move."""
        from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES
        from nlapt_gui.widgets.title_bar import KIND_CLOSE, _WindowButton

        button = _WindowButton(KIND_CLOSE, THEMES[DEFAULT_THEME])
        qtbot.addWidget(button)
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(5, 5),
            QPointF(5, 5),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        button.mousePressEvent(event)
        assert event.isAccepted()
