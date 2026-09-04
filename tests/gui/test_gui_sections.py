"""Tests for the right-panel section widgets (fr / ps / tr / hist)."""

from __future__ import annotations

from typing import Iterator

import pytest
from PySide6.QtWidgets import QLabel, QPushButton

from nlapt.app import NLaptApp
from nlapt.core.config import AppConfig, LLMProfile, RequestControl
from nlapt.llm.base import LLMRequest, register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.controller import AppController
from nlapt_gui.settings import UISettings
from nlapt_gui.translate_bridge import NOTE_UNCONFIGURED
from nlapt_gui.widgets.sections.find_replace import (
    INFO_MATCHES,
    INFO_NO_MATCH,
    INFO_PROMPT,
    FindReplaceSection,
)
from nlapt_gui.widgets.sections.history import (
    BUTTON_REVERT,
    PILL_CURRENT,
    TOAST_CLEARED,
    HistorySection,
)
from nlapt_gui.widgets.sections.prefix_suffix import PrefixSuffixSection
from nlapt_gui.widgets.sections.translate import (
    MODE_SEGMENTS,
    MODE_WHOLE,
    PENDING_NOTE,
    TOAST_NO_CJK,
    TOAST_SWAPPED,
    TranslateSection,
)

# Unique api_type for this test module.
API_TYPE = "gui-sections-mock"

# Deterministic "translations" keyed by the source segment text.
TRANSLATIONS: dict[str, str] = {
    "1girl": "一个女孩",
    "solo": "单人",
    "long hair": "长发",
    "少女站在樱花树下。masterpiece": "a girl under the cherry tree, masterpiece",
}


def _provider(request: LLMRequest) -> str:
    prompt = request.messages[0].text
    for source, translated in TRANSLATIONS.items():
        if source in prompt:
            return translated
    return "mock-out"


register_client(API_TYPE, lambda profile: MockLLMClient(_provider))


def _configured_app() -> NLaptApp:
    profile = LLMProfile(
        name="default", api_type=API_TYPE, base_url="http://mock", text_model="m1"
    )
    config = AppConfig(
        profiles=(profile,),
        active_profile="default",
        request=RequestControl(max_retries=0),
    )
    return NLaptApp(config=config)


@pytest.fixture()
def tr_controller(qtbot, demo_dataset) -> Iterator[AppController]:
    """Controller with the demo dataset open and a working mock LLM."""
    ctrl = AppController(_configured_app(), settings=UISettings())
    with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
        ctrl.open_dataset(demo_dataset)
    yield ctrl


@pytest.fixture()
def tr_toasts(tr_controller) -> list[tuple[str, str]]:
    collected: list[tuple[str, str]] = []
    tr_controller.toast_requested.connect(
        lambda text, kind: collected.append((text, kind))
    )
    return collected


class TestFindReplaceSection:
    def test_initial_prompt_info(self, qtbot, controller) -> None:
        section = FindReplaceSection(controller)
        qtbot.addWidget(section)
        assert section.info_label.text() == INFO_PROMPT

    def test_live_match_info_current_scope(self, qtbot, controller) -> None:
        section = FindReplaceSection(controller)
        qtbot.addWidget(section)
        section.find_input.setText("1girl")
        assert section.info_label.text() == INFO_MATCHES.format(total=1, files=1)
        assert section.info_label.property("muted") is False

    def test_live_match_info_all_scope(self, qtbot, controller) -> None:
        section = FindReplaceSection(controller)
        qtbot.addWidget(section)
        section.find_input.setText("1girl")
        section.scope.set_current("all")
        assert section.info_label.text() == INFO_MATCHES.format(total=3, files=3)

    def test_no_match_info(self, qtbot, controller) -> None:
        section = FindReplaceSection(controller)
        qtbot.addWidget(section)
        section.find_input.setText("zzz-not-there")
        assert section.info_label.text() == INFO_NO_MATCH
        assert section.info_label.property("muted") is True

    def test_case_sensitivity_toggle_updates_info(self, qtbot, controller) -> None:
        section = FindReplaceSection(controller)
        qtbot.addWidget(section)
        section.find_input.setText("1GIRL")
        assert section.info_label.text() == INFO_MATCHES.format(total=1, files=1)
        section.case_toggle.setChecked(True)
        assert section.info_label.text() == INFO_NO_MATCH
        assert section.case_toggle.property("chipOn") is True

    def test_info_updates_on_selection_change(self, qtbot, controller) -> None:
        section = FindReplaceSection(controller)
        qtbot.addWidget(section)
        section.scope.set_current("selected")
        section.find_input.setText("1girl")
        assert section.info_label.text() == INFO_NO_MATCH
        controller.toggle_selected("0001.png")
        controller.toggle_selected("0002.png")
        assert section.info_label.text() == INFO_MATCHES.format(total=2, files=2)

    def test_replace_all_runs_batch_and_toasts(self, qtbot, controller, toasts) -> None:
        section = FindReplaceSection(controller)
        qtbot.addWidget(section)
        section.find_input.setText("solo")
        section.replace_input.setText("duo")
        with qtbot.waitSignal(controller.batch_finished, timeout=2000):
            section.apply_button.click()
        assert "duo" in controller.record("0001.png").text
        assert "solo" not in controller.record("0001.png").text
        assert ("已在 1 个文件中替换 1 处", "ok") in toasts
        assert controller.history.entries("0001.png")[0].label == "查找替换 ×1"

    def test_replace_all_empty_find_warns(self, qtbot, controller, toasts) -> None:
        section = FindReplaceSection(controller)
        qtbot.addWidget(section)
        section.apply_button.click()
        assert ("请输入查找内容", "warn") in toasts

    def test_whole_word_toggle_updates_info(self, qtbot, controller) -> None:
        section = FindReplaceSection(controller)
        qtbot.addWidget(section)
        assert section.word_toggle.isChecked() is False
        section.find_input.setText("girl")  # 词内子串:命中 "1girl"
        assert section.info_label.text() == INFO_MATCHES.format(total=1, files=1)
        section.word_toggle.setChecked(True)  # 整词:不再匹配词内子串
        assert section.info_label.text() == INFO_NO_MATCH
        assert section.word_toggle.property("chipOn") is True

    def test_replace_all_whole_word_via_toggle(self, qtbot, controller, toasts) -> None:
        section = FindReplaceSection(controller)
        qtbot.addWidget(section)
        section.find_input.setText("hair")
        section.replace_input.setText("HAIR")
        section.word_toggle.setChecked(True)
        with qtbot.waitSignal(controller.batch_finished, timeout=2000):
            section.apply_button.click()
        assert "long HAIR" in controller.record("0001.png").text


class TestPrefixSuffixSection:
    def test_defaults_match_design(self, qtbot, controller) -> None:
        section = PrefixSuffixSection(controller)
        qtbot.addWidget(section)
        assert section.as_tag is True
        assert section.position.current == "prefix"
        assert section.scope.scope == "current"

    def test_apply_prefix_as_tag(self, qtbot, controller, toasts) -> None:
        section = PrefixSuffixSection(controller)
        qtbot.addWidget(section)
        section.text_input.setText("aoba")
        with qtbot.waitSignal(controller.batch_finished, timeout=2000):
            section.apply_button.click()
        assert controller.record("0001.png").text.startswith("aoba, ")
        assert controller.history.entries("0001.png")[0].label == "添加前缀「aoba」"
        assert ("已为 1 个文件添加前缀", "ok") in toasts

    def test_apply_suffix(self, qtbot, controller, toasts) -> None:
        controller.set_current("0002.png")
        section = PrefixSuffixSection(controller)
        qtbot.addWidget(section)
        section.text_input.setText("masterpiece")
        section.position.set_current("suffix")
        with qtbot.waitSignal(controller.batch_finished, timeout=2000):
            section.apply_button.click()
        assert controller.record("0002.png").text.endswith(", masterpiece")
        assert ("已为 1 个文件添加后缀", "ok") in toasts

    def test_empty_text_warns(self, qtbot, controller, toasts) -> None:
        section = PrefixSuffixSection(controller)
        qtbot.addWidget(section)
        section.apply_button.click()
        assert ("请输入前缀/后缀内容", "warn") in toasts

    def test_as_tag_toggle_updates_chip(self, qtbot, controller) -> None:
        section = PrefixSuffixSection(controller)
        qtbot.addWidget(section)
        section.as_tag_toggle.setChecked(False)
        assert section.as_tag is False
        assert section.as_tag_toggle.property("chipOn") is False


class TestTranslateSection:
    def test_rows_match_segments(self, qtbot, tr_controller) -> None:
        section = TranslateSection(tr_controller)
        qtbot.addWidget(section)
        sources = [row.source for row in section.rows()]
        assert sources == list(tr_controller.segments("0001.png"))
        assert len(sources) == 4

    def test_rows_receive_translations(self, qtbot, tr_controller) -> None:
        section = TranslateSection(tr_controller)
        qtbot.addWidget(section)
        section.set_live(True)  # the card is inert until expanded
        qtbot.waitUntil(
            lambda: all(row.ready for row in section.rows()), timeout=2000
        )
        first = section.rows()[0]
        assert first.dst_label.text() == TRANSLATIONS["1girl"]
        assert first.swap_button.isEnabled()

    def test_swap_replaces_segment(self, qtbot, tr_controller, tr_toasts) -> None:
        section = TranslateSection(tr_controller)
        qtbot.addWidget(section)
        section.set_live(True)
        qtbot.waitUntil(
            lambda: all(row.ready for row in section.rows()), timeout=2000
        )
        section.rows()[0].swap_button.click()
        assert tr_controller.record("0001.png").text.startswith("一个女孩, ")
        assert (
            tr_controller.history.entries("0001.png")[0].label == "翻译替换"
        )
        assert (TOAST_SWAPPED, "info") in tr_toasts

    def test_translate_all_cjk_replaces_segments(
        self, qtbot, tr_controller, tr_toasts
    ) -> None:
        section = TranslateSection(tr_controller)
        qtbot.addWidget(section)
        section.set_live(True)
        qtbot.waitUntil(
            lambda: all(row.ready for row in section.rows()), timeout=2000
        )
        with qtbot.waitSignal(section.bridge.all_done, timeout=2000):
            section.all_en_button.click()
        text = tr_controller.record("0001.png").text
        assert "少女站在樱花树下。masterpiece" not in text
        assert "a girl under the cherry tree, masterpiece" in text
        assert (
            tr_controller.history.entries("0001.png")[0].label == "中文全部转为英文"
        )
        assert ("已将 1 个中文片段转为英文", "ok") in tr_toasts

    def test_translate_all_without_cjk_toasts(
        self, qtbot, tr_controller, tr_toasts
    ) -> None:
        tr_controller.set_current("0002.png")
        section = TranslateSection(tr_controller)
        qtbot.addWidget(section)
        section.all_en_button.click()
        assert (TOAST_NO_CJK, "info") in tr_toasts

    def test_unconfigured_guided_state(self, qtbot, controller) -> None:
        section = TranslateSection(controller)
        qtbot.addWidget(section)
        assert not section.bridge.configured()
        assert section.hint_row.isVisibleTo(section)
        assert not section.all_en_button.isEnabled()
        for row in section.rows():
            assert row.dst_label.text() == NOTE_UNCONFIGURED
            assert not row.swap_button.isEnabled()

    def test_settings_link_emits_request(self, qtbot, controller) -> None:
        section = TranslateSection(controller)
        qtbot.addWidget(section)
        with qtbot.waitSignal(section.open_settings_requested, timeout=1000):
            section.settings_link.click()

    def test_pending_note_before_result(self, qtbot, tr_controller) -> None:
        section = TranslateSection(tr_controller)
        qtbot.addWidget(section)
        # Rows start pending or already resolved (queued worker results).
        for row in section.rows():
            assert row.dst_label.text() in (
                PENDING_NOTE,
                TRANSLATIONS.get(row.source, "mock-out"),
            )


class TestTranslateSectionWholeMode:
    """整段对照: the whole caption is one compare unit."""

    FULL_TEXT = "1girl, solo, long hair, 少女站在樱花树下。masterpiece"

    def make_whole(self, qtbot, tr_controller) -> TranslateSection:
        section = TranslateSection(tr_controller)
        qtbot.addWidget(section)
        section.mode_bar.set_current(MODE_WHOLE)
        return section

    def test_whole_mode_shows_one_row_with_full_caption(
        self, qtbot, tr_controller
    ) -> None:
        section = self.make_whole(qtbot, tr_controller)
        assert section.whole_mode
        rows = section.rows()
        assert [row.source for row in rows] == [self.FULL_TEXT]

    def test_switch_back_restores_segment_rows(self, qtbot, tr_controller) -> None:
        section = self.make_whole(qtbot, tr_controller)
        section.mode_bar.set_current(MODE_SEGMENTS)
        assert not section.whole_mode
        assert len(section.rows()) == 4

    def test_whole_swap_replaces_entire_caption(
        self, qtbot, tr_controller, tr_toasts
    ) -> None:
        section = self.make_whole(qtbot, tr_controller)
        section.set_live(True)
        qtbot.waitUntil(
            lambda: all(row.ready for row in section.rows()), timeout=2000
        )
        section.rows()[0].swap_button.click()
        # The mock maps the full caption (contains "1girl") to 一个女孩.
        assert tr_controller.record("0001.png").text == "一个女孩"
        assert tr_controller.history.entries("0001.png")[0].label == "翻译替换"
        assert (TOAST_SWAPPED, "info") in tr_toasts

    def test_whole_mode_translate_all_commits_one_unit(
        self, qtbot, tr_controller, tr_toasts
    ) -> None:
        section = self.make_whole(qtbot, tr_controller)
        with qtbot.waitSignal(section.bridge.all_done, timeout=2000):
            section.all_en_button.click()
        assert tr_controller.record("0001.png").text == "一个女孩"
        assert ("已将 1 个中文片段转为英文", "ok") in tr_toasts
        assert section.mode_bar.isEnabled()  # unlocked after the batch


class TestHistorySection:
    def test_initial_single_row_with_current_pill(self, qtbot, controller) -> None:
        section = HistorySection(controller)
        qtbot.addWidget(section)
        assert section.row_count() == 1
        row = section._row_frames[0]
        pill_labels = [
            label for label in row.findChildren(QLabel) if label.text() == PILL_CURRENT
        ]
        assert len(pill_labels) == 1
        assert pill_labels[0].property("pill") == "accentSoft"
        assert row.property("accentSoftBox") is True
        assert BUTTON_REVERT not in [b.text() for b in row.findChildren(QPushButton)]

    def test_rows_grow_with_edits(self, qtbot, controller) -> None:
        section = HistorySection(controller)
        qtbot.addWidget(section)
        controller.set_caption("0001.png", "edited text", "自由编辑")
        assert section.row_count() == 2
        revert_buttons = section._rows_host.findChildren(QPushButton, "histRevert")
        assert len(revert_buttons) == 1

    def test_revert_restores_caption(self, qtbot, controller, toasts) -> None:
        section = HistorySection(controller)
        qtbot.addWidget(section)
        original = controller.record("0001.png").text
        controller.set_caption("0001.png", "edited text", "自由编辑")
        revert_button = section._rows_host.findChildren(QPushButton, "histRevert")[0]
        revert_button.click()
        assert controller.record("0001.png").text == original
        # Cursor model: the newer entry stays listed so the user can jump back.
        assert section.row_count() == 2
        assert controller.history.current_index("0001.png") == 1
        assert any(text.startswith("已回退到 ") for text, _kind in toasts)
        # ...and jumping forward again works from the panel (wait for the old
        # rows' deleteLater so findChildren only sees the rebuilt buttons).
        qtbot.waitUntil(
            lambda: len(section._rows_host.findChildren(QPushButton, "histRevert")) == 1,
            timeout=2000,
        )
        forward_button = section._rows_host.findChildren(QPushButton, "histRevert")[0]
        forward_button.click()
        assert controller.record("0001.png").text == "edited text"
        assert controller.history.current_index("0001.png") == 0

    def test_clear_history_keeps_current(self, qtbot, controller, toasts) -> None:
        section = HistorySection(controller)
        qtbot.addWidget(section)
        controller.set_caption("0001.png", "v2", "编辑")
        controller.set_caption("0001.png", "v3", "编辑")
        assert section.row_count() == 3
        section.clear_button.click()
        assert section.row_count() == 1
        assert controller.record("0001.png").text == "v3"
        assert (TOAST_CLEARED, "info") in toasts

    def test_rows_follow_current_file(self, qtbot, controller) -> None:
        section = HistorySection(controller)
        qtbot.addWidget(section)
        controller.set_caption("0001.png", "changed", "编辑")
        assert section.row_count() == 2
        controller.set_current("0002.png")
        assert section.row_count() == 1
