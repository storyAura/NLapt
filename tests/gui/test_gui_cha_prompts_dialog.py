"""Tests for nlapt_gui.widgets.cha_prompts_dialog (CHA prompt editor)."""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton

from nlapt_gui.layered_prompts import (
    CARD_PLACEHOLDERS,
    CHARACTER_CARD_PROMPT,
    POSE_SCENE_PROMPT,
    format_card_stats,
    missing_placeholders,
)
from nlapt_gui.widgets.cha_prompts_dialog import (
    BUTTON_RESTORE,
    WARN_MISSING_FMT,
    WINDOW_TITLE,
    CHAPromptsDialog,
)


class TestCHAPromptsDialog:
    def test_empty_settings_show_builtin(self, qtbot) -> None:
        dialog = CHAPromptsDialog()
        qtbot.addWidget(dialog)
        assert dialog.windowTitle() == WINDOW_TITLE
        assert dialog.card_edit.toPlainText() == CHARACTER_CARD_PROMPT
        assert dialog.scene_edit.toPlainText() == POSE_SCENE_PROMPT
        assert dialog.card_stats.text() == format_card_stats(CHARACTER_CARD_PROMPT)
        assert dialog.result_prompts() == ("", "")
        assert dialog.card_warn.isHidden()
        assert dialog.scene_warn.isHidden()

    def test_custom_text_is_kept(self, qtbot) -> None:
        dialog = CHAPromptsDialog("CARD {opening} {name}", "SCENE {ROSTER} {NAMES}")
        qtbot.addWidget(dialog)
        assert dialog.card_edit.toPlainText() == "CARD {opening} {name}"
        assert dialog.scene_edit.toPlainText() == "SCENE {ROSTER} {NAMES}"
        assert dialog.result_prompts() == (
            "CARD {opening} {name}",
            "SCENE {ROSTER} {NAMES}",
        )

    def test_restore_defaults_normalises_to_empty(self, qtbot) -> None:
        dialog = CHAPromptsDialog("custom card", "custom scene")
        qtbot.addWidget(dialog)
        restores = [
            button
            for button in dialog.findChildren(QPushButton)
            if button.text() == BUTTON_RESTORE
        ]
        assert len(restores) == 2
        for button in restores:
            button.click()
        assert dialog.card_edit.toPlainText() == CHARACTER_CARD_PROMPT
        assert dialog.scene_edit.toPlainText() == POSE_SCENE_PROMPT
        assert dialog.result_prompts() == ("", "")

    def test_missing_placeholder_warning(self, qtbot) -> None:
        dialog = CHAPromptsDialog()
        qtbot.addWidget(dialog)
        dialog.card_edit.setPlainText("no placeholders here")
        missing = missing_placeholders("no placeholders here", CARD_PLACEHOLDERS)
        assert not dialog.card_warn.isHidden()
        assert dialog.card_warn.text() == WARN_MISSING_FMT.format(tokens=" ".join(missing))
        dialog.card_edit.setPlainText(CHARACTER_CARD_PROMPT)
        assert dialog.card_warn.isHidden()
        assert dialog.card_warn.text() == ""
