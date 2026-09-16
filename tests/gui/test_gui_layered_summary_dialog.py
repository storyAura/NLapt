"""Tests for the CHA标注 closing dialog (locked cards to copy)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog

from nlapt_gui.layered_prompts import CardRoster, CharacterCard, LayeredSummary
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import DEFAULT_THEME
from nlapt_gui.widgets import layered_summary_flow as flow_module
from nlapt_gui.widgets.layered_summary_dialog import (
    BUTTON_QUARANTINE_FMT,
    CARD_MATCHED_FMT,
    CARD_TITLE_FMT,
    COPIED_ALL,
    COPIED_FMT,
    FAILED_TITLE_FMT,
    NONE_QUARANTINED_FMT,
    NONE_TITLE_FMT,
    ROW_THUMB,
    WINDOW_TITLE,
    WINDOW_TITLE_CANCELLED,
    LayeredSummaryDialog,
)
from nlapt_gui.widgets.main_window import MainWindow
from nlapt_gui.widgets.thumbnails import ThumbnailLoader

EMA = CharacterCard.from_text("ema", "monosaba", "ema, black hair. She wears a red dress.")
RIN = CharacterCard.from_text("rin", "", "rin, white hair. She wears a blue uniform.")
ROSTER = CardRoster((EMA, RIN))


def _summary(**overrides: object) -> LayeredSummary:
    base = {
        "roster": ROSTER,
        "matched_counts": (5, 2),
        "none_keys": ("0003.png",),
        "failed_keys": (),
        "cancelled": False,
    }
    base.update(overrides)
    return LayeredSummary(**base)  # type: ignore[arg-type]


@pytest.fixture()
def dialog(qtbot) -> LayeredSummaryDialog:
    widget = LayeredSummaryDialog(_summary())
    qtbot.addWidget(widget)
    return widget


class TestLayeredSummaryDialog:
    def test_renders_every_card_with_counts(self, dialog) -> None:
        assert dialog.windowTitle() == WINDOW_TITLE
        assert len(dialog.cards) == 2
        first, second = dialog.cards
        assert first.body.toPlainText() == EMA.text
        assert second.body.toPlainText() == RIN.text
        assert first.matched_label.text() == CARD_MATCHED_FMT.format(n=5)
        assert second.matched_label.text() == CARD_MATCHED_FMT.format(n=2)
        titles = [label.text() for label in first.findChildren(type(first.matched_label))]
        assert CARD_TITLE_FMT.format(n=1, name="ema") in titles

    def test_none_rows_and_failed_list(self, qtbot) -> None:
        widget = LayeredSummaryDialog(
            _summary(none_keys=("0003.png", "0004.png"), failed_keys=("0009.png",))
        )
        qtbot.addWidget(widget)
        assert widget.none_title.text() == NONE_TITLE_FMT.format(n=2)
        assert [row.key for row in widget.none_rows] == ["0003.png", "0004.png"]
        assert all(not row.check.isChecked() for row in widget.none_rows)
        assert widget.failed_title.text() == FAILED_TITLE_FMT.format(n=1)
        assert widget.failed_list.toPlainText() == "0009.png"
        assert not widget.none_title.isHidden()
        assert not widget.failed_title.isHidden()
        assert not widget.isModal()

    def test_empty_lists_are_hidden(self, qtbot) -> None:
        widget = LayeredSummaryDialog(_summary(none_keys=(), failed_keys=()))
        qtbot.addWidget(widget)
        assert widget.none_title.isHidden() and widget.none_scroll.isHidden()
        assert widget.failed_title.isHidden() and widget.failed_list.isHidden()
        assert widget.cancelled_note.isHidden()
        assert widget.quarantine_button.isHidden()

    def test_ticked_rows_drive_quarantine_request(self, qtbot) -> None:
        widget = LayeredSummaryDialog(
            _summary(none_keys=("0003.png", "0004.png", "0005.png"))
        )
        qtbot.addWidget(widget)
        requested: list[object] = []
        widget.quarantine_requested.connect(requested.append)
        assert not widget.quarantine_button.isEnabled()
        assert widget.quarantine_button.text() == BUTTON_QUARANTINE_FMT.format(n=0)
        widget.none_rows[0].check.setChecked(True)
        widget.none_rows[2].check.setChecked(True)
        assert widget.quarantine_button.isEnabled()
        assert widget.quarantine_button.text() == BUTTON_QUARANTINE_FMT.format(n=2)
        widget.quarantine_button.click()
        assert requested == [("0003.png", "0005.png")]

        widget.mark_quarantined(("0003.png", "0005.png"))
        assert widget.none_rows[0].quarantined and widget.none_rows[2].quarantined
        assert not widget.none_rows[0].check.isEnabled()
        assert not widget.none_rows[1].quarantined
        assert widget.checked_none_keys() == ()
        assert not widget.quarantine_button.isEnabled()
        assert widget.status_label.text() == NONE_QUARANTINED_FMT.format(n=2)

    def test_double_click_row_activates_key(self, qtbot) -> None:
        widget = LayeredSummaryDialog(_summary(none_keys=("0003.png",)))
        qtbot.addWidget(widget)
        seen: list[str] = []
        widget.key_activated.connect(seen.append)
        qtbot.mouseDClick(widget.none_rows[0], Qt.MouseButton.LeftButton)
        assert seen == ["0003.png"]

    def test_thumbnails_come_from_loader(self, qtbot, controller) -> None:
        loader = ThumbnailLoader()
        widget = LayeredSummaryDialog(
            _summary(none_keys=("0001.png",)),
            loader=loader,
            resolve_path=controller.image_path,
        )
        qtbot.addWidget(widget)
        row = widget.none_rows[0]
        qtbot.waitUntil(lambda: row.thumb.pixmap() is not None and not row.thumb.pixmap().isNull())
        assert row.thumb.pixmap().height() <= ROW_THUMB

    def test_cancelled_changes_title_and_note(self, qtbot) -> None:
        widget = LayeredSummaryDialog(_summary(cancelled=True))
        qtbot.addWidget(widget)
        assert widget.windowTitle() == WINDOW_TITLE_CANCELLED
        assert not widget.cancelled_note.isHidden()

    def test_copy_one_card(self, dialog) -> None:
        dialog.cards[1].copy_button.click()
        assert QGuiApplication.clipboard().text() == RIN.text
        assert dialog.status_label.text() == COPIED_FMT.format(n=2)

    def test_copy_all_joins_with_blank_line(self, dialog) -> None:
        dialog.copy_all_button.click()
        assert QGuiApplication.clipboard().text() == f"{EMA.text}\n\n{RIN.text}"
        assert dialog.all_cards_text() == f"{EMA.text}\n\n{RIN.text}"
        assert dialog.status_label.text() == COPIED_ALL

    def test_close_is_default_and_accepts(self, dialog) -> None:
        assert dialog.close_button.isDefault()
        dialog.close_button.click()
        assert dialog.result() == QDialog.DialogCode.Accepted


class TestMainWindowWiring:
    def test_layered_finished_opens_summary(self, qtbot, controller) -> None:
        manager = ThemeManager(persist=False)
        manager.apply(DEFAULT_THEME)
        window = MainWindow(controller, manager)
        qtbot.addWidget(window)
        window.vision_bridge.layered_finished.emit(_summary())
        assert isinstance(window._layered_summary, LayeredSummaryDialog)
        assert window._layered_summary.cards[0].body.toPlainText() == EMA.text
        # A second batch replaces the previous recap instead of stacking.
        first = window._layered_summary
        window.vision_bridge.layered_finished.emit(_summary(matched_counts=(1, 1)))
        assert window._layered_summary is not first
        # Foreign payloads are ignored.
        window.vision_bridge.layered_finished.emit("not a summary")
        assert window._layered_summary.cards[0].matched_label.text() == CARD_MATCHED_FMT.format(n=1)

    def test_quarantine_request_confirms_then_calls_image_tools(
        self, qtbot, controller, monkeypatch
    ) -> None:
        manager = ThemeManager(persist=False)
        manager.apply(DEFAULT_THEME)
        window = MainWindow(controller, manager)
        qtbot.addWidget(window)
        window.vision_bridge.layered_finished.emit(
            _summary(none_keys=("0001.png", "10_concept/0003.png"))
        )
        dialog = window._layered_summary
        assert dialog is not None
        asked: list[tuple[str, str]] = []
        monkeypatch.setattr(
            flow_module,
            "ask_confirm",
            lambda _parent, title, text, **_kw: asked.append((title, text)) or True,
        )
        sent: list[tuple[Path, ...]] = []

        def fake_quarantine(paths: tuple[Path, ...]) -> bool:
            sent.append(paths)
            return True

        monkeypatch.setattr(window.image_tools, "quarantine", fake_quarantine)
        for row in dialog.none_rows:
            row.check.setChecked(True)
        dialog.quarantine_button.click()
        assert asked and "2" in asked[0][1]
        assert sent == [
            (controller.image_path("0001.png"), controller.image_path("10_concept/0003.png"))
        ]
        # The bridge's completion greys the rows out.
        window.image_tools.quarantine_finished.emit(2)
        assert all(row.quarantined for row in dialog.none_rows)

    def test_quarantine_cancelled_sends_nothing(self, qtbot, controller, monkeypatch) -> None:
        manager = ThemeManager(persist=False)
        manager.apply(DEFAULT_THEME)
        window = MainWindow(controller, manager)
        qtbot.addWidget(window)
        window.vision_bridge.layered_finished.emit(_summary(none_keys=("0001.png",)))
        dialog = window._layered_summary
        monkeypatch.setattr(flow_module, "ask_confirm", lambda *_a, **_k: False)
        sent: list[object] = []
        monkeypatch.setattr(window.image_tools, "quarantine", lambda paths: sent.append(paths) or True)
        dialog.none_rows[0].check.setChecked(True)
        dialog.quarantine_button.click()
        assert sent == []
        assert not dialog.none_rows[0].quarantined

    def test_double_click_row_sets_current_image(self, qtbot, controller) -> None:
        manager = ThemeManager(persist=False)
        manager.apply(DEFAULT_THEME)
        window = MainWindow(controller, manager)
        qtbot.addWidget(window)
        window.vision_bridge.layered_finished.emit(_summary(none_keys=("0002.png",)))
        dialog = window._layered_summary
        qtbot.mouseDClick(dialog.none_rows[0], Qt.MouseButton.LeftButton)
        assert controller.current_key == "0002.png"
