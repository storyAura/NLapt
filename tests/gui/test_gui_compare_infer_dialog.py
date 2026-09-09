"""Tests for the 多对比推标 review dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QDialog

from nlapt.core.config import AppConfig, LLMProfile, ModelRef
from nlapt.core.errors import LLMRequestError
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.compare_bridge import ERROR_TOO_FEW
from nlapt_gui.compare_config import CompareSettings, save_compare_settings
from nlapt_gui.prompt_store import VisionPrompts
from nlapt_gui.widgets.compare_infer_dialog import (
    BATCH_HISTORY_LABEL,
    BUTTON_WRITE_FMT,
    CARD_FAILED_FMT,
    CHOSEN_MARK,
    STATUS_DONE,
    CompareInferDialog,
    open_compare_infer,
)

API_A = "gui-compare-dialog-a"
API_B = "gui-compare-dialog-b"
API_FAIL = "gui-compare-dialog-fail"
register_client(API_A, lambda profile: MockLLMClient(lambda r: f"A says {r.model}"))
register_client(API_B, lambda profile: MockLLMClient(lambda r: f"B says {r.model}"))
register_client(
    API_FAIL,
    lambda profile: MockLLMClient(["x"], fail_times=99, failure=LLMRequestError("boom")),
)

K1 = "0001.png"
K2 = "0002.png"
REF_A = ModelRef("alpha", "a1")
REF_B = ModelRef("beta", "b1")
REF_FAIL = ModelRef("fail", "f1")


def _configure(controller, *, with_fail: bool = False) -> None:
    profiles = [
        LLMProfile(name="alpha", api_type=API_A, base_url="http://a", models=("a1",), enabled_models=("a1",)),
        LLMProfile(name="beta", api_type=API_B, base_url="http://b", models=("b1",), enabled_models=("b1",)),
    ]
    if with_fail:
        profiles.append(
            LLMProfile(name="fail", api_type=API_FAIL, base_url="http://f", models=("f1",), enabled_models=("f1",))
        )
    controller.reload_config(AppConfig(profiles=tuple(profiles)))


def _open(qtbot, controller, keys=(K1, K2), refs=(REF_A, REF_B)) -> CompareInferDialog:
    dialog = CompareInferDialog(controller, keys, refs, prompts=VisionPrompts())
    qtbot.addWidget(dialog)
    assert dialog.start() is None
    qtbot.waitUntil(lambda: dialog.status_label.text() == STATUS_DONE, timeout=4000)
    return dialog


def _card(dialog: CompareInferDialog, ref: ModelRef):
    return next(card for card in dialog.cards if card.ref == ref)


class TestReview:
    def test_cards_fill_and_picking_enables_write(self, qtbot, controller) -> None:
        _configure(controller)
        dialog = _open(qtbot, controller)
        assert dialog.current_key() == K1
        assert _card(dialog, REF_A).caption() == "A says a1"
        assert _card(dialog, REF_B).caption() == "B says b1"
        assert not dialog.write_button.isEnabled()
        _card(dialog, REF_B).radio.click()
        assert dialog.choice_for(K1) == REF_B
        assert dialog.chosen_caption(K1) == "B says b1"
        assert dialog.write_button.text() == BUTTON_WRITE_FMT.format(n=1)
        assert dialog.write_button.isEnabled()
        assert dialog.key_list.item(0).text() == f"{CHOSEN_MARK}{K1}"
        # Browsing to the other image shows its own results and no pick yet.
        dialog.key_list.setCurrentRow(1)
        assert dialog.current_key() == K2
        assert not _card(dialog, REF_A).radio.isChecked()
        assert not _card(dialog, REF_B).radio.isChecked()
        dialog.key_list.setCurrentRow(0)
        assert _card(dialog, REF_B).radio.isChecked()

    def test_edits_are_kept_per_image_and_written(self, qtbot, controller, demo_dataset: Path) -> None:
        _configure(controller)
        dialog = _open(qtbot, controller)
        _card(dialog, REF_A).radio.click()
        _card(dialog, REF_A).text.setPlainText("edited for one")
        dialog.key_list.setCurrentRow(1)
        assert _card(dialog, REF_A).caption() == "A says a1"  # other image untouched
        dialog.key_list.setCurrentRow(0)
        assert _card(dialog, REF_A).caption() == "edited for one"
        assert dialog.chosen_caption(K1) == "edited for one"
        with qtbot.waitSignal(controller.batch_finished, timeout=4000):
            assert dialog.write_chosen() is True
        assert (demo_dataset / "0001.txt").read_text(encoding="utf-8") == "edited for one"
        assert "B says" not in (demo_dataset / "0002.txt").read_text(encoding="utf-8")
        assert controller.record(K2).text.startswith("1girl, school uniform")  # unpicked untouched
        assert dialog.result() == QDialog.DialogCode.Accepted
        assert controller.history.entries(K1)[0].label == BATCH_HISTORY_LABEL

    def test_apply_all_picks_one_model_everywhere(self, qtbot, controller, demo_dataset: Path) -> None:
        _configure(controller)
        dialog = _open(qtbot, controller)
        dialog.apply_all_combo.setCurrentIndex(1)  # beta
        dialog.apply_all_button.click()
        assert dialog.choice_for(K1) == REF_B and dialog.choice_for(K2) == REF_B
        assert dialog.write_button.text() == BUTTON_WRITE_FMT.format(n=2)
        with qtbot.waitSignal(controller.batch_finished, timeout=4000):
            dialog.write_button.click()
        assert (demo_dataset / "0001.txt").read_text(encoding="utf-8") == "B says b1"
        assert (demo_dataset / "0002.txt").read_text(encoding="utf-8") == "B says b1"

    def test_failed_model_cannot_be_picked(self, qtbot, controller) -> None:
        _configure(controller, with_fail=True)
        dialog = _open(qtbot, controller, keys=(K1,), refs=(REF_A, REF_FAIL))
        failed = _card(dialog, REF_FAIL)
        assert not failed.radio.isEnabled()
        assert failed.status.text() == CARD_FAILED_FMT.format(message="boom")
        dialog.apply_all_combo.setCurrentIndex(1)
        dialog.apply_all_button.click()
        assert dialog.choice_for(K1) is None
        assert not dialog.write_button.isEnabled()


class TestLaunch:
    def test_open_toasts_when_too_few_models(self, qtbot, controller, toasts) -> None:
        _configure(controller)
        save_compare_settings(CompareSettings(models=(REF_A,)))
        assert open_compare_infer(controller, (K1,), None) is None
        assert (ERROR_TOO_FEW.format(n=2), "warn") in toasts

    def test_open_runs_configured_models(self, qtbot, controller, monkeypatch) -> None:
        _configure(controller)
        save_compare_settings(CompareSettings(models=(REF_A, REF_B)))
        opened: list[CompareInferDialog] = []

        def fake_exec(self) -> int:  # noqa: ANN001
            opened.append(self)
            return 0

        monkeypatch.setattr(CompareInferDialog, "exec", fake_exec)
        dialog = open_compare_infer(controller, (K1,), None)
        assert dialog is not None and opened == [dialog]
        assert [card.ref for card in dialog.cards] == [REF_A, REF_B]
        qtbot.waitUntil(lambda: dialog.status_label.text() == STATUS_DONE, timeout=4000)
