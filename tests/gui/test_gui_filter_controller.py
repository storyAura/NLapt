"""Real-dataset coverage for filtered navigation and selection scopes."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.app import NLaptApp
from nlapt_gui.controller import AppController
from nlapt_gui.settings import UISettings

from tests.gui.conftest import DEMO_CAPTIONS, DEMO_KEYS

K1, K2, K3, K4 = DEMO_KEYS
FOLDER = "10_concept"


@pytest.mark.parametrize("delta, expected", [(1, K1), (-1, K2)])
def test_unmatched_navigation_enters_result_boundary(
    controller: AppController, delta: int, expected: str
) -> None:
    controller.set_current(K3)
    controller.set_filter("hair")
    assert controller.current_key == K3
    assert controller.pos_label() == "未匹配 / 2"
    assert controller.can_navigate()
    controller.nav(delta)
    assert controller.current_key == expected
    controller.nav(delta)
    assert controller.current_key == (K2 if delta > 0 else K1)


def test_filter_preserves_current_draft_and_history(controller: AppController) -> None:
    controller.set_current(K2)
    controller.set_caption(K2, "unsaved draft", "edit")
    entries = controller.history.entries(K2)
    current_changes: list[str] = []
    controller.current_changed.connect(current_changes.append)

    controller.set_filter("0003")
    assert controller.pos_label() == "未匹配 / 1"
    controller.set_filter("no matching result")
    assert controller.pos_label() == "未匹配 / 0"
    assert not controller.can_navigate()
    controller.set_filter("")
    assert controller.pos_label() == "2 / 4"
    assert controller.can_navigate()
    assert current_changes == []
    assert controller.current_key == K2
    assert controller.record(K2).text == "unsaved draft"
    assert controller.record(K2).dirty
    assert controller.history.entries(K2) == entries


def test_empty_dataset_cannot_navigate(qapp) -> None:
    controller = AppController(NLaptApp(), settings=UISettings())
    assert controller.pos_label() == "- / 0"
    assert not controller.can_navigate()
    controller.nav(1)
    controller.nav(-1)
    assert controller.current_key is None


def test_filter_signals_observe_intersected_selection_without_current_jump(
    controller: AppController,
) -> None:
    controller.select_all()
    controller.set_current(K4)
    changes: list[tuple[str, tuple[str, ...], str | None]] = []
    current_changes: list[str] = []
    controller.filter_changed.connect(
        lambda _text: changes.append(
            ("filter", controller.selected_keys(), controller.current_key)
        )
    )
    controller.selection_changed.connect(
        lambda: changes.append(
            ("selection", controller.selected_keys(), controller.current_key)
        )
    )
    controller.current_changed.connect(current_changes.append)

    controller.set_filter("1girl")
    assert changes == [
        ("filter", (K1, K2, K3), K4),
        ("selection", (K1, K2, K3), K4),
    ]
    assert current_changes == []
    assert controller.multi_mode()
    assert controller.editor_keys() == (K1, K2, K3)
    assert controller.pos_label() == "- / 3"

    controller.set_filter("0003")
    assert controller.selected_keys() == (K3,)
    assert not controller.multi_mode()
    assert controller.editor_keys() == (K4,)
    assert controller.pos_label() == "未匹配 / 1"
    assert controller.current_key == K4


def test_search_change_does_not_restore_hidden_selection(controller: AppController) -> None:
    controller.select_all()
    controller.set_filter("0003")
    assert controller.selected_keys() == (K3,)
    controller.set_filter("")
    assert controller.selected_keys() == (K3,)
    controller.set_filter("zzz-no-match")
    assert controller.selected_keys() == ()


def test_filter_emits_selection_only_for_membership_change(
    controller: AppController,
) -> None:
    controller.toggle_selected(K1)
    changes: list[str] = []
    controller.selection_changed.connect(lambda: changes.append("selection"))
    controller.filter_changed.connect(lambda text: changes.append(text))
    controller.set_filter("hair")
    assert changes == ["hair"]
    controller.set_filter("hair")
    assert changes == ["hair"]


def test_filtered_selection_replaces_explicit_global_selection(
    controller: AppController,
) -> None:
    controller.set_filter("0003")
    controller.select_all()
    assert controller.selected_keys() == DEMO_KEYS
    assert controller.scope_keys("all") == DEMO_KEYS
    controller.select_filtered()
    assert controller.selected_keys() == (K3,)
    assert controller.scope_keys("selected") == (K3,)
    controller.clear_selection()
    assert controller.selected_keys() == ()


def test_explicit_global_selection_still_navigates_hidden_keys(
    controller: AppController,
) -> None:
    controller.set_filter("zzz-no-match")
    controller.select_all()
    assert controller.can_navigate()
    assert controller.pos_label() == "1 / 4"
    controller.nav(1)
    assert controller.current_key == K2
    controller.clear_selection()
    assert not controller.can_navigate()
    assert controller.pos_label() == "未匹配 / 0"
    controller.nav(1)
    assert controller.current_key == K2


def test_folder_checkbox_changes_only_matching_keys(controller: AppController) -> None:
    controller.set_filter("1girl")
    assert controller.folder_keys(FOLDER) == (K3, K4)
    assert controller.folder_selection_state(FOLDER) == "none"
    controller.set_folder_selected(FOLDER, True)
    assert controller.selected_keys() == (K3,)
    assert controller.folder_selection_state(FOLDER) == "all"
    controller.select_all()
    controller.set_folder_selected(FOLDER, False)
    assert controller.selected_keys() == (K1, K2, K4)
    assert controller.folder_selection_state(FOLDER) == "none"
    controller.set_folder_selected(FOLDER, True)
    assert controller.selected_keys() == DEMO_KEYS
    assert controller.folder_keys(FOLDER) == (K3, K4)


def test_folder_checkbox_tristate_counts_matching_results(
    controller: AppController,
) -> None:
    controller.set_filter("png")
    controller.toggle_selected(K3)
    assert controller.folder_selection_state(FOLDER) == "some"
    controller.set_folder_selected(FOLDER, True)
    assert controller.folder_selection_state(FOLDER) == "all"
    controller.set_filter("0001")
    controller.select_all()
    assert controller.folder_selection_state(FOLDER) == "none"
    controller.set_folder_selected(FOLDER, False)
    assert controller.selected_keys() == DEMO_KEYS


def test_selected_batch_processes_only_filtered_selection(
    controller: AppController, demo_dataset: Path, qtbot
) -> None:
    controller.set_filter("0003")
    controller.select_filtered()
    with qtbot.waitSignal(controller.batch_finished, timeout=2000):
        controller.replace_all("1girl", "1woman", False, "selected")
    assert controller.record(K3).text == "1woman, yukata, fireworks"
    assert (demo_dataset / K3).with_suffix(".txt").read_text(
        encoding="utf-8"
    ) == "1woman, yukata, fireworks"
    assert controller.record(K1).text == DEMO_CAPTIONS[K1]
    assert controller.record(K2).text == DEMO_CAPTIONS[K2]
    assert controller.record(K4).text == ""
    assert (demo_dataset / K1).with_suffix(".txt").read_text(
        encoding="utf-8"
    ) == DEMO_CAPTIONS[K1]
