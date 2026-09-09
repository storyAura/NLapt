"""Tests for the grouped 工具 popup (incl. the 当前模型 quick switch)."""

from __future__ import annotations

from PySide6.QtWidgets import QMenu

from nlapt.core.config import ROLE_TEXT, ROLE_VISION, ModelRef

from nlapt_gui.model_targets import LABEL_STALE, LABEL_UNSET, ModelChoice
from nlapt_gui.theme.tokens import THEMES
from nlapt_gui.widgets.tools_menu import (
    ACTION_FIND_DUPLICATES,
    ACTION_FLATTEN_ALPHA,
    ACTION_INFER_CHA,
    ACTION_INFER_COMPARE,
    ACTION_MODEL_TEXT,
    ACTION_MODEL_VISION,
    ACTION_UNDO_IMAGE_OP,
    LABEL_INFER_CHA,
    LABEL_INFER_COMPARE,
    LABEL_MODEL_TEXT,
    LABEL_MODEL_VISION,
    MENU_CLEAR_TARGET,
    MENU_POOL_EMPTY,
    MENU_VISION_TAG,
    ToolsMenuPopup,
)

CHOICES = (
    ModelChoice(ModelRef("alpha", "gpt-plain"), "alpha · gpt-plain", False),
    ModelChoice(ModelRef("alpha", "gpt-4o"), "alpha · gpt-4o", True),
    ModelChoice(ModelRef("beta", "llava"), "beta · llava", True),
)


def _popup(qtbot, **kwargs) -> ToolsMenuPopup:
    popup = ToolsMenuPopup(THEMES["雾灰"], **kwargs)
    qtbot.addWidget(popup)
    return popup


def test_popup_emits_action(qtbot) -> None:
    popup = _popup(qtbot)
    with qtbot.waitSignal(popup.action_triggered, timeout=1000) as blocker:
        popup._buttons[ACTION_INFER_CHA].click()
    assert blocker.args == [ACTION_INFER_CHA]
    assert LABEL_INFER_CHA in popup._buttons[ACTION_INFER_CHA].text()


def test_text_drawer_is_not_in_the_popup(qtbot) -> None:
    """修改工具 has its own rail button; the popup must not duplicate it."""
    popup = _popup(qtbot)
    assert "text_tools" not in popup._buttons
    assert all("修改工具" not in button.text() for button in popup._buttons.values())


def test_compare_action_listed_in_annotate_group(qtbot) -> None:
    popup = _popup(qtbot)
    assert popup._buttons[ACTION_INFER_COMPARE].text() == LABEL_INFER_COMPARE
    assert popup._buttons[ACTION_INFER_COMPARE].isEnabled()
    with qtbot.waitSignal(popup.action_triggered, timeout=1000) as blocker:
        popup._buttons[ACTION_INFER_COMPARE].click()
    assert blocker.args == [ACTION_INFER_COMPARE]


def test_busy_disables_image_actions_and_model_switch(qtbot) -> None:
    popup = _popup(qtbot, busy=True)
    assert not popup._buttons[ACTION_FLATTEN_ALPHA].isEnabled()
    assert not popup._buttons[ACTION_UNDO_IMAGE_OP].isEnabled()
    assert not popup._buttons[ACTION_FIND_DUPLICATES].isEnabled()
    assert not popup._buttons[ACTION_MODEL_TEXT].isEnabled()
    assert not popup._buttons[ACTION_MODEL_VISION].isEnabled()
    assert popup._buttons[ACTION_INFER_CHA].isEnabled()
    popup.set_busy(False)
    assert popup._buttons[ACTION_FLATTEN_ALPHA].isEnabled()
    assert popup._buttons[ACTION_MODEL_TEXT].isEnabled()


class TestModelGroup:
    def test_buttons_show_current_targets(self, qtbot) -> None:
        popup = _popup(
            qtbot,
            choices=CHOICES,
            text_target=ModelRef("alpha", "gpt-plain"),
            vision_target=ModelRef("beta", "llava"),
        )
        assert popup._buttons[ACTION_MODEL_TEXT].text() == LABEL_MODEL_TEXT.format(
            model="gpt-plain"
        )
        assert popup._buttons[ACTION_MODEL_VISION].text() == LABEL_MODEL_VISION.format(
            model="llava"
        )

    def test_unset_and_stale_targets(self, qtbot) -> None:
        popup = _popup(qtbot, choices=CHOICES, vision_target=ModelRef("beta", "gone"))
        assert popup.target_display(ROLE_TEXT) == LABEL_UNSET
        assert popup.target_display(ROLE_VISION) == LABEL_STALE

    def test_menu_lists_pool_with_current_checked(self, qtbot) -> None:
        popup = _popup(qtbot, choices=CHOICES, vision_target=ModelRef("alpha", "gpt-4o"))
        menu = popup.build_model_menu(ROLE_VISION)
        assert isinstance(menu, QMenu)
        labels = [a.text() for a in menu.actions() if a.text()]
        assert labels == [
            "alpha · gpt-plain",
            f"alpha · gpt-4o{MENU_VISION_TAG}",
            f"beta · llava{MENU_VISION_TAG}",
            MENU_CLEAR_TARGET,
        ]
        checked = [a.text() for a in menu.actions() if a.isChecked()]
        assert checked == [f"alpha · gpt-4o{MENU_VISION_TAG}"]

    def test_menu_without_pool_or_target(self, qtbot) -> None:
        popup = _popup(qtbot)
        menu = popup.build_model_menu(ROLE_TEXT)
        actions = [a for a in menu.actions() if a.text()]
        assert [a.text() for a in actions] == [MENU_POOL_EMPTY]
        assert not actions[0].isEnabled()

    def test_picking_emits_switch_and_closes(self, qtbot) -> None:
        popup = _popup(qtbot, choices=CHOICES, text_target=ModelRef("alpha", "gpt-plain"))
        popup.show()
        menu = popup.build_model_menu(ROLE_TEXT)
        target = next(a for a in menu.actions() if a.text().startswith("beta"))
        with qtbot.waitSignal(popup.model_switch_requested, timeout=1000) as blocker:
            target.trigger()
        assert blocker.args == [ROLE_TEXT, ModelRef("beta", "llava")]
        assert not popup.isVisible()

    def test_clear_entry_emits_unset_ref(self, qtbot) -> None:
        popup = _popup(qtbot, choices=CHOICES, text_target=ModelRef("alpha", "gpt-plain"))
        menu = popup.build_model_menu(ROLE_TEXT)
        clear = next(a for a in menu.actions() if a.text() == MENU_CLEAR_TARGET)
        with qtbot.waitSignal(popup.model_switch_requested, timeout=1000) as blocker:
            clear.trigger()
        assert blocker.args == [ROLE_TEXT, ModelRef()]

    def test_model_button_opens_menu(self, qtbot, monkeypatch) -> None:
        popup = _popup(qtbot, choices=CHOICES)
        opened: list[QMenu] = []
        monkeypatch.setattr(QMenu, "popup", lambda menu, *_a, **_k: opened.append(menu))
        popup._buttons[ACTION_MODEL_VISION].click()
        assert len(opened) == 1
        assert popup._model_menu is opened[0]
