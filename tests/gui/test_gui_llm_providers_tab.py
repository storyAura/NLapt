"""Tests for nlapt_gui.widgets.llm_providers_tab (multi-API profiles + pool)."""

from __future__ import annotations

from PySide6.QtCore import Qt

from nlapt.core.config import ROLE_TEXT, ROLE_VISION, AppConfig, LLMProfile, ModelRef
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.widgets.llm_providers_tab import (
    LABEL_MODELS,
    LABEL_MODELS_FILTERED,
    NEW_PROFILE_NAME,
    TOAST_DUPLICATE_NAME,
    TOAST_NEED_NAME,
    TOAST_NEED_PROFILE_URL,
    TOAST_NEED_TEXT_MODEL,
    VISION_TAG,
    LLMProvidersTab,
)

API_TYPE = "gui-providers-tab-mock"
register_client(API_TYPE, lambda profile: MockLLMClient(["ok"]))


def _config() -> AppConfig:
    alpha = LLMProfile(
        name="alpha",
        api_type=API_TYPE,
        base_url="http://a",
        api_key="sk-a",
        models=("a-text", "gpt-4o", "a-off"),
        enabled_models=("a-text", "gpt-4o"),
    )
    beta = LLMProfile(
        name="beta",
        api_type="openai",
        base_url="http://b",
        models=("b-vl",),
        enabled_models=("b-vl",),
    )
    return AppConfig(
        profiles=(alpha, beta),
        text_target=ModelRef("alpha", "a-text"),
        vision_target=ModelRef("beta", "b-vl"),
    )


def _tab(qtbot, config: AppConfig | None = None) -> LLMProvidersTab:
    tab = LLMProvidersTab(config, api_types=(API_TYPE, "openai"))
    qtbot.addWidget(tab)
    return tab


def _checked(tab: LLMProvidersTab) -> list[str]:
    return [
        str(tab.model_list.item(i).data(Qt.ItemDataRole.UserRole))
        for i in range(tab.model_list.count())
        if tab.model_list.item(i).checkState() == Qt.CheckState.Checked
    ]


class TestPrefill:
    def test_lists_profiles_and_selects_text_target_profile(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        assert [tab.profile_list.item(i).text() for i in range(2)] == ["alpha", "beta"]
        assert tab.profile_list.currentRow() == 0
        assert tab.base_url.text() == "http://a"
        assert tab.api_key.text() == "sk-a"
        assert tab.api_type.currentText() == API_TYPE
        assert tab.model_list.count() == 3
        assert _checked(tab) == ["a-text", "gpt-4o"]
        assert tab.model_list.item(1).text() == f"gpt-4o{VISION_TAG}"

    def test_target_combos_cover_the_pool(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        refs = [tab.text_combo.itemData(i) for i in range(tab.text_combo.count())]
        assert refs == [
            ModelRef(),
            ModelRef("alpha", "a-text"),
            ModelRef("alpha", "gpt-4o"),
            ModelRef("beta", "b-vl"),
        ]
        assert tab.text_combo.currentData() == ModelRef("alpha", "a-text")
        assert tab.vision_combo.currentData() == ModelRef("beta", "b-vl")

    def test_empty_config_shows_hint_only(self, qtbot) -> None:
        tab = _tab(qtbot, None)
        assert tab.profile_list.count() == 0
        assert tab.current_profile() is None
        assert not tab.remove_button.isEnabled()
        assert tab.text_combo.count() == 1  # 未设置 only


class TestProfileList:
    def test_add_and_remove_profiles(self, qtbot) -> None:
        tab = _tab(qtbot, None)
        tab.add_profile()
        assert tab.profile_list.count() == 1
        assert tab.current_profile().name == NEW_PROFILE_NAME.format(n=1)
        assert tab.current_profile().api_type == API_TYPE
        tab.add_profile()
        assert tab.current_profile().name == NEW_PROFILE_NAME.format(n=2)
        tab.remove_profile()
        assert tab.profile_list.count() == 1
        assert tab.current_profile().name == NEW_PROFILE_NAME.format(n=1)

    def test_removing_the_targeted_profile_clears_its_targets(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.select_profile("beta")
        tab.remove_profile()
        assert [p.name for p in tab.profiles()] == ["alpha"]
        assert tab.text_target() == ModelRef("alpha", "a-text")
        assert not tab.vision_target().is_set()
        assert tab.vision_combo.currentIndex() == 0

    def test_switching_rows_keeps_edits(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.base_url.setText("http://a-new")
        tab.api_key.setText("sk-new")
        tab.select_profile("beta")
        assert tab.base_url.text() == "http://b"
        tab.select_profile("alpha")
        assert tab.base_url.text() == "http://a-new"
        assert tab.profiles()[0].api_key == "sk-new"

    def test_rename_follows_targets(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.name_edit.setText("alpha2")
        assert tab.profiles()[0].name == "alpha2"
        assert tab.profile_list.item(0).text() == "alpha2"
        assert tab.text_target() == ModelRef("alpha2", "a-text")
        assert tab.text_combo.currentData() == ModelRef("alpha2", "a-text")


class TestModelSwitches:
    def test_toggle_updates_enabled_models_and_pool(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.model_list.item(2).setCheckState(Qt.CheckState.Checked)
        assert tab.profiles()[0].enabled_models == ("a-text", "gpt-4o", "a-off")
        assert tab.text_combo.count() == 5

    def test_disabling_the_current_target_resets_it(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.model_list.item(0).setCheckState(Qt.CheckState.Unchecked)
        assert tab.profiles()[0].enabled_models == ("gpt-4o",)
        assert not tab.text_target().is_set()
        assert tab.text_combo.currentIndex() == 0
        assert tab.vision_target() == ModelRef("beta", "b-vl")

    def test_select_all_and_none(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.disable_all_button.click()
        assert tab.profiles()[0].enabled_models == ()
        assert _checked(tab) == []
        tab.enable_all_button.click()
        assert tab.profiles()[0].enabled_models == ("a-text", "gpt-4o", "a-off")

    def test_manual_add_enables_and_dedupes(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.new_model_edit.setText("  hand-typed ")
        tab.add_model()
        profile = tab.profiles()[0]
        assert profile.models[-1] == "hand-typed"
        assert profile.enabled_models[-1] == "hand-typed"
        assert tab.new_model_edit.text() == ""
        tab.new_model_edit.setText("hand-typed")
        tab.add_model()
        assert tab.profiles()[0].models.count("hand-typed") == 1

    def test_combo_pick_sets_targets(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.vision_combo.setCurrentIndex(2)  # alpha · gpt-4o
        assert tab.vision_target() == ModelRef("alpha", "gpt-4o")
        tab.text_combo.setCurrentIndex(0)
        assert not tab.text_target().is_set()


def _visible(tab: LLMProvidersTab) -> list[str]:
    return [
        str(tab.model_list.item(i).data(Qt.ItemDataRole.UserRole))
        for i in range(tab.model_list.count())
        if not tab.model_list.item(i).isHidden()
    ]


class TestModelFilter:
    def test_filter_hides_non_matching_case_insensitively(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        assert tab.models_label.text() == LABEL_MODELS
        tab.model_filter.setText("GPT")
        assert _visible(tab) == ["gpt-4o"]
        assert tab.models_label.text() == LABEL_MODELS_FILTERED.format(n=1, total=3)
        tab.model_filter.setText("a-")
        assert _visible(tab) == ["a-text", "a-off"]
        tab.model_filter.clear()
        assert _visible(tab) == ["a-text", "gpt-4o", "a-off"]
        assert tab.models_label.text() == LABEL_MODELS

    def test_filter_survives_profile_switch(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.model_filter.setText("vl")
        tab.select_profile("beta")
        assert _visible(tab) == ["b-vl"]
        tab.select_profile("alpha")
        assert _visible(tab) == []
        assert tab.models_label.text() == LABEL_MODELS_FILTERED.format(n=0, total=3)

    def test_select_all_and_none_scoped_to_visible(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.model_filter.setText("off")
        tab.enable_all_button.click()
        assert tab.profiles()[0].enabled_models == ("a-text", "gpt-4o", "a-off")
        tab.model_filter.setText("gpt")
        tab.disable_all_button.click()
        assert tab.profiles()[0].enabled_models == ("a-text", "a-off")
        # Without a filter the buttons still cover the whole catalog.
        tab.model_filter.clear()
        tab.disable_all_button.click()
        assert tab.profiles()[0].enabled_models == ()
        tab.enable_all_button.click()
        assert tab.profiles()[0].enabled_models == ("a-text", "gpt-4o", "a-off")


class TestBuildAndValidate:
    def test_build_config_applies_tab_state(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.concurrency_spin.setValue(7)
        tab.set_target(ROLE_TEXT, ModelRef("alpha", "gpt-4o"))
        base = AppConfig(active_profile="stale", snapshot_retention=3)
        built = tab.build_config(base)
        assert [p.name for p in built.profiles] == ["alpha", "beta"]
        assert built.text_target == ModelRef("alpha", "gpt-4o")
        assert built.vision_target == ModelRef("beta", "b-vl")
        assert built.active_profile == ""
        assert built.request.concurrency == 7
        assert built.snapshot_retention == 3
        vision = tab.vision_profile()
        assert vision is not None and vision.base_url == "http://b"
        assert vision.vision_model == "b-vl"

    def test_set_target_ignores_unknown_refs(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        tab.set_target(ROLE_VISION, ModelRef("ghost", "m"))
        assert not tab.vision_target().is_set()

    def test_validate_messages(self, qtbot) -> None:
        tab = _tab(qtbot, _config())
        assert tab.validate(require_text=True) is None
        tab.select_profile("beta")
        tab.base_url.setText("")
        assert tab.validate(require_text=False) == TOAST_NEED_PROFILE_URL.format(name="beta")
        tab.base_url.setText("http://b")
        tab.name_edit.setText("alpha")
        assert tab.validate(require_text=False) == TOAST_DUPLICATE_NAME.format(name="alpha")
        tab.name_edit.setText("")
        assert tab.validate(require_text=False) == TOAST_NEED_NAME
        tab.name_edit.setText("beta")
        tab.set_target(ROLE_TEXT, ModelRef())
        assert tab.validate(require_text=False) is None
        assert tab.validate(require_text=True) == TOAST_NEED_TEXT_MODEL
