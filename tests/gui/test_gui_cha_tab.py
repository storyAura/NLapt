"""Tests for nlapt_gui.widgets.cha_tab (CHA标注 settings page)."""

from __future__ import annotations

from nlapt.core.config import LLMProfile

from nlapt_gui.cha_config import API_MODE_OWN, CHASettings
from nlapt_gui.widgets.cha_tab import (
    PLACEHOLDER_MODEL,
    TOAST_NEED_BASE_URL,
    CHATab,
)

MAIN = LLMProfile(
    name="default",
    api_type="openai",
    base_url="http://main.local",
    api_key="sk-main",
    text_model="text-main",
    vision_model="vision-main",
)


def _tab(qtbot, *, profile: LLMProfile = MAIN) -> CHATab:
    tab = CHATab(main_profile_provider=lambda: profile)
    qtbot.addWidget(tab)
    return tab


class TestForm:
    def test_sync_hides_api_rows(self, qtbot) -> None:
        tab = _tab(qtbot)
        assert tab.sync_check.isChecked()
        assert tab.api_type.isHidden()
        assert tab.base_url.isHidden()
        assert tab.api_key.isHidden()
        tab.sync_check.setChecked(False)
        assert not tab.api_type.isHidden()
        assert not tab.base_url.isHidden()
        assert not tab.api_key.isHidden()

    def test_current_settings_from_fields(self, qtbot) -> None:
        tab = _tab(qtbot)
        tab.sync_check.setChecked(False)
        tab.base_url.setText("http://cha.local")
        tab.api_key.setText("sk-cha")
        tab.card_combos[0].setCurrentText("alpha")
        tab.card_combos[2].setCurrentText("gamma")
        tab.batch_combo.setCurrentText("batch-x")
        settings = tab.current_settings()
        assert settings.api_mode == API_MODE_OWN
        assert settings.base_url == "http://cha.local"
        assert settings.api_key == "sk-cha"
        assert settings.card_models == ("alpha", "", "gamma")
        assert settings.batch_model == "batch-x"

    def test_prefill_own_api(self, qtbot) -> None:
        tab = _tab(qtbot)
        tab.prefill(
            CHASettings(
                api_mode=API_MODE_OWN,
                api_type="ollama",
                base_url="http://own",
                api_key="k",
                card_models=("a", "b", "c"),
                batch_model="d",
            )
        )
        assert not tab.sync_check.isChecked()
        assert tab.api_type.currentText() == "ollama"
        assert tab.base_url.text() == "http://own"
        assert tab.card_combos[1].currentText() == "b"
        assert tab.batch_combo.currentText() == "d"
        assert not tab.base_url.isHidden()

    def test_placeholder_on_model_fields(self, qtbot) -> None:
        tab = _tab(qtbot)
        edit = tab.card_combos[0].lineEdit()
        assert edit is not None
        assert edit.placeholderText() == PLACEHOLDER_MODEL


class TestFetchModels:
    def test_fills_combos_and_keeps_typed_text(self, qtbot, monkeypatch) -> None:
        tab = _tab(qtbot)
        tab.card_combos[0].setCurrentText("keep-me")
        monkeypatch.setattr(
            "nlapt_gui.widgets.cha_tab.list_models",
            lambda _profile: ("alpha", "beta"),
        )
        tab.fetch_models()
        qtbot.waitUntil(lambda: tab.card_combos[0].count() > 1, timeout=2000)
        names = [tab.card_combos[0].itemText(i) for i in range(tab.card_combos[0].count())]
        assert "alpha" in names
        assert "beta" in names
        assert tab.card_combos[0].currentText() == "keep-me"
        assert tab.batch_combo.count() == tab.card_combos[0].count()

    def test_fetch_without_base_url_toasts(self, qtbot) -> None:
        empty = LLMProfile(name="default", api_type="openai", base_url="")
        tab = _tab(qtbot, profile=empty)
        toasts: list[tuple[str, str]] = []
        tab.toast_requested.connect(lambda text, kind: toasts.append((text, kind)))
        tab.fetch_models()
        assert (TOAST_NEED_BASE_URL, "warn") in toasts
