"""Tests for nlapt_gui.widgets.cha_tab (CHA标注 settings page)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy

from nlapt.core.config import LLMProfile, ModelRef

from nlapt_gui.cha_config import API_MODE_OWN, CHASettings
from nlapt_gui.model_targets import ModelChoice
from nlapt_gui.widgets.cha_tab import (
    GROUP_BASE_ENDPOINT,
    GROUP_HEADER_FMT,
    GROUP_STALE,
    MODEL_COMBO_MIN_W,
    PLACEHOLDER_MODEL,
    TOAST_NEED_BASE_URL,
    TOAST_STALE_REF,
    CHATab,
)

MAIN = LLMProfile(
    name="default",
    api_type="openai",
    base_url="http://main.local",
    api_key="sk-main",
    text_model="text-main",
    vision_model="vision-main",
    models=("text-main", "vision-main", "off-model"),
    enabled_models=("text-main", "vision-main"),
)
BETA = LLMProfile(
    name="beta",
    api_type="ollama",
    base_url="http://beta",
    models=("llava",),
    enabled_models=("llava",),
)


def _choice(profile: str, model: str, vision: bool = False) -> ModelChoice:
    return ModelChoice(ModelRef(profile, model), f"{profile} · {model}", vision)


GROUPS = (
    ("default", (_choice("default", "text-main"), _choice("default", "vision-main", True))),
    ("beta", (_choice("beta", "llava", True),)),
)


def _lookup(ref: ModelRef) -> LLMProfile | None:
    if ref == ModelRef("beta", "llava"):
        return LLMProfile(
            name="beta", api_type="ollama", base_url="http://beta", vision_model="llava"
        )
    if ref.profile == "default" and ref.model in MAIN.enabled_models:
        return LLMProfile(
            name="default", api_type="openai", base_url=MAIN.base_url, vision_model=ref.model
        )
    return None


def _tab(
    qtbot,
    *,
    profile: LLMProfile | None = MAIN,
    groups=GROUPS,
) -> CHATab:
    tab = CHATab(
        main_profile_provider=lambda: profile,
        pool_provider=lambda: groups,
        profile_lookup=_lookup,
    )
    qtbot.addWidget(tab)
    return tab


def _items(tab: CHATab, index: int = 0) -> list[str]:
    combo = tab.card_combos[index]
    return [combo.itemText(i) for i in range(combo.count())]


def _select_ref(tab: CHATab, index: int, ref: ModelRef) -> None:
    combo = tab.card_combos[index]
    position = next(i for i in range(combo.count()) if combo.itemData(i) == ref)
    combo.setCurrentIndex(position)


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

    def test_current_settings_mixes_pool_picks_and_typed_ids(self, qtbot) -> None:
        tab = _tab(qtbot)
        tab.sync_check.setChecked(False)
        tab.base_url.setText("http://cha.local")
        tab.api_key.setText("sk-cha")
        tab.card_combos[0].setCurrentText("alpha")  # typed -> bare ref
        _select_ref(tab, 2, ModelRef("beta", "llava"))  # picked -> pool ref
        tab.batch_combo.setCurrentText("batch-x")
        settings = tab.current_settings()
        assert settings.api_mode == API_MODE_OWN
        assert settings.base_url == "http://cha.local"
        assert settings.api_key == "sk-cha"
        assert settings.card_models == (
            ModelRef("", "alpha"),
            ModelRef(),
            ModelRef("beta", "llava"),
        )
        assert settings.batch_model == ModelRef("", "batch-x")

    def test_prefill_selects_pool_rows_and_types_bare_ids(self, qtbot) -> None:
        tab = _tab(qtbot)
        tab.prefill(
            CHASettings(
                api_mode=API_MODE_OWN,
                api_type="ollama",
                base_url="http://own",
                api_key="k",
                card_models=(ModelRef("beta", "llava"), "b", ModelRef()),
                batch_model=ModelRef("default", "vision-main"),
            )
        )
        assert not tab.sync_check.isChecked()
        assert tab.api_type.currentText() == "ollama"
        assert tab.base_url.text() == "http://own"
        assert tab.card_combos[0].currentText() == "beta · llava"
        assert tab.card_combos[1].currentText() == "b"
        assert tab.card_combos[2].currentText() == ""
        assert tab.batch_combo.currentText() == "default · vision-main"
        assert not tab.base_url.isHidden()
        # Round-trips through current_settings unchanged.
        settings = tab.current_settings()
        assert settings.card_models[0] == ModelRef("beta", "llava")
        assert settings.card_models[1] == ModelRef("", "b")
        assert settings.batch_model == ModelRef("default", "vision-main")

    def test_placeholder_on_model_fields(self, qtbot) -> None:
        tab = _tab(qtbot)
        edit = tab.card_combos[0].lineEdit()
        assert edit is not None
        assert edit.placeholderText() == PLACEHOLDER_MODEL


class TestPoolPickers:
    def test_combos_fill_the_row(self, qtbot) -> None:
        tab = _tab(qtbot)
        for combo in (*tab.card_combos, tab.batch_combo):
            assert combo.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
            assert combo.minimumWidth() == MODEL_COMBO_MIN_W

    def test_lists_whole_pool_grouped_by_provider(self, qtbot) -> None:
        tab = _tab(qtbot)
        assert _items(tab) == [
            "",
            GROUP_HEADER_FMT.format(name="default"),
            "default · text-main",
            "default · vision-main",
            GROUP_HEADER_FMT.format(name="beta"),
            "beta · llava",
        ]
        assert _items(tab, 2) == _items(tab)
        assert tab.batch_combo.count() == tab.card_combos[0].count()
        # Header rows cannot be selected.
        header = tab.card_combos[0].model().item(1)
        assert not header.flags() & Qt.ItemFlag.ItemIsSelectable
        assert not header.flags() & Qt.ItemFlag.ItemIsEnabled

    def test_pool_listed_regardless_of_sync_mode(self, qtbot) -> None:
        tab = _tab(qtbot)
        tab.sync_check.setChecked(False)
        assert "beta · llava" in _items(tab)

    def test_refresh_keeps_picks_and_typed_text(self, qtbot) -> None:
        holder = {"groups": GROUPS}
        tab = CHATab(
            main_profile_provider=lambda: MAIN,
            pool_provider=lambda: holder["groups"],
            profile_lookup=_lookup,
        )
        qtbot.addWidget(tab)
        tab.card_combos[0].setCurrentText("keep-me")
        _select_ref(tab, 1, ModelRef("beta", "llava"))
        holder["groups"] = (GROUPS[1],)
        tab.refresh_pool_models()
        assert tab.card_combos[0].currentText() == "keep-me"
        assert tab.current_settings().card_models[1] == ModelRef("beta", "llava")
        assert GROUP_HEADER_FMT.format(name="default") not in _items(tab)

    def test_stale_persisted_ref_stays_selected_and_marked(self, qtbot) -> None:
        tab = _tab(qtbot)
        tab.prefill(CHASettings(card_models=(ModelRef("gone", "x"), "", "")))
        assert tab.card_combos[0].currentText() == "gone · x"
        assert GROUP_HEADER_FMT.format(name=GROUP_STALE) in _items(tab)
        assert tab.current_settings().card_models[0] == ModelRef("gone", "x")


class TestFetchModels:
    def test_fetched_ids_listed_under_base_endpoint_group(self, qtbot, monkeypatch) -> None:
        tab = _tab(qtbot)
        tab.card_combos[0].setCurrentText("keep-me")
        monkeypatch.setattr(
            "nlapt_gui.widgets.cha_tab.list_models",
            lambda _profile: ("alpha", "vision-main"),
        )
        tab.fetch_models()
        qtbot.waitUntil(lambda: "alpha" in _items(tab), timeout=2000)
        items = _items(tab)
        header = items.index(GROUP_HEADER_FMT.format(name=GROUP_BASE_ENDPOINT))
        assert items[header + 1 :] == ["alpha", "vision-main"]
        assert "beta · llava" in items  # pool groups stay
        assert tab.card_combos[0].currentText() == "keep-me"
        combo = tab.card_combos[1]
        combo.setCurrentIndex(header + 1)
        assert tab.current_settings().card_models[1] == ModelRef("", "alpha")
        tab.refresh_pool_models()
        assert "alpha" in _items(tab)

    def test_fetch_without_base_url_toasts(self, qtbot) -> None:
        empty = LLMProfile(name="default", api_type="openai", base_url="")
        tab = _tab(qtbot, profile=empty)
        toasts: list[tuple[str, str]] = []
        tab.toast_requested.connect(lambda text, kind: toasts.append((text, kind)))
        tab.fetch_models()
        assert (TOAST_NEED_BASE_URL, "warn") in toasts


class TestProbe:
    def test_probe_uses_pool_ref_api(self, qtbot) -> None:
        tab = _tab(qtbot)
        _select_ref(tab, 0, ModelRef("beta", "llava"))
        profile = tab._probe_profile()
        assert profile is not None and profile.base_url == "http://beta"
        assert profile.vision_model == "llava"

    def test_probe_typed_id_uses_base_endpoint(self, qtbot) -> None:
        tab = _tab(qtbot)
        tab.card_combos[1].setCurrentText("typed")
        profile = tab._probe_profile()
        assert profile is not None and profile.base_url == "http://main.local"
        assert profile.vision_model == "typed"

    def test_probe_stale_ref_toasts(self, qtbot) -> None:
        tab = _tab(qtbot)
        tab.prefill(CHASettings(card_models=(ModelRef("gone", "x"), "", "")))
        toasts: list[tuple[str, str]] = []
        tab.toast_requested.connect(lambda text, kind: toasts.append((text, kind)))
        assert tab._probe_profile() is None
        assert (TOAST_STALE_REF.format(label="gone · x"), "warn") in toasts
