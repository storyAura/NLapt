"""Regression tests for issue #3.

Part A - the 设置 dialog's 翻译服务 provider dropdown:
  * selecting a provider shows ONLY that provider's key rows and fully hides
    the others (label + field), leaving no stray empty row/gap;
  * the stark bare HLine separator is gone (replaced by a themed divider);
  * switching to another provider and back fully restores the LLM section.

Part B - the program logo (nlapt_gui.theme.logo):
  * LogoWidget paints without error at the 22px bar size and 256px;
  * make_app_icon returns a non-null QIcon rendered from the same mark;
  * the title bar uses LogoWidget and updates it on theme change.

Runs offscreen via the shared conftest; no network, no real sleeps.
"""

from __future__ import annotations

from typing import Iterator

import pytest
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QFrame

from nlapt.app import NLaptApp
from nlapt.llm.web_translate import (
    PROVIDER_BAIDU,
    PROVIDER_CUSTOM,
    PROVIDER_DEEPL,
    PROVIDER_DEEPLX,
    PROVIDER_GOOGLE,
    PROVIDER_LLM,
    PROVIDER_LOCAL_MT,
)

from nlapt_gui.controller import AppController
from nlapt_gui.settings import UISettings
from nlapt_gui.theme.logo import ICON_SIZE, LOGO_SIZE, LogoWidget, make_app_icon
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES
from nlapt_gui.widgets.settings_dialog import SettingsDialog
from nlapt_gui.widgets.title_bar import LOGO_PX, TITLE_BAR_HEIGHT, TitleBar


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def sl_controller(qtbot) -> Iterator[AppController]:
    yield AppController(NLaptApp(), settings=UISettings())


@pytest.fixture()
def dialog(qtbot, sl_controller) -> SettingsDialog:
    dlg = SettingsDialog(sl_controller, api_types=("openai", "anthropic"))
    qtbot.addWidget(dlg)
    dlg.show()
    qtbot.waitExposed(dlg)
    return dlg


@pytest.fixture()
def manager(qtbot) -> ThemeManager:
    mgr = ThemeManager(persist=False)
    mgr.apply(DEFAULT_THEME)
    return mgr


def _select(dialog: SettingsDialog, provider_id: str) -> None:
    index = dialog.provider.findData(provider_id)
    assert index >= 0, f"provider {provider_id!r} not in dropdown"
    dialog.provider.setCurrentIndex(index)


def _baidu_widgets(dialog: SettingsDialog) -> tuple[object, ...]:
    return (
        dialog._baidu_appid_label,
        dialog.baidu_appid,
        dialog._baidu_key_label,
        dialog.baidu_key,
    )


def _deepl_widgets(dialog: SettingsDialog) -> tuple[object, ...]:
    return (dialog._deepl_key_label, dialog.deepl_key)


def _deeplx_widgets(dialog: SettingsDialog) -> tuple[object, ...]:
    return (
        dialog._deeplx_url_label,
        dialog.deeplx_url,
        dialog._deeplx_token_label,
        dialog.deeplx_token,
    )


def _custom_widgets(dialog: SettingsDialog) -> tuple[object, ...]:
    return (
        dialog._custom_url_label,
        dialog.custom_base_url,
        dialog._custom_key_label,
        dialog.custom_api_key,
        dialog._custom_model_label,
        dialog.custom_model,
    )


def _all_secret_widgets(dialog: SettingsDialog) -> tuple[object, ...]:
    return (
        _baidu_widgets(dialog)
        + _deepl_widgets(dialog)
        + _deeplx_widgets(dialog)
        + _custom_widgets(dialog)
    )


# --------------------------------------------------------------------------- #
# Part A - provider dropdown row visibility
# --------------------------------------------------------------------------- #
class TestProviderRows:
    def test_llm_hides_all_key_rows(self, dialog: SettingsDialog) -> None:
        _select(dialog, PROVIDER_LLM)
        for widget in _all_secret_widgets(dialog):
            assert not widget.isVisible()

    def test_google_hides_all_key_rows(self, dialog: SettingsDialog) -> None:
        _select(dialog, PROVIDER_GOOGLE)
        for widget in _all_secret_widgets(dialog):
            assert not widget.isVisible()

    def test_baidu_shows_only_baidu_rows(self, dialog: SettingsDialog) -> None:
        _select(dialog, PROVIDER_BAIDU)
        for widget in _baidu_widgets(dialog):
            assert widget.isVisible()
        for widget in (
            _deepl_widgets(dialog)
            + _deeplx_widgets(dialog)
            + _custom_widgets(dialog)
        ):
            assert not widget.isVisible()

    def test_deepl_shows_only_deepl_rows(self, dialog: SettingsDialog) -> None:
        _select(dialog, PROVIDER_DEEPL)
        for widget in _deepl_widgets(dialog):
            assert widget.isVisible()
        for widget in (
            _baidu_widgets(dialog)
            + _deeplx_widgets(dialog)
            + _custom_widgets(dialog)
        ):
            assert not widget.isVisible()

    def test_deeplx_shows_only_deeplx_rows(self, dialog: SettingsDialog) -> None:
        _select(dialog, PROVIDER_DEEPLX)
        for widget in _deeplx_widgets(dialog):
            assert widget.isVisible()
        for widget in (
            _baidu_widgets(dialog)
            + _deepl_widgets(dialog)
            + _custom_widgets(dialog)
        ):
            assert not widget.isVisible()

    def test_local_mt_hides_key_rows(self, dialog: SettingsDialog) -> None:
        _select(dialog, PROVIDER_LOCAL_MT)
        assert dialog.local_mt_tier.isVisible()
        for widget in (
            _baidu_widgets(dialog)
            + _deepl_widgets(dialog)
            + _deeplx_widgets(dialog)
            + _custom_widgets(dialog)
        ):
            assert not widget.isVisible()

    def test_local_mt_picker_lists_three_tiers(self, dialog: SettingsDialog) -> None:
        _select(dialog, PROVIDER_GOOGLE)
        assert dialog.local_mt_tier.isVisible()
        assert dialog.local_mt_tier.list.count() == 3
        assert dialog.local_mt_tier.findData("fast") == 0
        assert dialog.local_mt_tier.findData("balanced") == 1
        assert dialog.local_mt_tier.findData("quality") == 2

    def test_custom_shows_only_custom_rows(self, dialog: SettingsDialog) -> None:
        _select(dialog, PROVIDER_CUSTOM)
        for widget in _custom_widgets(dialog):
            assert widget.isVisible()
        for widget in (
            _baidu_widgets(dialog)
            + _deepl_widgets(dialog)
            + _deeplx_widgets(dialog)
        ):
            assert not widget.isVisible()

    def test_switch_away_and_back_restores_rows(self, dialog: SettingsDialog) -> None:
        # baidu -> deepl -> baidu must fully restore the baidu rows and hide
        # deepl's ("选择了其他选项后却没有重复回去").
        _select(dialog, PROVIDER_BAIDU)
        _select(dialog, PROVIDER_DEEPL)
        _select(dialog, PROVIDER_BAIDU)
        for widget in _baidu_widgets(dialog):
            assert widget.isVisible()
        for widget in _deepl_widgets(dialog):
            assert not widget.isVisible()

    def test_switch_to_llm_restores_cleanly(self, dialog: SettingsDialog) -> None:
        _select(dialog, PROVIDER_DEEPL)
        _select(dialog, PROVIDER_LLM)
        for widget in _baidu_widgets(dialog) + _deepl_widgets(dialog):
            assert not widget.isVisible()
        # The LLM profile form stays fully available on its own tab.
        dialog.tabs.setCurrentIndex(dialog.tabs.indexOf(dialog.api_type.parentWidget()))
        for widget in (dialog.api_type, dialog.base_url, dialog.text_model):
            assert widget.isVisible()

    def test_registration_note_updates_per_provider(
        self, dialog: SettingsDialog
    ) -> None:
        _select(dialog, PROVIDER_BAIDU)
        baidu_note = dialog.registration_note.text()
        _select(dialog, PROVIDER_DEEPL)
        deepl_note = dialog.registration_note.text()
        assert baidu_note and deepl_note
        assert baidu_note != deepl_note


class TestNoBareHLine:
    def test_divider_is_not_a_sunken_hline(self, dialog: SettingsDialog) -> None:
        assert dialog.divider.frameShape() == QFrame.Shape.NoFrame
        assert bool(dialog.divider.property("divider")) is True

    def test_no_child_frame_uses_hline(self, dialog: SettingsDialog) -> None:
        for frame in dialog.findChildren(QFrame):
            assert frame.frameShape() != QFrame.Shape.HLine


# --------------------------------------------------------------------------- #
# Part B - the program logo
# --------------------------------------------------------------------------- #
class TestLogoWidget:
    def test_paints_at_bar_and_icon_sizes(self, qtbot, manager) -> None:
        for size in (LOGO_SIZE, 256):
            widget = LogoWidget(manager.tokens, size)
            qtbot.addWidget(widget)
            pixmap = QPixmap(size, size)
            widget.render(pixmap)
            assert not pixmap.isNull()

    def test_default_size_is_bar_size(self, qtbot, manager) -> None:
        widget = LogoWidget(manager.tokens)
        qtbot.addWidget(widget)
        assert widget.width() == LOGO_SIZE == LOGO_PX

    def test_set_tokens_repaints_without_error(self, qtbot, manager) -> None:
        widget = LogoWidget(manager.tokens)
        qtbot.addWidget(widget)
        widget.set_tokens(THEMES["墨黑"])
        pixmap = QPixmap(LOGO_SIZE, LOGO_SIZE)
        widget.render(pixmap)
        assert not pixmap.isNull()


class TestAppIcon:
    def test_make_app_icon_returns_non_null(self, manager) -> None:
        icon = make_app_icon(manager.tokens)
        assert isinstance(icon, QIcon)
        assert not icon.isNull()

    def test_make_app_icon_pixmap_has_size(self, manager) -> None:
        icon = make_app_icon(manager.tokens, ICON_SIZE)
        pixmap = icon.pixmap(ICON_SIZE, ICON_SIZE)
        assert not pixmap.isNull()
        assert pixmap.width() == ICON_SIZE


class TestTitleBarUsesLogo:
    def test_title_bar_logo_is_logo_widget(self, qtbot, sl_controller, manager) -> None:
        bar = TitleBar(sl_controller, manager)
        bar.resize(1360, TITLE_BAR_HEIGHT)
        qtbot.addWidget(bar)
        assert isinstance(bar.logo, LogoWidget)
        assert bar.logo.width() == LOGO_PX

    def test_theme_change_updates_logo_tokens(
        self, qtbot, sl_controller, manager
    ) -> None:
        bar = TitleBar(sl_controller, manager)
        qtbot.addWidget(bar)
        # Applying a new theme must repaint the logo via apply_tokens ->
        # LogoWidget.set_tokens (no exception, tokens propagated).
        manager.apply("墨黑")
        assert bar.logo._tokens is THEMES["墨黑"]
