"""Round-3 integration: the program logo is wired as the window/app icon.

Covers the integrator wiring added in ``MainWindow`` (window icon set on
construction and refreshed on theme changes) and the ``QApplication`` icon set
in ``nlapt_gui.__main__``. Offscreen, no real waits.
"""

from __future__ import annotations

import pytest

from nlapt_gui.theme.logo import ICON_SIZE, make_app_icon
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import THEMES
from nlapt_gui.widgets.main_window import MainWindow


@pytest.fixture()
def manager(qtbot) -> ThemeManager:
    mgr = ThemeManager(persist=False)
    mgr.apply("雾灰")
    return mgr


@pytest.fixture()
def window(qtbot, controller, manager) -> MainWindow:
    win = MainWindow(controller, manager)
    qtbot.addWidget(win)
    return win


class TestMakeAppIcon:
    # qtbot ensures a QApplication exists (QPixmap requires QGuiApplication).
    def test_returns_non_null_icon(self, qtbot) -> None:
        icon = make_app_icon(THEMES["雾灰"])
        assert not icon.isNull()

    def test_renders_requested_size(self, qtbot) -> None:
        icon = make_app_icon(THEMES["雾灰"])
        sizes = icon.availableSizes()
        assert sizes, "icon should expose at least one rendered size"
        assert sizes[0].width() == ICON_SIZE
        assert sizes[0].height() == ICON_SIZE

    def test_distinct_accents_produce_distinct_icons(self, qtbot) -> None:
        light = make_app_icon(THEMES["明亮"])
        dark = make_app_icon(THEMES["墨黑"])
        assert light.cacheKey() != dark.cacheKey()


class TestWindowIconWiring:
    def test_window_has_icon_on_construction(self, window: MainWindow) -> None:
        assert not window.windowIcon().isNull()

    def test_window_icon_refreshes_on_theme_change(
        self, window: MainWindow, manager: ThemeManager
    ) -> None:
        before = window.windowIcon().cacheKey()
        manager.apply("墨黑")
        after = window.windowIcon().cacheKey()
        # A fresh icon is painted for the new tokens (different pixmap object).
        assert after != before
        assert not window.windowIcon().isNull()


class TestAppIconWiring:
    def test_main_sets_application_icon(self, qtbot, monkeypatch) -> None:
        """``main()`` sets a non-null QApplication window icon (taskbar).

        A second ``QApplication`` cannot exist alongside the pytest-qt
        singleton, so ``QApplication(...)`` is proxied to the live instance
        (class attributes such as ``.instance`` still resolve for pytest-qt),
        ``exec`` is stubbed to skip the event loop, and ``MainWindow`` is
        stubbed so no real window is shown.
        """
        import PySide6.QtWidgets as qtw
        from PySide6.QtGui import QIcon

        import nlapt_gui.widgets.main_window as mw_module

        real_cls = qtw.QApplication
        real_app = real_cls.instance()
        assert real_app is not None
        real_app.setWindowIcon(QIcon())  # clear so we prove main() sets it
        assert real_app.windowIcon().isNull()

        class _AppProxy:
            """Callable returns the live app; attributes defer to the class."""

            def __call__(self, *args: object, **kwargs: object) -> object:
                return real_app

            def __getattr__(self, name: str) -> object:
                return getattr(real_cls, name)

        class _StubWindow:
            def __init__(self, controller: object, theme_manager: object) -> None:
                self._controller = controller

            def show(self) -> None:  # noqa: D401 - test stub
                pass

        monkeypatch.setattr(real_cls, "exec", lambda self: 0)
        monkeypatch.setattr(qtw, "QApplication", _AppProxy())
        monkeypatch.setattr(mw_module, "MainWindow", _StubWindow)

        from nlapt_gui import __main__ as entry

        rc = entry.main([])
        assert rc == 0
        assert not real_app.windowIcon().isNull()
