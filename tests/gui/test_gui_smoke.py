"""Offscreen end-to-end smoke: MainWindow + real controller on the demo dataset.

One primary interaction per panel: pick a file (left), switch editor modes,
edit a chip (middle), run 全部替换 scope=current (tools overlay) and toggle
the theme from the left rail - asserting expected state transitions and
zero exceptions.
Also verifies the packaging reserve: the spec compiles and the ``nlapt-gui``
entry point resolves.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import DEFAULT_THEME
from nlapt_gui.widgets.main_window import MainWindow
from nlapt_gui.widgets.sections.find_replace import FindReplaceSection

K1 = "0001.png"
K2 = "0002.png"

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def window(qtbot, controller) -> MainWindow:
    manager = ThemeManager(persist=False)
    manager.apply(DEFAULT_THEME)
    win = MainWindow(controller, manager)
    qtbot.addWidget(win)
    win.show()
    qtbot.waitExposed(win)
    return win


class TestEndToEnd:
    def test_pick_file_in_file_panel(self, qtbot, window, controller) -> None:
        controller.set_current(K1)
        qtbot.waitUntil(
            lambda: window.file_panel.cell(K2) is not None
            and window.file_panel.cell(K2).width() > 30,
            timeout=2000,
        )
        cell = window.file_panel.cell(K2)
        qtbot.mouseClick(cell, Qt.MouseButton.LeftButton, pos=cell.rect().center())
        assert controller.current_key == K2

    def test_switch_editor_modes(self, qtbot, window, controller) -> None:
        for mode in ("sents", "text", "chips"):
            qtbot.mouseClick(
                window.editor_panel._tab_buttons[mode], Qt.MouseButton.LeftButton
            )
            assert controller.mode == mode
            qtbot.wait(10)
        blocks = window.editor_panel.blocks()
        assert blocks and blocks[0].key == controller.current_key

    def test_edit_chip_in_editor(self, qtbot, window, controller) -> None:
        controller.set_current(K1)
        controller.set_mode("chips")
        qtbot.waitUntil(
            lambda: window.editor_panel.block_for(K1) is not None, timeout=2000
        )
        editor = window.editor_panel.block_for(K1).editor
        qtbot.waitUntil(lambda: len(editor.chips()) > 0, timeout=2000)
        editor.start_edit(0)
        editor.set_editor_text("1boy")
        qtbot.keyClick(editor._field, Qt.Key.Key_Return)
        assert controller.segments(K1)[0] == "1boy"
        assert controller.history.entries(K1)[0].label.startswith("编辑分段")

    def test_replace_all_scope_current(self, qtbot, window, controller, toasts) -> None:
        controller.set_current(K2)
        section = window.tools_panel.section("fr").findChild(FindReplaceSection)
        assert section is not None
        section.scope.set_current("current")
        section.find_input.setText("school uniform")
        section.replace_input.setText("sailor suit")
        with qtbot.waitSignal(controller.batch_finished, timeout=2000):
            qtbot.mouseClick(section.apply_button, Qt.MouseButton.LeftButton)
        assert "sailor suit" in controller.record(K2).text
        assert any("已在 1 个文件中替换" in text for text, _kind in toasts)

    def test_toggle_theme_from_rail(self, qtbot, window, controller) -> None:
        popup = window.rail.open_theme_popup()
        qtbot.addWidget(popup)
        target = next(row for row in popup.rows() if row.theme_name == "深邃")
        qtbot.mouseClick(target, Qt.MouseButton.LeftButton)
        assert controller.settings.theme == "深邃"
        # Back to the default so later tests see a known state.
        window._theme_manager.apply(DEFAULT_THEME)
        assert controller.settings.theme == DEFAULT_THEME

    def test_save_flow_after_edits(self, qtbot, window, controller) -> None:
        controller.set_caption(K1, "smoke save text", "编辑")
        with qtbot.waitSignal(controller.files_saved, timeout=2000):
            qtbot.keyClick(window, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
        saved = (controller.image_path(K1).with_suffix(".txt")).read_text(encoding="utf-8")
        assert saved == "smoke save text"


class TestPackagingReserve:
    def test_spec_compiles(self) -> None:
        spec = REPO_ROOT / "packaging" / "nlapt.spec"
        source = spec.read_text(encoding="utf-8")
        compile(source, str(spec), "exec")

    def test_gui_script_entry_resolves(self) -> None:
        module = importlib.import_module("nlapt_gui.__main__")
        assert callable(module.main)

    def test_pyproject_declares_gui_script(self) -> None:
        import tomllib

        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert data["project"]["gui-scripts"]["nlapt-gui"] == "nlapt_gui.__main__:main"
        assert "PySide6" in data["project"]["optional-dependencies"]["gui"]

    def test_build_script_exists(self) -> None:
        assert (REPO_ROOT / "packaging" / "build.ps1").exists()
        assert (REPO_ROOT / "packaging" / "README.md").exists()
