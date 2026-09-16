"""Runtime theme glyphs: file generation, QSS urls, and control rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from nlapt_gui.resources import app_data_dir
from nlapt_gui.theme import THEMES, ThemeManager, build_qss
from nlapt_gui.theme.glyphs import (
    GLYPH_DIR_NAME,
    ensure_glyphs,
    svg_check,
    svg_chevron,
    svg_partial,
)

COLOR_NEAR = 48
INDICATOR_W = 18


def _manhattan(color: QColor, hex_color: str) -> int:
    target = QColor(hex_color)
    return (
        abs(color.red() - target.red())
        + abs(color.green() - target.green())
        + abs(color.blue() - target.blue())
    )


def _pixels(image) -> list[QColor]:
    return [
        image.pixelColor(x, y)
        for x in range(image.width())
        for y in range(image.height())
    ]


def _has_hex(pixels: list[QColor], hex_color: str) -> bool:
    wanted = QColor(hex_color)
    return any(color == wanted for color in pixels)


def _has_near(pixels: list[QColor], hex_color: str, threshold: int = COLOR_NEAR) -> bool:
    return any(_manhattan(color, hex_color) < threshold for color in pixels)


class TestEnsureGlyphs:
    def test_writes_three_files_with_token_colors(self, tmp_path: Path, qapp) -> None:
        tokens = THEMES["雾灰"]
        paths = ensure_glyphs(tokens, tmp_path)
        assert paths is not None
        chevron = Path(paths.chevron)
        check = Path(paths.check)
        partial = Path(paths.partial)
        assert chevron.parent == tmp_path
        assert chevron.read_text(encoding="utf-8") == svg_chevron(tokens.text3)
        assert check.read_text(encoding="utf-8") == svg_check(tokens.onaccent)
        assert partial.read_text(encoding="utf-8") == svg_partial(tokens.onaccent)
        assert f'stroke="{tokens.text3}"' in chevron.read_text(encoding="utf-8")
        assert QPixmap(str(chevron)).isNull() is False

    def test_second_call_keeps_mtime(self, tmp_path: Path) -> None:
        tokens = THEMES["深邃"]
        first = ensure_glyphs(tokens, tmp_path)
        assert first is not None
        path = Path(first.chevron)
        mtime = path.stat().st_mtime_ns
        second = ensure_glyphs(tokens, tmp_path)
        assert second is not None
        assert second.chevron == first.chevron
        assert path.stat().st_mtime_ns == mtime

    def test_default_dir_is_app_data_theme(self) -> None:
        tokens = THEMES["明亮"]
        paths = ensure_glyphs(tokens)
        assert paths is not None
        assert Path(paths.check).parent == app_data_dir() / GLYPH_DIR_NAME

    def test_unwritable_returns_none_and_qss_has_no_image(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(self: Path, *_args: object, **_kwargs: object) -> None:
            raise OSError("read-only")

        monkeypatch.setattr(Path, "write_text", boom)
        assert ensure_glyphs(THEMES["墨黑"], tmp_path) is None
        qss = build_qss(THEMES["墨黑"], None)
        assert "image:" not in qss


class TestQssGlyphSelectors:
    def test_indicator_and_header_selectors(self, tmp_path: Path) -> None:
        tokens = THEMES["石墨"]
        glyphs = ensure_glyphs(tokens, tmp_path)
        qss = build_qss(tokens, glyphs)
        assert "QAbstractItemView::indicator" in qss
        assert "QCheckBox::indicator:indeterminate" in qss
        assert "QHeaderView::section" in qss
        assert tokens.surface2 in qss
        assert f'url("{glyphs.check}")' in qss
        assert f'url("{glyphs.partial}")' in qss


class TestRenderedControls:
    @pytest.mark.parametrize("name", tuple(THEMES))
    def test_checklist_glyphs_combo_arrow_and_header(
        self, qapp, qtbot, name: str
    ) -> None:
        previous_qss = qapp.styleSheet()
        previous_palette = qapp.palette()
        manager = ThemeManager(qapp, persist=False)
        try:
            tokens = manager.apply(name)
            window = QWidget()
            qtbot.addWidget(window)
            layout = QVBoxLayout(window)
            checklist = QListWidget(window)
            layout.addWidget(checklist)
            boxes: list[QCheckBox] = []
            states = (
                Qt.CheckState.Unchecked,
                Qt.CheckState.Checked,
                Qt.CheckState.PartiallyChecked,
            )
            for state in states:
                item = QListWidgetItem(state.name, checklist)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(state)
                box = QCheckBox(state.name, window)
                box.setCheckState(state)
                layout.addWidget(box)
                boxes.append(box)
            combo = QComboBox(window)
            combo.addItem("Model")
            layout.addWidget(combo)
            tree = QTreeWidget(window)
            tree.setHeaderLabels(["Model", "Size"])
            layout.addWidget(tree)
            window.resize(360, 320)
            window.show()
            qapp.processEvents()

            header = tree.header().grab().toImage()
            assert _has_near(_pixels(header), tokens.surface2)

            for index, box in enumerate(boxes):
                row = checklist.visualItemRect(checklist.item(index))
                list_shot = (
                    checklist.viewport()
                    .grab(row.adjusted(0, 0, INDICATOR_W - row.width(), 0))
                    .toImage()
                )
                box_shot = box.grab().toImage().copy(0, 0, INDICATOR_W, box.height())
                for image in (list_shot, box_shot):
                    colors = _pixels(image)
                    if index == 0:
                        assert _has_hex(colors, tokens.accent) is False
                    else:
                        assert _has_hex(colors, tokens.accent)
                        assert _has_near(colors, tokens.onaccent)

            arrow = combo.grab().toImage().copy(
                combo.width() - 20, 8, 12, max(8, combo.height() - 16)
            )
            background = QColor(tokens.surface)
            assert any(
                arrow.pixelColor(x, y) != background
                for x in range(arrow.width())
                for y in range(arrow.height())
            ), "Combo arrow did not load"
        finally:
            qapp.setStyleSheet(previous_qss)
            qapp.setPalette(previous_palette)

    def test_native_controls_follow_theme_switch(self, qapp, qtbot) -> None:
        previous_qss = qapp.styleSheet()
        previous_palette = qapp.palette()
        spin = QSpinBox()
        qtbot.addWidget(spin)
        spin.show()
        manager = ThemeManager(qapp, persist=False)
        try:
            for name in ("明亮", "深邃", "雾灰"):
                tokens = manager.apply(name)
                qapp.processEvents()
                assert spin.palette().color(QPalette.ColorRole.Base) == QColor(
                    tokens.surface
                )
        finally:
            qapp.setStyleSheet(previous_qss)
            qapp.setPalette(previous_palette)
