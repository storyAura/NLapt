"""Dialog: replace transparent backgrounds with a solid or random color."""

from __future__ import annotations

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from shiboken6 import isValid

from nlapt.core.errors import NLaptError
from nlapt.images.alpha import (
    FlattenReport,
    FlattenSpec,
    MODE_FIXED,
    MODE_RANDOM,
    has_transparency,
)

from nlapt_gui.controller import AppController
from nlapt_gui.image_tools_bridge import OP_FLATTEN, ImageToolsBridge
from nlapt_gui.widgets.dialogs import CenteredDialog
from nlapt_gui.widgets.sections.common import ScopeRow
from nlapt_gui.widgets.tools_panel import SegmentedBar
from nlapt_gui.workers import run_async

WINDOW_TITLE = "替换透明底"
HINT_BACKUP = "将原图备份到数据集 .backups/images/ 后再覆盖写入。"
LABEL_MODE = "替换方式"
LABEL_FIXED = "固定纯色"
LABEL_RANDOM = "随机颜色"
LABEL_PRESETS = "预设颜色"
LABEL_CUSTOM = "自定义…"
LABEL_PALETTE = "随机调色板（从这些颜色里为每张图抽色）"
LABEL_ADD_SWATCH = "添加颜色"
BUTTON_RUN = "开始替换"
BUTTON_CANCEL = "取消"
STATS_SCANNING = "正在统计透明底…"
STATS_READY = "范围内 {n} 张，其中 {alpha} 张有透明底"
STATS_EMPTY = "范围内没有图片"
PROGRESS_FMT = "{done} / {total}"

PRESET_WHITE = (255, 255, 255)
PRESET_BLACK = (0, 0, 0)
PRESET_GRAY = (160, 160, 160)
PRESETS = (PRESET_WHITE, PRESET_BLACK, PRESET_GRAY)
SWATCH_PX = 22
DIALOG_W = 460


def _rgb_css(color: tuple[int, int, int]) -> str:
    return f"rgb({color[0]}, {color[1]}, {color[2]})"


class FlattenAlphaDialog(CenteredDialog):
    """Scope + color mode + run flatten through :class:`ImageToolsBridge`."""

    def __init__(
        self,
        controller: AppController,
        bridge: ImageToolsBridge,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._bridge = bridge
        self._color = PRESET_WHITE
        self._palette: list[tuple[int, int, int]] = [PRESET_WHITE, PRESET_GRAY, PRESET_BLACK]
        self._scan_gen = 0
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumWidth(DIALOG_W)
        self.setModal(True)

        hint = QLabel(HINT_BACKUP, self)
        hint.setWordWrap(True)
        hint.setProperty("muted", True)

        self.scope_row = ScopeRow(controller, self)
        self.scope_row.changed.connect(lambda _s: self._scan_alpha())

        self.mode = SegmentedBar(
            ((MODE_FIXED, LABEL_FIXED), (MODE_RANDOM, LABEL_RANDOM)),
            current=MODE_FIXED,
            parent=self,
        )
        self.mode.changed.connect(lambda _m: self._refresh_mode())

        self.preset_row = QHBoxLayout()
        self.preset_row.setContentsMargins(0, 0, 0, 0)
        self.preset_row.setSpacing(8)
        preset_label = QLabel(LABEL_PRESETS, self)
        preset_label.setProperty("muted", True)
        self.preset_row.addWidget(preset_label)
        self._preset_buttons: list[QPushButton] = []
        for color in PRESETS:
            button = self._swatch_button(color)
            button.clicked.connect(lambda _=False, c=color: self._set_color(c))
            self.preset_row.addWidget(button)
            self._preset_buttons.append(button)
        self.custom_button = QPushButton(LABEL_CUSTOM, self)
        self.custom_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.custom_button.clicked.connect(self._pick_custom)
        self.preset_row.addWidget(self.custom_button)
        self.preset_row.addStretch(1)

        self.palette_host = QWidget(self)
        self.palette_layout = QHBoxLayout(self.palette_host)
        self.palette_layout.setContentsMargins(0, 0, 0, 0)
        self.palette_layout.setSpacing(6)
        palette_label = QLabel(LABEL_PALETTE, self.palette_host)
        palette_label.setProperty("muted", True)
        self.palette_layout.addWidget(palette_label)
        self._rebuild_palette_chips()

        self.stats_label = QLabel(STATS_SCANNING, self)
        self.stats_label.setProperty("muted", True)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.hide()

        self.cancel_button = QPushButton(BUTTON_CANCEL, self)
        self.cancel_button.setProperty("variant", "outline")
        self.cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cancel_button.clicked.connect(self.reject)
        self.run_button = QPushButton(BUTTON_RUN, self)
        self.run_button.setProperty("variant", "accent")
        self.run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.run_button.clicked.connect(self._run)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)
        layout.addWidget(hint)
        layout.addWidget(self.scope_row)
        mode_row = QHBoxLayout()
        mode_caption = QLabel(LABEL_MODE, self)
        mode_caption.setProperty("muted", True)
        mode_row.addWidget(mode_caption)
        mode_row.addWidget(self.mode, 1)
        layout.addLayout(mode_row)
        layout.addLayout(self.preset_row)
        layout.addWidget(self.palette_host)
        layout.addWidget(self.stats_label)
        layout.addWidget(self.progress)
        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.run_button)
        layout.addLayout(actions)

        bridge.progress.connect(self._on_progress)
        bridge.flatten_finished.connect(self._on_finished)
        self._refresh_mode()
        self._scan_alpha()

    def current_spec(self) -> FlattenSpec:
        if self.mode.current == MODE_RANDOM:
            return FlattenSpec(mode=MODE_RANDOM, color=self._color, palette=tuple(self._palette))
        return FlattenSpec(mode=MODE_FIXED, color=self._color)

    def _keys(self) -> tuple[str, ...]:
        return self._controller.scope_keys(self.scope_row.scope)

    def _refresh_mode(self) -> None:
        self.palette_host.setVisible(self.mode.current == MODE_RANDOM)

    def _set_color(self, color: tuple[int, int, int]) -> None:
        self._color = color

    def _pick_custom(self) -> None:
        chosen = QColorDialog.getColor(QColor(*self._color), self, LABEL_CUSTOM)
        if chosen.isValid():
            self._set_color((chosen.red(), chosen.green(), chosen.blue()))

    def _add_palette_color(self) -> None:
        chosen = QColorDialog.getColor(QColor(*self._color), self, LABEL_ADD_SWATCH)
        if not chosen.isValid():
            return
        color = (chosen.red(), chosen.green(), chosen.blue())
        if color not in self._palette:
            self._palette.append(color)
            self._rebuild_palette_chips()

    def _rebuild_palette_chips(self) -> None:
        # Remove previous swatches (keep the first label and last add+stretch).
        while self.palette_layout.count() > 1:
            item = self.palette_layout.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for color in self._palette:
            chip = self._swatch_button(color, parent=self.palette_host)
            chip.setToolTip(_rgb_css(color))
            chip.clicked.connect(lambda _=False, c=color: self._remove_palette(c))
            self.palette_layout.addWidget(chip)
        add = QPushButton(LABEL_ADD_SWATCH, self.palette_host)
        add.setCursor(Qt.CursorShape.PointingHandCursor)
        add.clicked.connect(self._add_palette_color)
        self.palette_layout.addWidget(add)
        self.palette_layout.addStretch(1)

    def _remove_palette(self, color: tuple[int, int, int]) -> None:
        if len(self._palette) <= 1:
            return
        self._palette = [item for item in self._palette if item != color]
        self._rebuild_palette_chips()

    def _swatch_button(
        self, color: tuple[int, int, int], *, parent: QWidget | None = None
    ) -> QPushButton:
        button = QPushButton(parent or self)
        button.setFixedSize(SWATCH_PX, SWATCH_PX)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setStyleSheet(
            f"background: {_rgb_css(color)}; border: 1px solid palette(mid);"
            " border-radius: 4px;"
        )
        return button

    def _scan_alpha(self) -> None:
        keys = self._keys()
        if not keys:
            self.stats_label.setText(STATS_EMPTY)
            self.run_button.setEnabled(False)
            return
        self.stats_label.setText(STATS_SCANNING)
        self.run_button.setEnabled(False)
        paths = []
        for key in keys:
            try:
                paths.append(self._controller.image_path(key))
            except NLaptError:
                continue
        self._scan_gen += 1
        generation = self._scan_gen

        def job() -> int:
            return sum(1 for path in paths if has_transparency(path))

        def done(count: object) -> None:
            if not isValid(self) or generation != self._scan_gen:
                return
            alpha = count if isinstance(count, int) else 0
            self.stats_label.setText(STATS_READY.format(n=len(keys), alpha=alpha))
            self.run_button.setEnabled(alpha > 0)

        def failed(message: str) -> None:
            if not isValid(self) or generation != self._scan_gen:
                return
            self.stats_label.setText(message)
            self.run_button.setEnabled(bool(keys))

        run_async(QThreadPool.globalInstance(), job, on_done=done, on_error=failed)

    def _run(self) -> None:
        keys = self._keys()
        if not keys:
            return
        self.run_button.setEnabled(False)
        self.progress.setRange(0, max(1, len(keys)))
        self.progress.setValue(0)
        self.progress.show()
        self._bridge.flatten(keys, self.current_spec())

    def _on_progress(self, description: str, done: int, total: int) -> None:
        if description != OP_FLATTEN:
            return
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.progress.setFormat(PROGRESS_FMT.format(done=done, total=total))

    def _on_finished(self, report: object) -> None:
        self.progress.hide()
        self.run_button.setEnabled(True)
        if isinstance(report, FlattenReport):
            self.accept()

