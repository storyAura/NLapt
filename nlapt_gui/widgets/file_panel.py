"""Left column 文件列表: dataset header, search, view modes, folder groups.

Faithful to the CaptionForge design: header (folder icon, dataset name,
mono path, refresh/open buttons), search input, view segmented control,
全选/已选/清除 row, collapsible folder groups with a reflowing thumbnail
grid or list rows, empty state and the footer stat line.

文件夹多选 + 推标: every folder header carries a tri-state checkbox that
selects the whole folder, an ALL row above the groups selects everything,
and right-clicking opens the 推标 menu scoped to the click target — an
image cell (这张图片, or 已选 + 全部 when it is part of a multi-selection),
a folder header (此文件夹 / 此文件夹未标注 / 全部), or the ALL row and the
根目录 group (全部 / 全部未标注 only). All state
flows through :class:`AppController`; outward signals are
``open_folder_requested``, ``infer_requested(keys, engine)`` and
``layered_infer_requested(keys)`` (the main window routes the latter two
into the vision bridge).
"""

from __future__ import annotations

import math

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent, QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger

from nlapt_gui import anim as ui_anim
from nlapt_gui.controller import AppController
from nlapt_gui.theme.tokens import ThemeTokens
from nlapt_gui.widgets.dialogs import ask_confirm
from nlapt_gui.widgets.file_panel_infer import (
    CONFIRM_INFER_TEXT,
    CONFIRM_INFER_TITLE,
    ENGINE_CONFIRM_WORDS,
    build_infer_actions,
    popup_infer_menu,
)
from nlapt_gui.widgets.thumb_cells import (
    ListRow,
    ThumbCell,
    WarnDot,
    make_icon,
    mono_font,
    tokens_for_settings,
    ui_font,
)
from nlapt_gui.widgets.thumbnails import ThumbnailLoader

_LOGGER = get_logger(__name__)

PANEL_WIDTH = 260
HEADER_ICON_PX = 17
TOOL_BUTTON_PX = 26
TOOL_ICON_PX = 13
SEARCH_HEIGHT = 30
SEG_BUTTON_W = 52
SEG_BUTTON_H = 24
GRID_GAP = 10
LIST_GAP = 6
GROUP_HEADER_H = 27
HEADER_INSET = (6, 0, 6, 0)
HEADER_GAP = 7
BIG_THUMB_MIN = 150  # design: minmax(150px,1fr) in big mode
ARROW_OPEN_DEG = 0.0
ARROW_CLOSED_DEG = -90.0
COLLAPSE_MS = 240
LEGEND_DOT_PX = 7
FOOTER_DOT_GAP = 4
_MAX_WIDGET_H = 16_777_215  # Qt QWIDGETSIZE_MAX

# Exact strings from the design.
SEARCH_PLACEHOLDER = "搜索文件名 / 标签…"
TEXT_SELECT_ALL = "全选"
TEXT_SELECTED_FMT = "已选 {n}"
TEXT_CLEAR = "清除"
TEXT_NO_MATCH = "没有匹配的文件"
TEXT_UNSAVED = "未保存"
TIP_REFRESH = "刷新"
TIP_OPEN_FOLDER = "打开文件夹"
COUNT_FMT = "{n} 张"
COUNT_FILTERED_FMT = "{k}/{n} 张"
VIEW_TIPS = {"list": "详细列表", "mid": "中图网格", "big": "大图网格"}
VIEW_LABELS = {"list": "列表", "mid": "中图", "big": "大图"}
# ALL row + folder checkbox strings.
TEXT_ALL_ROW = "ALL"
TIP_FOLDER_CHECK = "选中 / 取消选中整个文件夹"
TIP_ALL_CHECK = "选中 / 取消选中全部文件"

# Selection coverage -> Qt check state (folder checkbox + ALL row).
_COVERAGE_STATES = {
    "all": Qt.CheckState.Checked,
    "some": Qt.CheckState.PartiallyChecked,
    "none": Qt.CheckState.Unchecked,
}


class _Arrow(QWidget):
    """Rotating ▾ disclosure arrow (0deg open / -90deg closed, animated)."""

    def __init__(self, open_: bool, panel: "FilePanel", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel = panel
        self._angle = ARROW_OPEN_DEG if open_ else ARROW_CLOSED_DEG
        self._anim: QPropertyAnimation | None = None
        self.setFixedSize(12, 12)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def _get_angle(self) -> float:
        return self._angle

    def _set_angle(self, value: float) -> None:
        self._angle = value
        self.update()

    angle = Property(float, _get_angle, _set_angle)

    def animate_to(self, open_: bool) -> None:
        end = ARROW_OPEN_DEG if open_ else ARROW_CLOSED_DEG
        ui_anim.stop_animation(self._anim)
        self._anim = None
        if not ui_anim.animations_enabled():
            self._set_angle(end)
            return
        motion = QPropertyAnimation(self, b"angle", self)
        motion.setDuration(COLLAPSE_MS)
        motion.setEasingCurve(QEasingCurve.Type.OutCubic)
        motion.setEndValue(end)
        motion.start()
        self._anim = motion

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(self.rect().center())
        painter.rotate(self._angle)
        painter.setFont(ui_font(9))
        painter.setPen(QColor(self._panel.current_tokens().text3))
        painter.drawText(QRectF(-6, -6, 12, 12), Qt.AlignmentFlag.AlignCenter, "▾")
        painter.end()


class _ThumbGrid(QWidget):
    """Reflowing grid: auto-fill minmax(min_w, 1fr) columns like the design."""

    def __init__(self, cells: list[ThumbCell], min_w: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cells = cells
        self._min_w = max(1, min_w)
        for cell in cells:
            cell.setParent(self)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        width = self.width()
        if not self._cells or width < self._min_w // 2:
            return
        cols = max(1, (width + GRID_GAP) // (self._min_w + GRID_GAP))
        cell_w = (width - GRID_GAP * (cols - 1)) / cols
        cell_h = round(cell_w * 4 / 3)  # design aspect-ratio 3/4
        for index, cell in enumerate(self._cells):
            row, col = divmod(index, cols)
            x = round(col * (cell_w + GRID_GAP))
            right = round((col + 1) * (cell_w + GRID_GAP)) - GRID_GAP
            cell.setGeometry(x, row * (cell_h + GRID_GAP), right - x, cell_h)
        rows = math.ceil(len(self._cells) / cols)
        height = rows * cell_h + (rows - 1) * GRID_GAP if rows else 0
        if self.height() != height:
            self.setFixedHeight(height)


class _FolderGroup(QWidget):
    """One collapsible folder section: header row + animated content."""

    def __init__(
        self,
        panel: "FilePanel",
        folder: str,
        count_text: str,
        content: QWidget,
        open_: bool,
    ) -> None:
        super().__init__(panel)
        self._panel = panel
        self.folder = folder
        self._open = open_
        self._anim: QPropertyAnimation | None = None

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(0)

        self.header = QPushButton(self)
        self.header.setFixedHeight(GROUP_HEADER_H)
        self.header.setCursor(Qt.CursorShape.PointingHandCursor)
        # Right-click a folder header -> the 推标 (batch caption) menu.
        self.header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.header.customContextMenuRequested.connect(
            lambda pos: panel.show_infer_menu(self.folder, self.header, pos)
        )
        header_lay = QHBoxLayout(self.header)
        header_lay.setContentsMargins(*HEADER_INSET)
        header_lay.setSpacing(HEADER_GAP)
        # Folder multi-select checkbox (checkbox clicks never toggle collapse:
        # the checkbox consumes its own mouse events).
        self.check = QCheckBox(self.header)
        self.check.setToolTip(TIP_FOLDER_CHECK)
        self.check.setCursor(Qt.CursorShape.PointingHandCursor)
        self.check.clicked.connect(
            lambda _checked: panel._on_folder_check_clicked(self.folder)
        )
        header_lay.addWidget(self.check)
        self.arrow = _Arrow(open_, panel, self.header)
        header_lay.addWidget(self.arrow)
        self._icon = QLabel(self.header)
        self._icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        header_lay.addWidget(self._icon)
        self.name_label = QLabel(folder, self.header)
        self.name_label.setFont(ui_font(12, QFont.Weight.DemiBold))
        self.name_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        header_lay.addWidget(self.name_label, 1)
        self.count_label = QLabel(count_text, self.header)
        self.count_label.setFont(mono_font(10))
        self.count_label.setProperty("muted", True)
        self.count_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        header_lay.addWidget(self.count_label)
        box.addWidget(self.header)

        self._content = QWidget(self)
        content_lay = QVBoxLayout(self._content)
        content_lay.setContentsMargins(2, 6, 2, 6)
        content_lay.setSpacing(0)
        content_lay.addWidget(content)
        box.addWidget(self._content)
        if not open_:
            self._content.setMaximumHeight(0)

        self.header.clicked.connect(self._on_header_clicked)
        self.refresh_icon()

    @property
    def is_open(self) -> bool:
        return self._open

    def set_check_state(self, coverage: str) -> None:
        """Reflect the folder's selection coverage ('all'|'some'|'none')."""
        self.check.blockSignals(True)
        self.check.setCheckState(_COVERAGE_STATES[coverage])
        self.check.blockSignals(False)

    def refresh_icon(self) -> None:
        tokens = self._panel.current_tokens()
        self._icon.setPixmap(make_icon("folder", tokens.text2, 13).pixmap(13, 13))

    def _on_header_clicked(self) -> None:
        self._panel._toggle_folder(self)

    def set_open(self, open_: bool, animate: bool) -> None:
        if open_ == self._open:
            return
        self._open = open_
        ui_anim.stop_animation(self._anim)
        self._anim = None
        visible = self._content.height()
        if not animate or not ui_anim.animations_enabled():
            self._content.setMaximumHeight(_MAX_WIDGET_H if open_ else 0)
            self.arrow._set_angle(ARROW_OPEN_DEG if open_ else ARROW_CLOSED_DEG)
            return
        self.arrow.animate_to(open_)
        # Pin max-height to the *visible* height first. After expand we lift
        # it to QWIDGETSIZE_MAX; a 0→0 tween then leaves that sentinel in
        # place and the folder looks stuck open.
        if visible <= 0:
            visible = max(self._content.sizeHint().height(), 0)
        if open_:
            self._content.setMaximumHeight(_MAX_WIDGET_H)
            target = max(self._content.sizeHint().height(), 1)
            self._content.setMaximumHeight(visible)
        elif visible <= 0:
            self._content.setMaximumHeight(0)
            return
        else:
            target = 0
            self._content.setMaximumHeight(visible)
        motion = QPropertyAnimation(self._content, b"maximumHeight", self)
        motion.setDuration(COLLAPSE_MS)
        motion.setEasingCurve(QEasingCurve.Type.OutCubic)
        motion.setStartValue(visible)
        motion.setEndValue(target)
        if open_:
            motion.finished.connect(self._unlock_content_height)
        motion.start()
        self._anim = motion

    def _unlock_content_height(self) -> None:
        if self._open:
            self._content.setMaximumHeight(_MAX_WIDGET_H)


class _AllRow(QWidget):
    """The ALL row above the folder groups: select-everything checkbox.

    Mirrors a folder header (checkbox + bold name + count) but covers the
    whole dataset; right-click opens the 推标 menu scoped to 全部/全部未标注.
    """

    def __init__(self, panel: "FilePanel", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel = panel
        self.setFixedHeight(GROUP_HEADER_H)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(
            lambda pos: panel.show_infer_menu(None, self, pos)
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(*HEADER_INSET)
        row.setSpacing(HEADER_GAP)
        self.check = QCheckBox(self)
        self.check.setToolTip(TIP_ALL_CHECK)
        self.check.setCursor(Qt.CursorShape.PointingHandCursor)
        self.check.clicked.connect(lambda _c: panel._on_all_row_clicked())
        row.addWidget(self.check)
        self.name_label = QLabel(TEXT_ALL_ROW, self)
        self.name_label.setFont(ui_font(12, QFont.Weight.DemiBold))
        row.addWidget(self.name_label, 1)
        self.count_label = QLabel("", self)
        self.count_label.setFont(mono_font(10))
        self.count_label.setProperty("muted", True)
        row.addWidget(self.count_label)

    def refresh(self, coverage: str, total: int) -> None:
        self.check.blockSignals(True)
        self.check.setCheckState(_COVERAGE_STATES[coverage])
        self.check.blockSignals(False)
        self.count_label.setText(COUNT_FMT.format(n=total))
        self.setVisible(total > 0)


class FilePanel(QFrame):
    """The design's 文件列表 column, 300px wide, driven by AppController."""

    open_folder_requested = Signal()
    # Batch 推标 request: (keys tuple, engine "llm"|"local").
    infer_requested = Signal(object, str)
    # 分层推标 wizard: keys tuple (engine is chosen inside the dialog).
    layered_infer_requested = Signal(object)

    def __init__(
        self,
        controller: AppController,
        *,
        loader: ThumbnailLoader | None = None,
        tokens: ThemeTokens | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._loader = loader if loader is not None else ThumbnailLoader(parent=self)
        self._tokens = tokens if tokens is not None else tokens_for_settings(controller.settings)
        self._cells: dict[str, QWidget] = {}
        self._groups: list[_FolderGroup] = []
        self.setProperty("panel", True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(PANEL_WIDTH)
        self._build_ui()
        self._connect_controller()
        self._refresh_header()
        self._rebuild_groups()
        self._refresh_counts()
        self._refresh_view_buttons()
        self._apply_icon_colors()

    @property
    def thumbnail_loader(self) -> ThumbnailLoader:
        """Shared thumbnail loader (wizard preview; do not touch the FS)."""
        return self._loader

    # -- construction ---------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Compact dataset line: name + path + refresh / open (open also lives on the rail).
        header = QWidget(self)
        header_lay = QHBoxLayout(header)
        header_lay.setContentsMargins(12, 10, 12, 4)
        header_lay.setSpacing(6)
        self._header_icon = QLabel(header)
        header_lay.addWidget(self._header_icon)
        names = QVBoxLayout()
        names.setContentsMargins(0, 0, 0, 0)
        names.setSpacing(0)
        self.name_label = QLabel(header)
        self.name_label.setFont(ui_font(12.5, QFont.Weight.DemiBold))
        names.addWidget(self.name_label)
        self.path_label = QLabel(header)
        self.path_label.setFont(mono_font(10))
        self.path_label.setProperty("muted", True)
        names.addWidget(self.path_label)
        header_lay.addLayout(names, 1)
        self.refresh_button = QPushButton(header)
        self.refresh_button.setFixedSize(TOOL_BUTTON_PX, TOOL_BUTTON_PX)
        self.refresh_button.setToolTip(TIP_REFRESH)
        self.refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_button.clicked.connect(self._controller.refresh)
        header_lay.addWidget(self.refresh_button)
        self.open_button = QPushButton(header)
        self.open_button.setFixedSize(TOOL_BUTTON_PX, TOOL_BUTTON_PX)
        self.open_button.setToolTip(TIP_OPEN_FOLDER)
        self.open_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_button.clicked.connect(self.open_folder_requested.emit)
        header_lay.addWidget(self.open_button)
        root.addWidget(header)

        # search input with magnifier icon
        search_row = QWidget(self)
        search_lay = QHBoxLayout(search_row)
        search_lay.setContentsMargins(12, 4, 12, 6)
        self.search_edit = QLineEdit(search_row)
        self.search_edit.setFixedHeight(SEARCH_HEIGHT)
        self.search_edit.setPlaceholderText(SEARCH_PLACEHOLDER)
        self.search_edit.setFont(ui_font(12))
        self.search_edit.setClearButtonEnabled(True)
        self._search_action = self.search_edit.addAction(
            make_icon("search", self._tokens.text3, TOOL_ICON_PX),
            QLineEdit.ActionPosition.LeadingPosition,
        )
        self.search_edit.textChanged.connect(self._controller.set_filter)
        search_lay.addWidget(self.search_edit)
        root.addWidget(search_row)

        # view segmented | 已选 n  (全选 / 清除 stay for the existing selection model)
        view_row = QWidget(self)
        view_lay = QHBoxLayout(view_row)
        view_lay.setContentsMargins(12, 0, 12, 8)
        view_lay.setSpacing(8)
        seg_bar = QFrame(view_row)
        seg_bar.setProperty("segBar", True)
        seg_lay = QHBoxLayout(seg_bar)
        seg_lay.setContentsMargins(2, 2, 2, 2)
        seg_lay.setSpacing(2)
        self.view_buttons: dict[str, QPushButton] = {}
        for mode in ("list", "mid", "big"):
            btn = QPushButton(VIEW_LABELS[mode], seg_bar)
            btn.setProperty("seg", True)
            btn.setFixedSize(SEG_BUTTON_W, SEG_BUTTON_H)
            btn.setToolTip(VIEW_TIPS[mode])
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, m=mode: self._controller.set_view_mode(m))
            seg_lay.addWidget(btn)
            self.view_buttons[mode] = btn
        view_lay.addWidget(seg_bar)
        view_lay.addStretch(1)
        self.select_all_box = QCheckBox(TEXT_SELECT_ALL, view_row)
        self.select_all_box.setFont(ui_font(11.5))
        self.select_all_box.setCursor(Qt.CursorShape.PointingHandCursor)
        self.select_all_box.clicked.connect(self._on_select_all_clicked)
        view_lay.addWidget(self.select_all_box)
        self.selected_label = QLabel(TEXT_SELECTED_FMT.format(n=0), view_row)
        self.selected_label.setFont(ui_font(11.5))
        self.selected_label.setProperty("muted", True)
        view_lay.addWidget(self.selected_label)
        self.clear_button = QPushButton(TEXT_CLEAR, view_row)
        self.clear_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_button.clicked.connect(self._controller.clear_selection)
        view_lay.addWidget(self.clear_button)
        root.addWidget(view_row)

        # scrollable folder groups
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._groups_host = QWidget(self._scroll)
        self._groups_lay = QVBoxLayout(self._groups_host)
        self._groups_lay.setContentsMargins(12, 0, 12, 10)
        self._groups_lay.setSpacing(LIST_GAP)
        # ALL row on top of the folders (select everything / 推标全部).
        self.all_row = _AllRow(self, self._groups_host)
        self._groups_lay.addWidget(self.all_row)
        self.empty_label = QLabel(TEXT_NO_MATCH, self._groups_host)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setFont(ui_font(12))
        self.empty_label.setProperty("muted", True)
        self.empty_label.setContentsMargins(10, 26, 10, 26)
        self.empty_label.hide()
        self._groups_lay.addWidget(self.empty_label)
        self._groups_lay.addStretch(1)
        self._scroll.setWidget(self._groups_host)
        root.addWidget(self._scroll, 1)

        # footer: stat line + warn legend
        divider = QFrame(self)
        divider.setProperty("divider", True)
        divider.setFixedHeight(1)
        root.addWidget(divider)
        footer = QWidget(self)
        footer_lay = QHBoxLayout(footer)
        footer_lay.setContentsMargins(14, 9, 14, 9)
        footer_lay.setSpacing(10)
        self.stat_label = QLabel(footer)
        self.stat_label.setFont(ui_font(11))
        self.stat_label.setProperty("secondary", True)
        footer_lay.addWidget(self.stat_label)
        footer_lay.addStretch(1)
        # Live unsaved indicator (was a static legend in the prototype): shows
        # the real dirty count and hides entirely when everything is saved.
        legend = QHBoxLayout()
        legend.setContentsMargins(0, 0, 0, 0)
        legend.setSpacing(FOOTER_DOT_GAP)
        self._unsaved_dot = WarnDot(LEGEND_DOT_PX, self.current_tokens, footer)
        legend.addWidget(self._unsaved_dot)
        self.unsaved_label = QLabel(TEXT_UNSAVED, footer)
        self.unsaved_label.setFont(ui_font(11))
        self.unsaved_label.setProperty("muted", True)
        legend.addWidget(self.unsaved_label)
        footer_lay.addLayout(legend)
        root.addWidget(footer)
        self._refresh_unsaved_indicator()

    def _connect_controller(self) -> None:
        c = self._controller
        c.dataset_opened.connect(self._on_dataset_opened)
        c.filter_changed.connect(self._on_filter_changed)
        c.view_mode_changed.connect(self._on_view_mode_changed)
        c.selection_changed.connect(self._on_selection_changed)
        c.current_changed.connect(self._on_current_changed)
        c.caption_changed.connect(self._on_caption_changed)

    # -- tokens ----------------------------------------------------------------------
    def current_tokens(self) -> ThemeTokens:
        """Tokens provider for custom-painted children."""
        return self._tokens

    def apply_tokens(self, tokens: ThemeTokens) -> None:
        """React to a theme change: re-tint icons and repaint painted parts."""
        self._tokens = tokens
        self._apply_icon_colors()
        for group in self._groups:
            group.refresh_icon()
        self.update()
        for cell in self._cells.values():
            cell.update()

    def _apply_icon_colors(self) -> None:
        t = self._tokens
        self._header_icon.setPixmap(
            make_icon("folder", t.accent, HEADER_ICON_PX).pixmap(HEADER_ICON_PX, HEADER_ICON_PX)
        )
        self.refresh_button.setIcon(make_icon("refresh", t.text3, TOOL_ICON_PX))
        self.open_button.setIcon(make_icon("folder_open", t.text3, TOOL_ICON_PX))
        self._search_action.setIcon(make_icon("search", t.text3, TOOL_ICON_PX))
        self.clear_button.setStyleSheet(
            f"QPushButton {{ color: {t.accent}; background: transparent;"
            f" border: none; padding: 0; font-size: 11.5px; }}"
            f"QPushButton:hover {{ color: {t.accent2}; }}"
        )
        self._refresh_view_buttons()

    # -- controller reactions ----------------------------------------------------------
    def _on_dataset_opened(self, _result: object) -> None:
        self._loader.clear()
        self._refresh_header()
        self._rebuild_groups()
        self._refresh_counts()

    def _on_filter_changed(self, text: str) -> None:
        if self.search_edit.text() != text:
            self.search_edit.blockSignals(True)
            self.search_edit.setText(text)
            self.search_edit.blockSignals(False)
        self._rebuild_groups()
        self._refresh_counts()

    def _on_view_mode_changed(self, _mode: str) -> None:
        self._refresh_view_buttons()
        self._rebuild_groups()

    def _on_selection_changed(self) -> None:
        self._refresh_counts()
        self._repaint_cells()

    def _on_current_changed(self, _key: str) -> None:
        self._repaint_cells()

    def _on_caption_changed(self, key: str) -> None:
        cell = self._cells.get(key)
        if cell is not None:
            cell.update()
        self._refresh_counts()

    # -- interactions -----------------------------------------------------------------
    def _on_select_all_clicked(self, checked: bool) -> None:
        if checked:
            self._controller.select_all()
        else:
            self._controller.clear_selection()

    def _on_all_row_clicked(self) -> None:
        """ALL row checkbox: partial/none -> select everything; full -> clear."""
        controller = self._controller
        if len(controller.selected_keys()) == len(controller.keys()):
            controller.clear_selection()
        else:
            controller.select_all()

    def _on_folder_check_clicked(self, folder: str) -> None:
        """Folder checkbox: partial/none -> select the folder; full -> clear it."""
        state = self._controller.folder_selection_state(folder)
        self._controller.set_folder_selected(folder, state != "all")

    # -- 推标 context menu ---------------------------------------------------------------
    def infer_menu_actions(
        self, folder: str | None, image: str | None = None
    ) -> list[tuple[str, object]]:
        """(label, callable) entries for the 推标 menu (also the test seam)."""
        return build_infer_actions(self, folder, image)

    def show_infer_menu(
        self, folder: str | None, widget: QWidget, pos: QPoint, image: str | None = None
    ) -> None:
        """Popup the 推标 menu for an image cell / folder header / the ALL row."""
        popup_infer_menu(self, folder, widget, pos, image)

    def _request_infer(self, keys: tuple[str, ...], engine: str) -> None:
        """Confirm (captions get overwritten) then hand off to the bridge."""
        word = ENGINE_CONFIRM_WORDS.get(engine, engine)
        if not ask_confirm(
            self.window(),
            CONFIRM_INFER_TITLE,
            CONFIRM_INFER_TEXT.format(word=word, n=len(keys)),
        ):
            return
        self.infer_requested.emit(tuple(keys), engine)

    def _request_layered(self, keys: tuple[str, ...]) -> None:
        """Open the 分层推标 wizard (confirm lives on the last wizard page)."""
        self.layered_infer_requested.emit(tuple(keys))

    def _toggle_folder(self, group: _FolderGroup) -> None:
        open_ = not group.is_open
        group.set_open(open_, animate=True)
        folder_open = dict(self._controller.settings.folder_open)
        folder_open[group.folder] = open_
        self._controller.update_settings(folder_open=folder_open)

    # -- refreshers --------------------------------------------------------------------
    def _refresh_header(self) -> None:
        name, path = self._controller.dataset_label()
        self.name_label.setText(name)
        metrics = self.path_label.fontMetrics()
        available = PANEL_WIDTH - 2 * 12 - HEADER_ICON_PX - 2 * TOOL_BUTTON_PX - 4 * 9
        self.path_label.setText(
            metrics.elidedText(path, Qt.TextElideMode.ElideRight, max(40, available))
        )
        self.path_label.setToolTip(path)

    def _refresh_counts(self) -> None:
        controller = self._controller
        selected = len(controller.selected_keys())
        total = len(controller.keys())
        self.selected_label.setText(TEXT_SELECTED_FMT.format(n=selected))
        self.select_all_box.blockSignals(True)
        self.select_all_box.setChecked(total > 0 and selected == total)
        self.select_all_box.blockSignals(False)
        coverage = "all" if total and selected == total else "some" if selected else "none"
        self.all_row.refresh(coverage, total)
        for group in self._groups:
            group.set_check_state(controller.folder_selection_state(group.folder))
        self.stat_label.setText(controller.stat_line())
        self._refresh_unsaved_indicator()

    def _refresh_unsaved_indicator(self) -> None:
        """Live footer indicator: '未保存 n' while dirty, hidden when clean."""
        dirty = self._controller.dirty_count()
        visible = dirty > 0
        self._unsaved_dot.setVisible(visible)
        self.unsaved_label.setVisible(visible)
        if visible:
            self.unsaved_label.setText(f"{TEXT_UNSAVED} {dirty}")

    def _refresh_view_buttons(self) -> None:
        active = self._controller.view_mode
        for mode, btn in self.view_buttons.items():
            is_active = mode == active
            btn.setProperty("segActive", is_active)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _repaint_cells(self) -> None:
        for cell in self._cells.values():
            cell.update()

    # -- group building ------------------------------------------------------------------
    def _rebuild_groups(self) -> None:
        for group in self._groups:
            group.hide()
            group.deleteLater()
        self._groups = []
        self._cells = {}
        controller = self._controller
        filtering = bool(controller.filter_text.strip())
        visible_set = set(controller.filtered_keys())
        view_mode = controller.view_mode
        # Groups go right below the ALL row (which stays first forever).
        insert_at = self._groups_lay.indexOf(self.all_row) + 1
        total_visible = 0
        for folder in controller.folders():
            folder_keys = [k for k in controller.keys() if controller.folder_of(k) == folder]
            visible = [k for k in folder_keys if k in visible_set]
            if filtering and not visible:
                continue
            total_visible += len(visible)
            if filtering:
                count_text = COUNT_FILTERED_FMT.format(k=len(visible), n=len(folder_keys))
            else:
                count_text = COUNT_FMT.format(n=len(folder_keys))
            content = self._build_group_content(visible, view_mode)
            open_ = bool(self._controller.settings.folder_open.get(folder, True))
            group = _FolderGroup(self, folder, count_text, content, open_)
            group.set_check_state(controller.folder_selection_state(folder))
            self._groups_lay.insertWidget(insert_at, group)
            self._groups.append(group)
            insert_at += 1
        self.empty_label.setVisible(total_visible == 0 and bool(controller.keys()))

    def _build_group_content(self, keys: list[str], view_mode: str) -> QWidget:
        if view_mode == "list":
            host = QWidget()
            lay = QVBoxLayout(host)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(LIST_GAP)
            for key in keys:
                row = ListRow(key, self._controller, self._loader, self.current_tokens)
                self._cells[key] = row
                self._attach_cell_menu(row, key)
                lay.addWidget(row)
            return host
        min_w = BIG_THUMB_MIN if view_mode == "big" else self._controller.settings.thumb_min
        cells = []
        for key in keys:
            cell = ThumbCell(key, self._controller, self._loader, self.current_tokens)
            self._cells[key] = cell
            self._attach_cell_menu(cell, key)
            cells.append(cell)
        return _ThumbGrid(cells, min_w)

    def _attach_cell_menu(self, cell: QWidget, key: str) -> None:
        """Right-clicking an image cell opens the 推标 menu scoped to it."""
        cell.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        cell.customContextMenuRequested.connect(
            lambda pos, w=cell, k=key: self.show_infer_menu(None, w, pos, image=k)
        )

    # -- test/integration helpers ---------------------------------------------------------
    def cell(self, key: str) -> QWidget | None:
        """The grid cell / list row currently showing ``key`` (None if hidden)."""
        return self._cells.get(key)

    def folder_group(self, folder: str) -> _FolderGroup | None:
        for group in self._groups:
            if group.folder == folder:
                return group
        return None
