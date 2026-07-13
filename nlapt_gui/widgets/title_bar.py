"""Frameless title bar - logo, menus, theme popup and window controls.

Faithful to the design's 标题栏 block: 40px strip on the panel color with a
painted accent logo, ``NLapt`` + mono version label, five real
:class:`QMenu` buttons (文件/编辑/视图/工具/帮助), a theme button opening the
230px theme popup, and 44px min/max/close buttons (close hovers the
Windows-native red, provided as a theme token).

Dragging empty title-bar space starts a system move; double-click toggles
maximize/restore. The bar talks only to :class:`AppController` and
:class:`ThemeManager`; folder picking and the settings dialog are delegated
to the main window through signals.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger

from nlapt_gui import __version__
from nlapt_gui.controller import AppController
from nlapt_gui.theme.logo import LogoWidget
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import (
    MONO_STACK,
    THEMES,
    ThemeTokens,
    WINDOWS_CLOSE_HOVER,
    WINDOWS_CLOSE_HOVER_FG,
)
from nlapt_gui.widgets.dialogs import show_message

_LOGGER = get_logger(__name__)

# -- geometry from the design -----------------------------------------------------
TITLE_BAR_HEIGHT = 40
LOGO_PX = 22
WIN_BUTTON_W = 44
THEME_BUTTON_H = 28
DIVIDER_H = 18
THEME_POPUP_W = 230
SWATCH_PX = 13
GLYPH_PX = 10

# -- exact UI strings ---------------------------------------------------------------
APP_NAME = "NLapt"
VERSION_LABEL = f"自然语言标注工具 v{__version__}"
MENU_FILE = "文件"
MENU_EDIT = "编辑"
MENU_VIEW = "视图"
MENU_TOOLS = "工具"
MENU_HELP = "帮助"
ACTION_OPEN_FOLDER = "打开文件夹…"
ACTION_REFRESH = "刷新"
ACTION_SAVE = "保存"
ACTION_SAVE_ALL = "全部保存"
ACTION_QUIT = "退出"
ACTION_UNDO = "撤销"
ACTION_COPY = "复制标注"
SUBMENU_VIEW_MODE = "视图模式"
SUBMENU_THEME = "主题"
ACTION_COLORS = "色彩设置…"
ACTION_SETTINGS = "设置…"
# 快捷键 was removed from 帮助 (spec module 3.5): the status bar at the bottom
# of the window already lists the live shortcut hints.
ACTION_GUIDE = "使用说明"
ACTION_ABOUT = "关于"
ABOUT_TITLE = "关于 NLapt"
GUIDE_TITLE = "使用说明"
GUIDE_TEXT = (
    "NLapt 采用三栏工作流:\n\n"
    "左栏 — 文件列表与多选:浏览数据集,支持 Shift 范围选、"
    "Alt 取消选,对多张图片批量操作。\n\n"
    "中栏 — 预览 + 编辑:图片预览与三种编辑模式"
    "(胶囊 / 分句 / 文本),悬浮工具栏可分段 / 插入 / 翻译 / 重译。\n\n"
    "右栏 — 修改工具:查找替换、前缀 / 后缀、翻译对照、历史记录。\n\n"
    "保存与快照:标注仅在显式「保存 / 全部保存」时写入 .txt 文件;"
    "批量操作会自动创建快照,未保存的草稿在崩溃后可恢复。"
)
ABOUT_TEXT = (
    f"{APP_NAME}\n"
    f"{VERSION_LABEL}\n\n"
    "面向自然语言标注(caption)处理的桌面编辑工具。\n"
    "在三栏界面中高效编辑、批量修改并翻译图像标注,\n"
    "兼容 kohya 等「图片 + 同名 .txt」数据集格式。"
)
POPUP_TITLE = "界面主题"
POPUP_CUSTOM_COLORS = "自定义色彩…"
VIEW_MODE_LABELS: tuple[tuple[str, str], ...] = (
    ("list", "列表视图"),
    ("mid", "网格视图"),
    ("big", "大图视图"),
)

# Title bar left inset aligned with the panel headers below (16px grid).
BAR_LEFT_INSET = 16

# Window-control glyph kinds.
KIND_MIN = "min"
KIND_MAX = "max"
KIND_CLOSE = "close"

_MONO_FAMILY = ", ".join(f'"{name}"' for name in MONO_STACK)


class _PaletteIcon(QWidget):
    """14px palette glyph inside the theme button (design's stroke icon)."""

    def __init__(self, tokens: ThemeTokens, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tokens = tokens
        self.setFixedSize(14, 14)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_tokens(self, tokens: ThemeTokens) -> None:
        self._tokens = tokens
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scale = self.width() / 16.0
        pen = QPen(QColor(self._tokens.text2), 1.5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(8 * scale, 8 * scale), 6.2 * scale, 6.2 * scale)
        path = QPainterPath(QPointF(8 * scale, 1.8 * scale))
        path.cubicTo(
            QPointF(6.4 * scale, 4.6 * scale),
            QPointF(5.0 * scale, 6.4 * scale),
            QPointF(6.9 * scale, 8.3 * scale),
        )
        path.cubicTo(
            QPointF(8.4 * scale, 9.5 * scale),
            QPointF(7.4 * scale, 11.6 * scale),
            QPointF(7.8 * scale, 14.2 * scale),
        )
        painter.drawPath(path)
        painter.end()


class _WindowButton(QWidget):
    """44px min/max/close button with a painted 10x10 glyph.

    Painted (not QSS-styled) so the close button can swap to the
    Windows-native hover red with a light glyph, per the design.
    """

    clicked = Signal()

    def __init__(
        self, kind: str, tokens: ThemeTokens, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.kind = kind
        self._tokens = tokens
        self._hover = False
        self.setFixedWidth(WIN_BUTTON_W)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def set_tokens(self, tokens: ThemeTokens) -> None:
        self._tokens = tokens
        self.update()

    def event(self, ev: QEvent) -> bool:  # noqa: N802 - Qt override
        if ev.type() == QEvent.Type.HoverEnter:
            self._hover = True
            self.update()
        elif ev.type() == QEvent.Type.HoverLeave:
            self._hover = False
            self.update()
        return super().event(ev)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        # Accept the press so it does not bubble to TitleBar.mousePressEvent,
        # which would otherwise start a system move and swallow the click's
        # release (leaving the window control unclickable on Windows).
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        tokens = self._tokens
        if self._hover:
            bg = WINDOWS_CLOSE_HOVER if self.kind == KIND_CLOSE else tokens.surface2
            painter.fillRect(self.rect(), QColor(bg))
            glyph = WINDOWS_CLOSE_HOVER_FG if self.kind == KIND_CLOSE else tokens.text
        else:
            glyph = tokens.text2
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(glyph), 1))
        left = (self.width() - GLYPH_PX) / 2.0
        top = (self.height() - GLYPH_PX) / 2.0
        if self.kind == KIND_MIN:
            mid = top + GLYPH_PX / 2.0
            painter.drawLine(QPointF(left, mid), QPointF(left + GLYPH_PX, mid))
        elif self.kind == KIND_MAX:
            painter.drawRect(QRectF(left + 0.5, top + 0.5, GLYPH_PX - 1, GLYPH_PX - 1))
        else:
            painter.drawLine(QPointF(left, top), QPointF(left + GLYPH_PX, top + GLYPH_PX))
            painter.drawLine(QPointF(left + GLYPH_PX, top), QPointF(left, top + GLYPH_PX))
        painter.end()


class _CheckMark(QWidget):
    """13px accent check for the active theme row."""

    def __init__(self, tokens: ThemeTokens, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tokens = tokens
        self.setFixedSize(SWATCH_PX, SWATCH_PX)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(self._tokens.accent), 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        scale = self.width() / 14.0
        path = QPainterPath(QPointF(2.5 * scale, 7.5 * scale))
        path.lineTo(QPointF(5.5 * scale, 10.5 * scale))
        path.lineTo(QPointF(11.5 * scale, 3.5 * scale))
        painter.drawPath(path)
        painter.end()


class _ThemeRow(QFrame):
    """One clickable theme row: three swatch dots + name + optional check."""

    picked = Signal(str)

    def __init__(
        self,
        theme_name: str,
        active: bool,
        tokens: ThemeTokens,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.theme_name = theme_name
        self.setProperty("themeRow", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        base = THEMES[theme_name]
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 7, 10, 7)
        row.setSpacing(9)
        swatches = QHBoxLayout()
        swatches.setSpacing(3)
        for color, bordered in ((base.bg, True), (base.surface, True), (base.accent, False)):
            dot = QLabel(self)
            dot.setFixedSize(SWATCH_PX, SWATCH_PX)
            border = f" border: 1px solid {tokens.bd2};" if bordered else ""
            dot.setStyleSheet(
                f"background: {color}; border-radius: {SWATCH_PX // 2}px;{border}"
            )
            swatches.addWidget(dot)
        row.addLayout(swatches)
        name = QLabel(theme_name, self)
        name.setStyleSheet(f"font-size: 12.5px; color: {tokens.text}; background: transparent;")
        row.addWidget(name, 1)
        if active:
            row.addWidget(_CheckMark(tokens, self))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self.picked.emit(self.theme_name)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ThemePopup(QFrame):
    """230px theme picker card (design 主题弹层), shown as a Qt popup."""

    theme_picked = Signal(str)
    colors_requested = Signal()

    def __init__(self, active_theme: str, tokens: ThemeTokens, parent: QWidget | None = None) -> None:
        super().__init__(
            parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint
        )
        self.setFixedWidth(THEME_POPUP_W)
        self.setStyleSheet(
            f"""
            ThemePopup {{
                background: {tokens.surface};
                border: 1px solid {tokens.bd};
                border-radius: 10px;
            }}
            QFrame[themeRow="true"] {{ border-radius: 7px; background: transparent; }}
            QFrame[themeRow="true"]:hover {{ background: {tokens.surface2}; }}
            QPushButton[popupAction="true"] {{
                background: transparent;
                color: {tokens.text2};
                border: none;
                border-radius: 7px;
                padding: 7px 10px;
                font-size: 12px;
                text-align: left;
            }}
            QPushButton[popupAction="true"]:hover {{
                background: {tokens.surface2};
                color: {tokens.text};
            }}
            """
        )
        column = QVBoxLayout(self)
        column.setContentsMargins(6, 6, 6, 6)
        column.setSpacing(0)
        title = QLabel(POPUP_TITLE, self)
        title.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {tokens.text3};"
            " padding: 6px 10px 8px; letter-spacing: 0.4px; background: transparent;"
        )
        column.addWidget(title)
        self._rows: list[_ThemeRow] = []
        for name in THEMES:
            row = _ThemeRow(name, name == active_theme, tokens, self)
            row.picked.connect(self._on_pick)
            column.addWidget(row)
            self._rows.append(row)
        divider = QFrame(self)
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background: {tokens.bd}; margin: 5px 6px;")
        column.addWidget(divider)
        self.colors_button = QPushButton(POPUP_CUSTOM_COLORS, self)
        self.colors_button.setProperty("popupAction", True)
        self.colors_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.colors_button.clicked.connect(self._on_colors)
        column.addWidget(self.colors_button)

    def rows(self) -> tuple[_ThemeRow, ...]:
        return tuple(self._rows)

    def _on_pick(self, name: str) -> None:
        self.theme_picked.emit(name)
        self.close()

    def _on_colors(self) -> None:
        self.colors_requested.emit()
        self.close()


class TitleBar(QFrame):
    """The frameless window's 40px title strip."""

    open_folder_requested = Signal()
    settings_requested = Signal()
    colors_requested = Signal()

    def __init__(
        self,
        controller: AppController,
        theme_manager: ThemeManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._manager = theme_manager
        self._popup: ThemePopup | None = None
        self.setFixedHeight(TITLE_BAR_HEIGHT)
        self._build()
        self._build_menus()
        self.apply_tokens(theme_manager.tokens)
        theme_manager.theme_changed.connect(self.apply_tokens)
        controller.view_mode_changed.connect(self._sync_view_actions)

    # -- construction --------------------------------------------------------------
    def _build(self) -> None:
        row = QHBoxLayout(self)
        # Left inset matches the panel headers below so the logo/menus align
        # with the workspace columns (spec module 4.1).
        row.setContentsMargins(BAR_LEFT_INSET, 0, 0, 0)
        row.setSpacing(4)
        self.logo = LogoWidget(self._manager.tokens, LOGO_PX, self)
        row.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignVCenter)
        # Name + mono subtitle share one baseline (design: display:flex;
        # align-items:baseline; gap:7px). They live in a content-hugging
        # container so both labels flush to the same bottom edge (baseline
        # approximation) while the container itself is vertically centered in
        # the 40px bar -- nothing overflows or overlaps the row below.
        self.name_group = QWidget(self)
        name_box = QHBoxLayout(self.name_group)
        name_box.setContentsMargins(8, 0, 10, 0)
        name_box.setSpacing(7)
        self.name_label = QLabel(APP_NAME, self.name_group)
        name_box.addWidget(self.name_label, 0, Qt.AlignmentFlag.AlignBottom)
        self.version_label = QLabel(VERSION_LABEL, self.name_group)
        self.version_label.setProperty("mono", "true")
        name_box.addWidget(self.version_label, 0, Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self.name_group, 0, Qt.AlignmentFlag.AlignVCenter)
        self._menu_row = QHBoxLayout()
        self._menu_row.setSpacing(2)
        row.addLayout(self._menu_row)
        row.addStretch(1)
        self.theme_button = QPushButton(self)
        self.theme_button.setProperty("themeButton", True)
        self.theme_button.setFixedHeight(THEME_BUTTON_H)
        self.theme_button.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_row = QHBoxLayout(self.theme_button)
        btn_row.setContentsMargins(11, 0, 11, 0)
        btn_row.setSpacing(7)
        self._palette_icon = _PaletteIcon(self._manager.tokens, self.theme_button)
        btn_row.addWidget(self._palette_icon)
        self.theme_name_label = QLabel(self._manager.theme_name, self.theme_button)
        self.theme_name_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        btn_row.addWidget(self.theme_name_label)
        self.theme_arrow_label = QLabel("▾", self.theme_button)
        self.theme_arrow_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        btn_row.addWidget(self.theme_arrow_label)
        self.theme_button.clicked.connect(self.open_theme_popup)
        row.addWidget(self.theme_button)
        self.divider = QFrame(self)
        self.divider.setFixedSize(1, DIVIDER_H)
        row.addSpacing(8)
        row.addWidget(self.divider)
        row.addSpacing(8)
        tokens = self._manager.tokens
        self.min_button = _WindowButton(KIND_MIN, tokens, self)
        self.max_button = _WindowButton(KIND_MAX, tokens, self)
        self.close_button = _WindowButton(KIND_CLOSE, tokens, self)
        self.min_button.clicked.connect(self._minimize)
        self.max_button.clicked.connect(self.toggle_max_restore)
        self.close_button.clicked.connect(lambda: self.window().close())
        for button in (self.min_button, self.max_button, self.close_button):
            row.addWidget(button)

    def _menu_button(self, label: str, menu: QMenu) -> QPushButton:
        button = QPushButton(label, self)
        button.setProperty("titleMenu", True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(
            lambda: menu.popup(button.mapToGlobal(QPoint(0, button.height() + 4)))
        )
        self._menu_row.addWidget(button)
        return button

    def _build_menus(self) -> None:
        controller = self._controller
        self.menus: dict[str, QMenu] = {}
        self.menu_buttons: dict[str, QPushButton] = {}

        file_menu = QMenu(MENU_FILE, self)
        self.action_open_folder = QAction(ACTION_OPEN_FOLDER, self)
        self.action_open_folder.triggered.connect(self.open_folder_requested.emit)
        self.action_refresh = QAction(ACTION_REFRESH, self)
        self.action_refresh.triggered.connect(controller.refresh)
        self.action_save = QAction(f"{ACTION_SAVE}\tCtrl+S", self)
        self.action_save.triggered.connect(controller.save_current)
        self.action_save_all = QAction(ACTION_SAVE_ALL, self)
        self.action_save_all.triggered.connect(controller.save_all)
        self.action_quit = QAction(ACTION_QUIT, self)
        self.action_quit.triggered.connect(lambda: self.window().close())
        file_menu.addActions(
            [self.action_open_folder, self.action_refresh, self.action_save, self.action_save_all]
        )
        file_menu.addSeparator()
        file_menu.addAction(self.action_quit)

        edit_menu = QMenu(MENU_EDIT, self)
        self.action_undo = QAction(f"{ACTION_UNDO}\tCtrl+Z", self)
        self.action_undo.triggered.connect(controller.undo_current)
        self.action_copy = QAction(ACTION_COPY, self)
        self.action_copy.triggered.connect(controller.copy_caption)
        edit_menu.addActions([self.action_undo, self.action_copy])

        view_menu = QMenu(MENU_VIEW, self)
        view_mode_menu = view_menu.addMenu(SUBMENU_VIEW_MODE)
        self._view_group = QActionGroup(self)
        self._view_group.setExclusive(True)
        self.view_mode_actions: dict[str, QAction] = {}
        for mode, label in VIEW_MODE_LABELS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(mode == controller.view_mode)
            action.triggered.connect(
                lambda _checked=False, m=mode: self._controller.set_view_mode(m)
            )
            self._view_group.addAction(action)
            view_mode_menu.addAction(action)
            self.view_mode_actions[mode] = action
        theme_menu = view_menu.addMenu(SUBMENU_THEME)
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        self.theme_actions: dict[str, QAction] = {}
        for name in THEMES:
            action = QAction(name, self)
            action.setCheckable(True)
            action.setChecked(name == self._manager.theme_name)
            action.triggered.connect(
                lambda _checked=False, n=name: self._manager.apply(n)
            )
            self._theme_group.addAction(action)
            theme_menu.addAction(action)
            self.theme_actions[name] = action
        self.action_colors = QAction(ACTION_COLORS, self)
        self.action_colors.triggered.connect(self.colors_requested.emit)
        view_menu.addAction(self.action_colors)

        tools_menu = QMenu(MENU_TOOLS, self)
        self.action_settings = QAction(ACTION_SETTINGS, self)
        self.action_settings.triggered.connect(self.settings_requested.emit)
        tools_menu.addAction(self.action_settings)

        help_menu = QMenu(MENU_HELP, self)
        # 快捷键 deliberately absent (module 3.5) — the status bar shows them live.
        self.action_guide = QAction(ACTION_GUIDE, self)
        self.action_guide.triggered.connect(self.show_guide)
        self.action_about = QAction(ACTION_ABOUT, self)
        self.action_about.triggered.connect(self.show_about)
        help_menu.addAction(self.action_guide)
        help_menu.addSeparator()
        help_menu.addAction(self.action_about)

        for label, menu in (
            (MENU_FILE, file_menu),
            (MENU_EDIT, edit_menu),
            (MENU_VIEW, view_menu),
            (MENU_TOOLS, tools_menu),
            (MENU_HELP, help_menu),
        ):
            self.menus[label] = menu
            self.menu_buttons[label] = self._menu_button(label, menu)

    # -- theme ----------------------------------------------------------------------
    def apply_tokens(self, tokens: ThemeTokens) -> None:
        """Restyle every token-dependent part of the bar."""
        self.setStyleSheet(
            f"""
            TitleBar {{
                background: {tokens.panel};
                border-bottom: 1px solid {tokens.bd};
            }}
            QPushButton[titleMenu="true"] {{
                padding: 5px 10px;
                border: none;
                border-radius: 6px;
                background: transparent;
                font-size: 12.5px;
                color: {tokens.text2};
            }}
            QPushButton[titleMenu="true"]:hover {{
                background: {tokens.surface2};
                color: {tokens.text};
            }}
            QPushButton[themeButton="true"] {{
                border: 1px solid {tokens.bd};
                border-radius: 7px;
                background: {tokens.surface};
                padding: 0;
            }}
            QPushButton[themeButton="true"]:hover {{ border-color: {tokens.bd2}; }}
            """
        )
        self.name_label.setStyleSheet(
            f"font-weight: 700; font-size: 13px; letter-spacing: 0.2px;"
            f" color: {tokens.text}; background: transparent;"
        )
        self.version_label.setStyleSheet(
            f"font-family: {_MONO_FAMILY}; font-size: 10px;"
            f" color: {tokens.text3}; background: transparent;"
        )
        self.theme_name_label.setStyleSheet(
            f"font-size: 12px; color: {tokens.text2}; background: transparent;"
        )
        self.theme_arrow_label.setStyleSheet(
            f"font-size: 9px; color: {tokens.text3}; background: transparent;"
        )
        self.divider.setStyleSheet(f"background: {tokens.bd};")
        self.logo.set_tokens(tokens)
        self._palette_icon.set_tokens(tokens)
        for button in (self.min_button, self.max_button, self.close_button):
            button.set_tokens(tokens)
        self.theme_name_label.setText(self._manager.theme_name)
        action = self.theme_actions.get(self._manager.theme_name)
        if action is not None:
            action.setChecked(True)

    def _sync_view_actions(self, view_mode: str) -> None:
        action = self.view_mode_actions.get(view_mode)
        if action is not None:
            action.setChecked(True)

    # -- theme popup ------------------------------------------------------------------
    def open_theme_popup(self) -> ThemePopup:
        """Build and show the theme picker under the theme button."""
        popup = ThemePopup(self._manager.theme_name, self._manager.tokens, self)
        popup.theme_picked.connect(self._pick_theme)
        popup.colors_requested.connect(self.colors_requested.emit)
        anchor = self.theme_button.mapToGlobal(
            QPoint(self.theme_button.width() - THEME_POPUP_W, self.theme_button.height() + 4)
        )
        popup.move(anchor)
        popup.show()
        self._popup = popup
        return popup

    def _pick_theme(self, name: str) -> None:
        _LOGGER.info("theme picked from popup: %s", name)
        self._manager.apply(name)

    # -- window controls / drag ---------------------------------------------------------
    def _minimize(self) -> None:
        self.window().showMinimized()

    def toggle_max_restore(self) -> None:
        window = self.window()
        if window.isMaximized():
            window.showNormal()
        else:
            window.showMaximized()

    def show_guide(self) -> None:
        """Explain the three-column workflow and the save/snapshot model."""
        show_message(self.window(), GUIDE_TITLE, GUIDE_TEXT)

    def show_about(self) -> None:
        show_message(self.window(), ABOUT_TITLE, ABOUT_TEXT)

    def _on_empty_bar(self, event: QMouseEvent) -> bool:
        """True when the press is on bare title-bar space (no child control)."""
        return self.childAt(event.position().toPoint()) is None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        # Only start a window move from empty bar space; presses on the menus,
        # theme button or window controls must reach those widgets.
        if event.button() == Qt.MouseButton.LeftButton and self._on_empty_bar(event):
            handle = self.window().windowHandle()
            if handle is not None:
                handle.startSystemMove()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton and self._on_empty_bar(event):
            self.toggle_max_restore()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
