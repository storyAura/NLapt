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

from PySide6.QtCore import QPoint, QPointF, Qt, Signal
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
    QWidget,
)

from nlapt.diagnostics import get_logger

from nlapt_gui import __version__
from nlapt_gui.controller import AppController
from nlapt_gui.theme.logo import LogoWidget
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import MONO_STACK, THEMES, ThemeTokens
from nlapt_gui.widgets.dialogs import show_message
from nlapt_gui.widgets.window_chrome import (
    KIND_CLOSE,
    KIND_MAX,
    KIND_MIN,
    THEME_POPUP_W,
    ThemePopup,
    WindowButton,
    WindowDragHelper,
    toggle_window_max_restore,
    wire_window_buttons,
)

# Test / import compatibility: the painted control used to live here.
_WindowButton = WindowButton

_LOGGER = get_logger(__name__)

# -- geometry from the design -----------------------------------------------------
TITLE_BAR_HEIGHT = 40
LOGO_PX = 22
THEME_BUTTON_H = 28
DIVIDER_H = 18

# -- exact UI strings ---------------------------------------------------------------
APP_NAME = "NLapt"
VERSION_LABEL = f"自然语言标注工具 v{__version__}"
MENU_FILE = "文件"
MENU_EDIT = "编辑"
MENU_VIEW = "视图"
MENU_TOOLS = "工具"
MENU_SETTINGS = "设置"
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
# 工具 menu is kept as a home for future tools (设置 moved to its own
# top-level entry); the placeholder is disabled until real tools land.
ACTION_TOOLS_PLACEHOLDER = "更多工具(规划中)"
# 快捷键 was removed from 帮助 (spec module 3.5): the status bar at the bottom
# of the window already lists the live shortcut hints. 使用说明 was dropped
# from the logo / help menus; the product sheet lives in docs/功能清单.md.
ACTION_ABOUT = "关于"
ABOUT_TITLE = "关于 NLapt"
AUTHOR = "storyAura"
ABOUT_TEXT = (
    f"{APP_NAME}\n"
    f"{VERSION_LABEL}\n\n"
    "面向图像生成与 LoRA 训练的桌面标注编辑器。\n"
    "处理「图片 + 同名 .txt」，单张精修，批量可回滚。\n\n"
    f"作者：{AUTHOR}"
)
VIEW_MODE_LABELS: tuple[tuple[str, str], ...] = (
    ("list", "列表视图"),
    ("mid", "网格视图"),
    ("big", "大图视图"),
)

# Title bar left inset aligned with the panel headers below (16px grid).
BAR_LEFT_INSET = 16

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
        self._drag = WindowDragHelper(self)
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
        self.min_button = WindowButton(KIND_MIN, tokens, self)
        self.max_button = WindowButton(KIND_MAX, tokens, self)
        self.close_button = WindowButton(KIND_CLOSE, tokens, self)
        wire_window_buttons(self.min_button, self.max_button, self.close_button, self)
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

        # 设置 stands alone as a top-level entry (see settings_button below);
        # the action object stays for programmatic/open-settings callers.
        self.action_settings = QAction(ACTION_SETTINGS, self)
        self.action_settings.triggered.connect(self.settings_requested.emit)

        tools_menu = QMenu(MENU_TOOLS, self)
        self.action_tools_placeholder = QAction(ACTION_TOOLS_PLACEHOLDER, self)
        self.action_tools_placeholder.setEnabled(False)
        tools_menu.addAction(self.action_tools_placeholder)

        help_menu = QMenu(MENU_HELP, self)
        # 快捷键 / 使用说明 deliberately absent — the status bar shows shortcuts,
        # and the logo menu (live UI) only keeps 关于 / 退出.
        self.action_about = QAction(ACTION_ABOUT, self)
        self.action_about.triggered.connect(self.show_about)
        help_menu.addAction(self.action_about)

        for label, menu in (
            (MENU_FILE, file_menu),
            (MENU_EDIT, edit_menu),
            (MENU_VIEW, view_menu),
            (MENU_TOOLS, tools_menu),
        ):
            self.menus[label] = menu
            self.menu_buttons[label] = self._menu_button(label, menu)
        # Top-level 设置: a direct button (no dropdown) that opens the dialog.
        self.settings_button = QPushButton(MENU_SETTINGS, self)
        self.settings_button.setProperty("titleMenu", True)
        self.settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_button.clicked.connect(self.settings_requested.emit)
        self._menu_row.addWidget(self.settings_button)
        self.menus[MENU_HELP] = help_menu
        self.menu_buttons[MENU_HELP] = self._menu_button(MENU_HELP, help_menu)

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
    def toggle_max_restore(self) -> None:
        """Delegate to the main window's animated toggle (fullscreen-safe)."""
        toggle_window_max_restore(self)

    def show_about(self) -> None:
        show_message(self.window(), ABOUT_TITLE, ABOUT_TEXT)

    def _on_empty_bar(self, event: QMouseEvent) -> bool:
        """True when the press is on bare title-bar space (no child control)."""
        return self.childAt(event.position().toPoint()) is None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag.press(event, self._on_empty_bar(event)):
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag.move(event):
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        self._drag.release()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag.double_click(event, self._on_empty_bar(event)):
            return
        super().mouseDoubleClickEvent(event)
