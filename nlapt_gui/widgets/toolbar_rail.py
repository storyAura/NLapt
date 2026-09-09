"""44px left icon rail — open / save / export / undo / theme / settings.

Common actions sit in a scrollable middle; 主题 and 设置 stay pinned at
the bottom. The rail talks only to :class:`AppController` and
:class:`ThemeManager`; folder picking, settings, export confirm, and the
tools drawer are delegated through signals.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from nlapt.core.config import ROLE_TEXT, ROLE_VISION
from nlapt.diagnostics import get_logger

from nlapt_gui.controller import AppController
from nlapt_gui.theme.logo import LogoWidget
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import ThemeTokens
from nlapt_gui.widgets.dialogs import show_message
from nlapt_gui.widgets.thumb_cells import make_icon
from nlapt_gui.widgets.title_bar import ABOUT_TEXT, ABOUT_TITLE
from nlapt_gui.widgets.tools_menu import ToolsMenuPopup
from nlapt_gui.widgets.window_chrome import ThemePopup, WindowDragHelper

_LOGGER = get_logger(__name__)

RAIL_W = 44
BTN_PX = 32
ICON_PX = 18
LOGO_PX = 22

TIP_OPEN = "打开文件夹"
TIP_SAVE = "保存"
TIP_SAVE_ALL = "全部保存"
TIP_EXPORT = "导出"
TIP_UNDO = "撤销"
TIP_REDO = "重做"
TIP_TOOLS = "修改工具"
TIP_TOOLS_MENU = "工具"
TIP_THEME = "切换主题"
TIP_SETTINGS = "设置"
TIP_LOGO = "NLapt"
ACTION_ABOUT = "关于"
ACTION_QUIT = "退出"


class ToolbarRail(QFrame):
    """Vertical icon strip on the left of the main window."""

    open_folder_requested = Signal()
    settings_requested = Signal()
    colors_requested = Signal()
    export_requested = Signal()
    tools_toggled = Signal(bool)
    tool_action_requested = Signal(str)
    model_switch_requested = Signal(str, object)  # (role, ModelRef)

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
        self._tools_menu: ToolsMenuPopup | None = None
        self._drag = WindowDragHelper(self)
        self._tools_open = False
        self.setFixedWidth(RAIL_W)
        self.setProperty("panel", True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._build()
        self.apply_tokens(theme_manager.tokens)
        theme_manager.theme_changed.connect(self.apply_tokens)
        controller.caption_changed.connect(lambda _k: self._refresh_history())
        controller.current_changed.connect(lambda _k: self._refresh_history())
        controller.history.changed.connect(lambda _k: self._refresh_history())
        controller.busy_changed.connect(self._on_busy)
        self._refresh_history()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 8, 6, 8)
        root.setSpacing(6)

        self.logo_button = QPushButton(self)
        self.logo_button.setFixedSize(BTN_PX, BTN_PX)
        self.logo_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.logo_button.setToolTip(TIP_LOGO)
        self.logo_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        logo_lay = QVBoxLayout(self.logo_button)
        logo_lay.setContentsMargins(5, 5, 5, 5)
        self.logo = LogoWidget(self._manager.tokens, LOGO_PX, self.logo_button)
        logo_lay.addWidget(self.logo, 0, Qt.AlignmentFlag.AlignCenter)
        self.logo_button.clicked.connect(self._open_logo_menu)
        root.addWidget(self.logo_button, 0, Qt.AlignmentFlag.AlignHCenter)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        mid = QWidget(scroll)
        mid_lay = QVBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)
        mid_lay.setSpacing(6)
        mid_lay.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        self.open_button = self._icon_button(TIP_OPEN, self.open_folder_requested.emit)
        self.tools_menu_button = self._icon_button(TIP_TOOLS_MENU, self.open_tools_menu)
        self.save_button = self._icon_button(TIP_SAVE, self._controller.save_current)
        self.save_all_button = self._icon_button(TIP_SAVE_ALL, self._controller.save_all)
        self.export_button = self._icon_button(TIP_EXPORT, self.export_requested.emit)
        self.undo_button = self._icon_button(TIP_UNDO, self._controller.undo_current)
        self.redo_button = self._icon_button(TIP_REDO, self._controller.redo_current)
        self.tools_button = self._icon_button(TIP_TOOLS, self._toggle_tools)
        mid_lay.addWidget(self.open_button)
        mid_lay.addWidget(self.tools_menu_button)
        mid_lay.addWidget(self.save_button)
        mid_lay.addWidget(self.save_all_button)
        mid_lay.addWidget(self.export_button)
        mid_lay.addWidget(self.undo_button)
        mid_lay.addWidget(self.redo_button)
        mid_lay.addWidget(self.tools_button)
        mid_lay.addStretch(1)
        scroll.setWidget(mid)
        root.addWidget(scroll, 1)

        self.theme_button = self._icon_button(TIP_THEME, self.open_theme_popup)
        self.settings_button = self._icon_button(TIP_SETTINGS, self.settings_requested.emit)
        root.addWidget(self.theme_button, 0, Qt.AlignmentFlag.AlignHCenter)
        root.addWidget(self.settings_button, 0, Qt.AlignmentFlag.AlignHCenter)

        self._icon_buttons: dict[str, QPushButton] = {
            "open": self.open_button,
            "apps": self.tools_menu_button,
            "save": self.save_button,
            "save_all": self.save_all_button,
            "export": self.export_button,
            "undo": self.undo_button,
            "redo": self.redo_button,
            "tools": self.tools_button,
            "theme": self.theme_button,
            "settings": self.settings_button,
        }

    def _icon_button(self, tip: str, slot) -> QPushButton:  # noqa: ANN001
        button = QPushButton(self)
        button.setFixedSize(BTN_PX, BTN_PX)
        button.setToolTip(tip)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setProperty("railBtn", True)
        button.clicked.connect(slot)
        return button

    def apply_tokens(self, tokens: ThemeTokens) -> None:
        self.setStyleSheet(
            f"""
            ToolbarRail {{
                background: {tokens.panel};
                border-right: 1px solid {tokens.bd};
            }}
            QPushButton[railBtn="true"] {{
                background: transparent;
                border: none;
                border-radius: 8px;
            }}
            QPushButton[railBtn="true"]:hover {{
                background: {tokens.surface2};
            }}
            QPushButton[railBtn="true"][railOn="true"] {{
                background: {tokens.surface2};
            }}
            """
        )
        self.logo.set_tokens(tokens)
        self.logo_button.setStyleSheet(
            f"QPushButton {{ background: {tokens.accent}; border: none;"
            f" border-radius: 8px; }}"
        )
        color = tokens.text2
        mapping = {
            "open": "folder_open",
            "apps": "apps",
            "save": "save",
            "save_all": "save_all",
            "export": "export",
            "undo": "undo",
            "redo": "redo",
            "tools": "tools",
            "theme": "theme",
            "settings": "settings",
        }
        for name, kind in mapping.items():
            self._icon_buttons[name].setIcon(make_icon(kind, color, ICON_PX))

    def _refresh_history(self) -> None:
        self.undo_button.setEnabled(self._controller.can_undo())
        self.redo_button.setEnabled(self._controller.can_redo())

    def _on_busy(self, busy: bool) -> None:
        self.save_button.setEnabled(not busy)
        self.save_all_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy)
        if not busy:
            self._refresh_history()

    def _toggle_tools(self) -> None:
        self.set_tools_open(not self._tools_open)

    def set_tools_open(self, open_: bool) -> None:
        self._tools_open = open_
        self.tools_button.setProperty("railOn", open_)
        self.tools_button.style().unpolish(self.tools_button)
        self.tools_button.style().polish(self.tools_button)
        self.tools_toggled.emit(open_)

    def tools_open(self) -> bool:
        return self._tools_open

    def open_tools_menu(self) -> ToolsMenuPopup:
        """Build and show the grouped 工具 popup next to the tools-menu button."""
        popup = ToolsMenuPopup(
            self._manager.tokens,
            busy=self._controller.batch_running(),
            choices=self._controller.model_pool(),
            text_target=self._controller.model_target(ROLE_TEXT),
            vision_target=self._controller.model_target(ROLE_VISION),
            parent=self,
        )
        popup.action_triggered.connect(self.tool_action_requested.emit)
        popup.model_switch_requested.connect(self.model_switch_requested.emit)
        anchor = self.tools_menu_button.mapToGlobal(
            QPoint(self.tools_menu_button.width() + 6, 0)
        )
        popup.move(anchor)
        popup.show()
        self._tools_menu = popup
        return popup

    def open_theme_popup(self) -> ThemePopup:
        """Build and show the theme picker next to the theme button."""
        popup = ThemePopup(self._manager.theme_name, self._manager.tokens, self)
        popup.theme_picked.connect(self._pick_theme)
        popup.colors_requested.connect(self.colors_requested.emit)
        anchor = self.theme_button.mapToGlobal(
            QPoint(self.theme_button.width() + 6, self.theme_button.height() - 8)
        )
        # Keep the card on-screen when the rail sits at the left edge.
        popup.move(anchor)
        popup.show()
        self._popup = popup
        return popup

    def _pick_theme(self, name: str) -> None:
        _LOGGER.info("theme picked from rail: %s", name)
        self._manager.apply(name)

    def _open_logo_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction(ACTION_ABOUT, self.show_about)
        menu.addSeparator()
        menu.addAction(ACTION_QUIT, lambda: self.window().close())
        menu.popup(self.logo_button.mapToGlobal(QPoint(self.logo_button.width() + 4, 0)))

    def show_about(self) -> None:
        show_message(self.window(), ABOUT_TITLE, ABOUT_TEXT)

    def _on_empty(self, event: QMouseEvent) -> bool:
        return self.childAt(event.position().toPoint()) is None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag.press(event, self._on_empty(event)):
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag.move(event):
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag.release()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag.double_click(event, self._on_empty(event)):
            return
        super().mouseDoubleClickEvent(event)
