"""Shared frameless chrome: window buttons, theme popup, drag-to-move.

Extracted from the old title bar so the left toolbar rail and the preview
info bar can reuse the same painted ─ □ × controls and the 230px theme
picker without duplicating token-driven paint code.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPaintEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nlapt_gui.theme.tokens import (
    THEMES,
    ThemeTokens,
    WINDOWS_CLOSE_HOVER,
    WINDOWS_CLOSE_HOVER_FG,
)

# -- geometry ----------------------------------------------------------------
WIN_BUTTON_W = 44
THEME_POPUP_W = 230
SWATCH_PX = 13
GLYPH_PX = 10

KIND_MIN = "min"
KIND_MAX = "max"
KIND_CLOSE = "close"

POPUP_TITLE = "界面主题"
POPUP_CUSTOM_COLORS = "自定义色彩…"


class WindowButton(QWidget):
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
        # Accept the press so it does not bubble to a drag host and start a
        # system move that would swallow the click's release.
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


class ThemeRow(QFrame):
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
    """230px theme picker card (设计 主题弹层), shown as a Qt popup."""

    theme_picked = Signal(str)
    colors_requested = Signal()

    def __init__(
        self, active_theme: str, tokens: ThemeTokens, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
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
        self._rows: list[ThemeRow] = []
        for name in THEMES:
            row = ThemeRow(name, name == active_theme, tokens, self)
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

    def rows(self) -> tuple[ThemeRow, ...]:
        return tuple(self._rows)

    def _on_pick(self, name: str) -> None:
        self.theme_picked.emit(name)
        self.close()

    def _on_colors(self) -> None:
        self.colors_requested.emit()
        self.close()


class WindowDragHelper:
    """Empty-space press → drag starts a system move; double-click zooms.

    The system move starts only after a real drag so a bare press does not
    swallow the double-click's second press (making 双击还原 flaky).
    """

    def __init__(self, host: QWidget) -> None:
        self._host = host
        self._press_pos: QPoint | None = None

    def clear(self) -> None:
        self._press_pos = None

    def press(self, event: QMouseEvent, empty: bool) -> bool:
        if event.button() == Qt.MouseButton.LeftButton and empty:
            self._press_pos = event.globalPosition().toPoint()
            event.accept()
            return True
        return False

    def move(self, event: QMouseEvent) -> bool:
        if (
            self._press_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
            and (event.globalPosition().toPoint() - self._press_pos).manhattanLength()
            >= QApplication.startDragDistance()
        ):
            self._press_pos = None
            window = self._host.window()
            if window.windowState() & (
                Qt.WindowState.WindowMaximized | Qt.WindowState.WindowFullScreen
            ):
                window.showNormal()
            handle = window.windowHandle()
            if handle is not None:
                handle.startSystemMove()
            event.accept()
            return True
        return False

    def release(self) -> None:
        self._press_pos = None

    def double_click(self, event: QMouseEvent, empty: bool) -> bool:
        if event.button() == Qt.MouseButton.LeftButton and empty:
            self._press_pos = None
            toggle_window_max_restore(self._host)
            event.accept()
            return True
        return False


def toggle_window_max_restore(widget: QWidget) -> None:
    """Delegate to the main window's animated toggle when available."""
    window = widget.window()
    toggle = getattr(window, "toggle_max_restore", None)
    if window is not widget and callable(toggle):
        toggle()
        return
    zoomed = window.windowState() & (
        Qt.WindowState.WindowMaximized | Qt.WindowState.WindowFullScreen
    )
    if zoomed:
        window.showNormal()
    else:
        window.showMaximized()


def wire_window_buttons(
    min_button: WindowButton,
    max_button: WindowButton,
    close_button: WindowButton,
    host: QWidget,
) -> None:
    """Connect ─ □ × to minimize / toggle-max / close on ``host``'s window."""
    min_button.clicked.connect(lambda: host.window().showMinimized())
    max_button.clicked.connect(lambda: toggle_window_max_restore(host))
    close_button.clicked.connect(lambda: host.window().close())
