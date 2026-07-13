"""Toast overlay - bottom-center notification stack (design Toast block).

Pills with a colored 8px dot slide in above the status bar, stack upward and
auto-dismiss after :data:`TOAST_DURATION_MS`. The overlay is transparent to
mouse events and is meant to be connected to
``AppController.toast_requested`` by the window integrator::

    overlay = ToastOverlay(main_window)
    controller.toast_requested.connect(overlay.show_toast)

Dot colors come from :class:`ThemeTokens`; call :meth:`set_tokens` from
``ThemeManager.theme_changed`` so future toasts follow theme switches.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger

from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES, ThemeTokens

_LOGGER = get_logger(__name__)

# Auto-dismiss delay from the prototype (setTimeout 2600).
TOAST_DURATION_MS = 2600
# Distance from the bottom edge (design: bottom 44px).
BOTTOM_MARGIN_PX = 44
# Gap between stacked toasts (design: gap 8px).
STACK_GAP_PX = 8
DOT_SIZE_PX = 8
# Pill inner padding (design: 8px 16px) and dot gap (design: 9px).
_PILL_MARGINS = (16, 8, 16, 8)
_PILL_GAP = 9

# Toast kinds (mirror AppController's toast_requested vocabulary).
KIND_OK = "ok"
KIND_WARN = "warn"
KIND_ERR = "err"
KIND_INFO = "info"


def _dot_color(tokens: ThemeTokens, kind: str) -> str:
    """Prototype mapping: warn/err/info -> warn/danger/accent, default ok."""
    if kind == KIND_WARN:
        return tokens.warn
    if kind == KIND_ERR:
        return tokens.danger
    if kind == KIND_INFO:
        return tokens.accent
    return tokens.ok


class ToastOverlay(QWidget):
    """Click-through overlay stacking toast pills bottom-center."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        duration_ms: int = TOAST_DURATION_MS,
        tokens: ThemeTokens | None = None,
    ) -> None:
        super().__init__(parent)
        self._duration_ms = duration_ms
        self._tokens = tokens if tokens is not None else THEMES[DEFAULT_THEME]
        self._pills: list[QFrame] = []
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, BOTTOM_MARGIN_PX)
        layout.setSpacing(STACK_GAP_PX)
        layout.addStretch(1)
        self._stack = layout
        if parent is not None:
            parent.installEventFilter(self)
            self._sync_geometry()

    # -- public API ---------------------------------------------------------------
    def set_tokens(self, tokens: ThemeTokens) -> None:
        """Adopt new theme tokens (connect to ThemeManager.theme_changed)."""
        self._tokens = tokens

    def show_toast(self, text: str, kind: str = KIND_INFO) -> None:
        """Append a toast pill; it fades in and auto-dismisses."""
        if not text:
            _LOGGER.warning("empty toast text ignored")
            return
        pill = self._build_pill(text, kind)
        self._pills.append(pill)
        self._stack.addWidget(pill, 0, Qt.AlignmentFlag.AlignHCenter)
        self.raise_()
        timer = QTimer(pill)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._dismiss(pill))
        timer.start(self._duration_ms)
        _LOGGER.debug("toast shown (%s): %s", kind, text)

    def active_texts(self) -> tuple[str, ...]:
        """Texts of the currently visible toasts (oldest first)."""
        return tuple(pill.property("toastText") for pill in self._pills)

    # -- internals ------------------------------------------------------------------
    def _build_pill(self, text: str, kind: str) -> QFrame:
        pill = QFrame(self)
        pill.setProperty("toast", True)
        pill.setProperty("toastText", text)
        pill.setProperty("toastKind", kind)
        row = QHBoxLayout(pill)
        row.setContentsMargins(*_PILL_MARGINS)
        row.setSpacing(_PILL_GAP)
        dot = QLabel(pill)
        dot.setFixedSize(DOT_SIZE_PX, DOT_SIZE_PX)
        # Color from the active theme tokens (never a literal in widget code).
        dot.setStyleSheet(
            f"background: {_dot_color(self._tokens, kind)};"
            f" border-radius: {DOT_SIZE_PX // 2}px;"
        )
        row.addWidget(dot)
        label = QLabel(text, pill)
        row.addWidget(label)
        return pill

    # NOTE: toasts appear without an opacity fade on purpose - a
    # QGraphicsOpacityEffect on a pill that is alive while the user resizes
    # the window re-renders through an effect buffer and can crash Qt.

    def _dismiss(self, pill: QFrame) -> None:
        if pill not in self._pills:
            return
        self._pills.remove(pill)
        self._stack.removeWidget(pill)
        pill.deleteLater()

    def _sync_geometry(self) -> None:
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if watched is self.parentWidget() and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            self._sync_geometry()
        return super().eventFilter(watched, event)
