"""Collapsible section card for the right tools panel.

A surface card (11px radius per design) with a clickable fixed-height header
row (icon + title + optional mono suffix + rotating chevron), a scrollable
content area, and a thin drag grip at the bottom that lets the user set the
section's visible height.

Sizing model (the part that must stay robust):

- The header has a FIXED height (:data:`HEADER_HEIGHT`) so it can never be
  clipped, no matter what the global stylesheet does to QPushButton size
  hints (a button hosting a child layout reports a hint from its own empty
  text, not from the layout - relying on that clipped the title).
- The body lives inside a section-owned frameless ``QScrollArea``. The
  applied height is either the content's natural hint (default) or the
  user-dragged value clamped to ``[MIN_CONTENT_HEIGHT, MAX_CONTENT_HEIGHT]``
  - deliberately independent of the content's minimum size, so dragging
  SHRINKS into a scrollbar and GROWS to reveal more (inner row lists get an
  expanding size policy so extra space shows more rows).
- Expand/collapse is an instant show/hide (zero flicker); the chevron still
  rotates between open (0deg) and closed (-90deg).

The open state is reported through ``toggled`` and the user-set height
through ``height_changed`` so the owner (ToolsPanel) can persist both.
"""

from __future__ import annotations

from PySide6.QtCore import Property, QEvent, QObject, Qt, Signal
from PySide6.QtGui import QMouseEvent, QPainter, QPaintEvent, QPalette, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger

_LOGGER = get_logger(__name__)

# Retained for constructor backward-compatibility; the expand/collapse is an
# instant toggle so no tween duration is actually consumed.
COLLAPSE_DURATION_MS = 240
# Chevron rotation while collapsed (design: rotate(-90deg)).
CLOSED_ROTATION_DEG = -90.0
OPEN_ROTATION_DEG = 0.0
# Fixed header height (design: 12.5px title + 10px vertical padding).
HEADER_HEIGHT = 40
# Header metrics from the design (padding 10px 13px, gap 9).
_HEADER_MARGINS = (13, 0, 13, 0)
_HEADER_GAP = 9
_ARROW_SIZE = 12
_ARROW_PEN_WIDTH = 1.4

# User-resizable visible-height clamp for the body (drag grip).
MIN_CONTENT_HEIGHT = 120
MAX_CONTENT_HEIGHT = 720
# Drag grip metrics.
_GRIP_HEIGHT = 9
_GRIP_LINE_WIDTH = 26
_GRIP_LINE_THICKNESS = 2.0


class ArrowIcon(QWidget):
    """Small chevron that rotates between open (0deg) and closed (-90deg)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rotation = OPEN_ROTATION_DEG
        self.setFixedSize(_ARROW_SIZE, _ARROW_SIZE)

    def _get_rotation(self) -> float:
        return self._rotation

    def _set_rotation(self, value: float) -> None:
        self._rotation = float(value)
        self.update()

    rotation = Property(float, _get_rotation, _set_rotation)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Muted chevron color straight from the themed palette (text3 role).
        color = self.palette().color(QPalette.ColorRole.PlaceholderText)
        pen = QPen(color, _ARROW_PEN_WIDTH)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.rotate(self._rotation)
        half = self.width() / 4
        painter.drawLine(int(-half), int(-half / 2), 0, int(half / 2))
        painter.drawLine(0, int(half / 2), int(half), int(-half / 2))
        painter.end()


class ResizeGrip(QWidget):
    """Thin drag handle at the bottom of an expanded card.

    Emits :attr:`resized` with the vertical pixel delta since the previous
    move event while the primary button is held; the owning section clamps
    and applies the delta to its visible height. Kept intentionally simple
    (raw deltas) so it is trivial to drive from a test.
    """

    resized = Signal(int)  # vertical pixel delta since the last move

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_GRIP_HEIGHT)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._last_y: float | None = None
        self._hover = False

    def event(self, ev: QEvent) -> bool:  # noqa: N802 - Qt override
        if ev.type() == QEvent.Type.HoverEnter:
            self._hover = True
            self.update()
        elif ev.type() == QEvent.Type.HoverLeave:
            self._hover = False
            self.update()
        return super().event(ev)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_y = event.globalPosition().y()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._last_y is not None:
            current = event.globalPosition().y()
            delta = int(round(current - self._last_y))
            if delta:
                self._last_y = current
                self.resized.emit(delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton and self._last_y is not None:
            self._last_y = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        if not self._hover:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.palette().color(QPalette.ColorRole.PlaceholderText)
        pen = QPen(color, _GRIP_LINE_THICKNESS)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        mid_y = self.height() / 2
        x0 = (self.width() - _GRIP_LINE_WIDTH) / 2
        painter.drawLine(int(x0), int(mid_y), int(x0 + _GRIP_LINE_WIDTH), int(mid_y))
        painter.end()


class CollapsibleSection(QFrame):
    """Surface card with an instant expand/collapse and a user-resizable body."""

    toggled = Signal(str, bool)  # section_id, is_open
    height_changed = Signal(str, int)  # section_id, applied body height

    def __init__(
        self,
        section_id: str,
        title: str,
        *,
        icon: QWidget | None = None,
        open: bool = True,  # noqa: A002 - mirrors the settings key semantics
        duration_ms: int = COLLAPSE_DURATION_MS,  # noqa: ARG002 - API compat
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._section_id = section_id
        self._open = open
        # User height override (None = follow the content's natural height).
        self._content_height: int | None = None
        self._content: QWidget | None = None
        self.setProperty("surfaceCard", True)

        # Fixed-height header: immune to QPushButton size-hint quirks, so the
        # title/icon can never be clipped by a stylesheet change.
        self._header = QPushButton(self)
        self._header.setProperty("collapsibleHeader", True)
        self._header.setFixedHeight(HEADER_HEIGHT)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(*_HEADER_MARGINS)
        header_layout.setSpacing(_HEADER_GAP)
        if icon is not None:
            icon.setParent(self._header)
            icon.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            header_layout.addWidget(icon)
        self._title = QLabel(title, self._header)
        self._title.setStyleSheet("font-size: 12.5px; font-weight: 600;")
        self._title.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header_layout.addWidget(self._title)
        self._suffix = QLabel("", self._header)
        self._suffix.setProperty("mono", True)
        self._suffix.setProperty("muted", True)
        self._suffix.setStyleSheet("font-size: 10px;")
        self._suffix.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header_layout.addWidget(self._suffix)
        header_layout.addStretch(1)
        self._arrow = ArrowIcon(self._header)
        self._arrow.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        header_layout.addWidget(self._arrow)
        self._header.clicked.connect(self.toggle)

        # Section-owned scroll viewport: the single authority on body height.
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.viewport().setAutoFillBackground(False)
        self._scroll.setAutoFillBackground(False)

        self._grip = ResizeGrip(self)
        self._grip.resized.connect(self._on_grip_drag)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._header)
        outer.addWidget(self._scroll)
        outer.addWidget(self._grip)

        self._apply_state_instantly()

    # -- public API -------------------------------------------------------------
    @property
    def section_id(self) -> str:
        return self._section_id

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def content_height(self) -> int | None:
        """User-set body height in px, or ``None`` when following content."""
        return self._content_height

    def set_content(self, widget: QWidget) -> None:
        """Install (or replace) the collapsible body widget."""
        old = self._scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        widget.setAutoFillBackground(False)
        # Inner row lists (translate/history) keep their own scrollbars but
        # must be free to grow when the user enlarges the section: lift their
        # hard caps and let them absorb extra vertical space.
        for inner in widget.findChildren(QScrollArea):
            inner.setMaximumHeight(16_777_215)
            policy = inner.sizePolicy()
            policy.setVerticalPolicy(QSizePolicy.Policy.Expanding)
            inner.setSizePolicy(policy)
        self._scroll.setWidget(widget)
        self._content = widget
        widget.installEventFilter(self)
        self._apply_body_height()

    def set_suffix(self, text: str) -> None:
        """Small mono label after the title (e.g. history '{n} 条')."""
        self._suffix.setText(text)

    def set_content_height(self, height: int | None) -> None:
        """Set the visible body height (``None`` = follow the content).

        Clamped to ``[MIN_CONTENT_HEIGHT, MAX_CONTENT_HEIGHT]`` and applied
        through the section's scroll viewport, so it works regardless of the
        content's own minimum size; reported via :attr:`height_changed`.
        """
        clamped = self._clamp_height(height) if height is not None else None
        if clamped == self._content_height:
            return
        self._content_height = clamped
        self._apply_body_height()
        if clamped is not None:
            self.height_changed.emit(self._section_id, clamped)
        _LOGGER.debug("section %s body height -> %s", self._section_id, clamped)

    def toggle(self) -> None:
        self.set_open(not self._open)

    def set_open(self, is_open: bool, *, animate: bool = True) -> None:  # noqa: ARG002
        # ``animate`` is accepted for backward-compatibility; the toggle is
        # always instant (zero flicker, no mid-animation relayout).
        if is_open == self._open:
            return
        self._open = is_open
        self._apply_state_instantly()
        self.toggled.emit(self._section_id, is_open)
        _LOGGER.debug("section %s -> %s", self._section_id, "open" if is_open else "closed")

    # -- internals ---------------------------------------------------------------
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        # Content grew/shrank (rows added, provider switched...): while
        # following the natural height, track it.
        if (
            watched is self._content
            and event.type() == QEvent.Type.LayoutRequest
            and self._content_height is None
        ):
            self._apply_body_height()
        return super().eventFilter(watched, event)

    @staticmethod
    def _clamp_height(height: int) -> int:
        return max(MIN_CONTENT_HEIGHT, min(MAX_CONTENT_HEIGHT, height))

    def _natural_height(self) -> int:
        if self._content is None:
            return MIN_CONTENT_HEIGHT
        return self._content.sizeHint().height()

    def applied_height(self) -> int:
        """The body height currently in effect (user value or natural)."""
        if self._content_height is not None:
            return self._content_height
        return min(MAX_CONTENT_HEIGHT, max(0, self._natural_height()))

    def _apply_body_height(self) -> None:
        if self._open:
            self._scroll.setFixedHeight(self.applied_height())

    def _apply_state_instantly(self) -> None:
        self._arrow.rotation = OPEN_ROTATION_DEG if self._open else CLOSED_ROTATION_DEG
        self._scroll.setVisible(self._open)
        # The drag grip only makes sense while the body is visible.
        self._grip.setVisible(self._open)
        if self._open:
            self._apply_body_height()
        else:
            self._scroll.setFixedHeight(0)

    def _on_grip_drag(self, delta: int) -> None:
        if not self._open or self._content is None:
            return
        self.set_content_height(self.applied_height() + delta)
