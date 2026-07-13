"""The NLapt program logo - a crisp vector mark painted with QPainter.

The mark is an accent rounded-square badge holding a small framed image
glyph (a sun over a mountain) with two caption text-lines below it - a
compact, intentional motif for an image-caption / annotation tool. It is
theme-aware: the badge uses the theme's ``accent`` and every foreground
element uses ``onaccent`` so it stays legible in every theme.

The same paint routine backs both the on-screen :class:`LogoWidget` (used in
the title bar) and :func:`make_app_icon` (the window / taskbar icon), so the
two never drift apart.

No color is hardcoded here: every color comes from a :class:`ThemeTokens`
instance passed in by the caller (design mandate - colors live in ``theme/``).
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from nlapt_gui.theme.tokens import ThemeTokens

# Default on-screen size for the title-bar logo (matches the 40px bar).
LOGO_SIZE = 22
# Default rendered icon size (window / taskbar).
ICON_SIZE = 256
# The mark is authored on a 24-unit grid, then scaled to the paint rect.
_GRID = 24.0


def _paint_mark(
    painter: QPainter, rect: QRectF, accent: QColor, onaccent: QColor
) -> None:
    """Paint the logo mark into ``rect`` using ``accent`` / ``onaccent``.

    The geometry is authored on a 24-unit grid and scaled uniformly so the
    mark stays crisp and balanced at any size (22px bar icon .. 256px app
    icon). Everything is centered inside ``rect``.
    """
    side = min(rect.width(), rect.height())
    unit = side / _GRID
    # Center the square mark inside a possibly non-square rect.
    ox = rect.x() + (rect.width() - side) / 2.0
    oy = rect.y() + (rect.height() - side) / 2.0

    def point(x: float, y: float) -> QPointF:
        return QPointF(ox + x * unit, oy + y * unit)

    def box(x: float, y: float, w: float, h: float) -> QRectF:
        return QRectF(ox + x * unit, oy + y * unit, w * unit, h * unit)

    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # 1) Rounded-square accent badge.
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(accent))
    painter.drawRoundedRect(box(1, 1, 22, 22), 6 * unit, 6 * unit)

    # 2) Framed image tile (upper region).
    frame_pen = QPen(onaccent, 1.35 * unit)
    frame_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(frame_pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(box(4.6, 4.2, 14.8, 9.0), 2.1 * unit, 2.1 * unit)

    # 3) Mountain inside the tile (filled onaccent).
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(onaccent))
    mountain = QPainterPath(point(5.6, 12.6))
    mountain.lineTo(point(9.6, 7.6))
    mountain.lineTo(point(12.4, 10.4))
    mountain.lineTo(point(15.6, 6.3))
    mountain.lineTo(point(18.4, 12.6))
    mountain.closeSubpath()
    painter.drawPath(mountain)

    # 4) Sun in the tile's upper-left.
    painter.drawEllipse(point(7.4, 6.9), 1.15 * unit, 1.15 * unit)

    # 5) Two caption text-lines beneath the tile.
    line_pen = QPen(onaccent, 1.7 * unit)
    line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(line_pen)
    painter.drawLine(point(5.2, 16.8), point(17.6, 16.8))
    painter.drawLine(point(5.2, 19.8), point(12.8, 19.8))


class LogoWidget(QWidget):
    """A theme-aware widget painting the NLapt logo mark at a fixed size."""

    def __init__(
        self,
        tokens: ThemeTokens,
        size: int = LOGO_SIZE,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._tokens = tokens
        self._size = size
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def set_tokens(self, tokens: ThemeTokens) -> None:
        """Update the theme tokens and repaint (accent / onaccent aware)."""
        self._tokens = tokens
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(self._size, self._size)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        _paint_mark(
            painter,
            QRectF(0, 0, self.width(), self.height()),
            QColor(self._tokens.accent),
            QColor(self._tokens.onaccent),
        )
        painter.end()


def make_app_icon(tokens: ThemeTokens, size: int = ICON_SIZE) -> QIcon:
    """Render the logo mark to a ``size`` px pixmap and wrap it as a QIcon.

    Used for the window / taskbar icon. The pixmap is transparent outside the
    rounded badge so the icon reads cleanly on any OS chrome.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    _paint_mark(
        painter,
        QRectF(0, 0, size, size),
        QColor(tokens.accent),
        QColor(tokens.onaccent),
    )
    painter.end()
    return QIcon(pixmap)


# Relative path (from the repository/bundle root) of the swappable .ico file.
APP_ICON_REL_PATH = "packaging/icon.ico"


def load_app_icon(tokens: ThemeTokens, size: int = ICON_SIZE) -> QIcon:
    """The application icon: ``packaging/icon.ico`` when present, else the
    painted theme-aware mark.

    The .ico file is the single swap point for a future custom icon (spec
    module 4.6) — dropping a replacement there changes the window, taskbar
    and packaged exe icon without touching code.
    """
    # Imported lazily: resources lives in the GUI package root and importing
    # it at module scope would be a needless cycle risk for theme consumers.
    from nlapt_gui.resources import resource_path

    ico = resource_path(APP_ICON_REL_PATH)
    if ico.is_file():
        icon = QIcon(str(ico))
        if not icon.isNull():
            return icon
    return make_app_icon(tokens, size)
