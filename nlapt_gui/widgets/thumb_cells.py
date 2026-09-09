"""Visual primitives for the left file panel: grid cells and list rows.

Also hosts the small shared helpers used across agent-A widgets (painter
icons, warn dots, font builders, token helpers). Every color painted here
comes from :mod:`nlapt_gui.theme.tokens`; the dark photo scrim and the
white overlay text of the design (``rgba(12,13,15,*)`` / ``#fff``) are
derived from the 墨黑 background and 明亮 on-accent tokens, which carry
exactly those values.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QPointF, QRect, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QWidget

from nlapt.core.errors import NLaptError
from nlapt.diagnostics import get_logger

from nlapt_gui.controller import AppController
from nlapt_gui.settings import UISettings
from nlapt_gui.theme.tokens import (
    DEFAULT_THEME,
    FONT_STACK,
    MONO_STACK,
    THEMES,
    ThemeTokens,
    accent_soft,
    with_accent,
)
from nlapt_gui.widgets.thumbnails import ThumbnailLoader

_LOGGER = get_logger(__name__)

TokensProvider = Callable[[], ThemeTokens]


def cover_source_rect(pixmap: QPixmap, target: QRectF) -> QRect:
    """Return the centered source crop needed to cover ``target``."""
    if pixmap.isNull() or target.width() <= 0 or target.height() <= 0:
        return QRect()
    source_ratio = pixmap.width() / pixmap.height()
    target_ratio = target.width() / target.height()
    if source_ratio > target_ratio:
        source_h = pixmap.height()
        source_w = max(1, round(source_h * target_ratio))
        left = (pixmap.width() - source_w) // 2
        return QRect(left, 0, source_w, source_h)
    source_w = pixmap.width()
    source_h = max(1, round(source_w / target_ratio))
    top = (pixmap.height() - source_h) // 2
    return QRect(0, top, source_w, source_h)

# The design's photo-overlay colors are theme independent; these theme rows
# carry exactly rgb(12,13,15) and pure white, so we source them from tokens.
_SCRIM_BASE_THEME = "墨黑"
_OVERLAY_TEXT_THEME = "明亮"

# -- grid cell metrics (design 文件列表 grid) --------------------------------------
GLOW_MARGIN = 3  # in-widget space reserved for the current-item soft glow
GLOW_WIDTH = 3.0
GRID_BORDER_W = 2.0
GRID_RADIUS = 9.0
GRID_CHECKBOX_SIZE = 17
GRID_CHECKBOX_RADIUS = 5.0
GRID_CHECKBOX_MARGIN = 6
GRID_DIRTY_DOT = 8
GRID_DIRTY_MARGIN = 9
GRID_DIRTY_RING_W = 2.5
GRID_OVERLAY_H = 36
GRID_OVERLAY_PAD_X = 7
GRID_OVERLAY_PAD_BOTTOM = 5
GRID_NAME_PX = 10.5
GRID_SEGS_PX = 9.5
SCRIM_ALPHA = 0.62
DIRTY_RING_ALPHA = 0.35
SEGS_TEXT_ALPHA = 0.85
CHECKBOX_BG_ALPHA = 0.78
CHECKBOX_BORDER_W = 1.5

# -- list row metrics (design list mode) ------------------------------------------
LIST_ROW_HEIGHT = 62
LIST_BORDER_W = 1.5
LIST_RADIUS = 8.0
LIST_PAD_X = 7
LIST_CHECKBOX_SIZE = 15
LIST_CHECKBOX_RADIUS = 4.0
LIST_THUMB_W = 34
LIST_THUMB_H = 44
LIST_THUMB_RADIUS = 6.0
LIST_GAP = 8
LIST_NAME_PX = 11
LIST_PREVIEW_PX = 10.5
LIST_SEGS_PX = 9.5
LIST_DIRTY_DOT = 6
LIST_PREVIEW_SEGS = 4

# Suffix appended to segment counts ("6段").
SEGS_SUFFIX = "段"

_ICON_UNIT = 16.0  # design icons use 16x16 viewBoxes
_ICON_DPR = 2.0


def tokens_for_settings(settings: UISettings) -> ThemeTokens:
    """Resolve the persisted theme/accent into ThemeTokens (never raises)."""
    base = THEMES.get(settings.theme, THEMES[DEFAULT_THEME])
    if not settings.accent or settings.accent == base.accent:
        return base
    try:
        return with_accent(base, settings.accent)
    except NLaptError:
        _LOGGER.warning("invalid persisted accent %r; using theme accent", settings.accent)
        return base


def scrim_color(alpha: float) -> QColor:
    """The design's near-black photo scrim (rgb(12,13,15)) at ``alpha``."""
    color = QColor(THEMES[_SCRIM_BASE_THEME].bg)
    color.setAlphaF(alpha)
    return color


def overlay_text_color(alpha: float = 1.0) -> QColor:
    """The design's white overlay text (#fff) at ``alpha``."""
    color = QColor(THEMES[_OVERLAY_TEXT_THEME].onaccent)
    color.setAlphaF(alpha)
    return color


def mono_font(px: float, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """Monospace UI font at a pixel size (design 'IBM Plex Mono' stack)."""
    font = QFont()
    font.setFamilies(list(MONO_STACK))
    font.setPixelSize(round(px))
    font.setWeight(weight)
    return font


def ui_font(px: float, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    """Sans UI font at a pixel size (design 'IBM Plex Sans' stack)."""
    font = QFont()
    font.setFamilies(list(FONT_STACK))
    font.setPixelSize(round(px))
    font.setWeight(weight)
    return font


# ---------------------------------------------------------------------------------
# painter icons (all colors provided by the caller from ThemeTokens)
# ---------------------------------------------------------------------------------

def _pen(color: QColor, width: float) -> QPen:
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _paint_folder(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.4))
    body = QPainterPath()
    body.addRoundedRect(QRectF(1.5, 4.5, 13.0, 9.0), 1.5, 1.5)
    p.drawPath(body)
    tab = QPainterPath(QPointF(2.5, 4.5))
    tab.lineTo(3.0, 2.5)
    tab.lineTo(6.0, 2.5)
    tab.lineTo(7.5, 4.5)
    p.drawPath(tab)


def _paint_folder_open(p: QPainter, color: QColor) -> None:
    _paint_folder(p, color)
    p.setPen(_pen(color, 1.5))
    p.drawLine(QPointF(8.0, 7.0), QPointF(8.0, 11.0))
    p.drawLine(QPointF(6.0, 9.0), QPointF(10.0, 9.0))


def _paint_refresh(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.5))
    p.drawArc(QRectF(2.5, 2.5, 11.0, 11.0), 60 * 16, 300 * 16)
    p.drawLine(QPointF(13.5, 1.5), QPointF(13.5, 4.5))
    p.drawLine(QPointF(13.5, 4.5), QPointF(10.5, 4.5))


def _paint_search(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.5))
    p.drawEllipse(QRectF(2.5, 2.5, 9.0, 9.0))
    p.drawLine(QPointF(10.5, 10.5), QPointF(14.0, 14.0))


def _paint_undo(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.5))
    path = QPainterPath(QPointF(2.5, 6.5))
    path.lineTo(9.5, 6.5)
    path.arcTo(QRectF(5.5, 6.5, 8.0, 8.0), 90.0, -180.0)
    path.lineTo(6.0, 14.5)
    p.drawPath(path)
    p.drawLine(QPointF(5.5, 3.5), QPointF(2.5, 6.5))
    p.drawLine(QPointF(2.5, 6.5), QPointF(5.5, 9.5))


def _paint_chevron_left(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.6))
    p.drawLine(QPointF(10.0, 3.0), QPointF(5.0, 8.0))
    p.drawLine(QPointF(5.0, 8.0), QPointF(10.0, 13.0))


def _paint_chevron_right(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.6))
    p.drawLine(QPointF(6.0, 3.0), QPointF(11.0, 8.0))
    p.drawLine(QPointF(11.0, 8.0), QPointF(6.0, 13.0))


def _paint_copy(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.4))
    p.drawRoundedRect(QRectF(5.0, 5.0, 8.5, 8.5), 1.5, 1.5)
    back = QPainterPath(QPointF(11.0, 5.0))
    back.lineTo(11.0, 2.0)
    back.lineTo(2.0, 2.0)
    back.lineTo(2.0, 11.0)
    back.lineTo(5.0, 11.0)
    p.drawPath(back)


def _paint_save(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.5))
    outline = QPainterPath(QPointF(2.5, 2.5))
    outline.lineTo(11.0, 2.5)
    outline.lineTo(13.5, 5.0)
    outline.lineTo(13.5, 13.5)
    outline.lineTo(2.5, 13.5)
    outline.closeSubpath()
    p.drawPath(outline)
    p.drawLine(QPointF(5.0, 2.5), QPointF(5.0, 6.0))
    p.drawLine(QPointF(5.0, 6.0), QPointF(10.0, 6.0))
    p.drawLine(QPointF(10.0, 6.0), QPointF(10.0, 2.5))
    p.drawLine(QPointF(5.0, 13.5), QPointF(5.0, 9.5))
    p.drawLine(QPointF(5.0, 9.5), QPointF(11.0, 9.5))
    p.drawLine(QPointF(11.0, 9.5), QPointF(11.0, 13.5))


def _paint_view_list(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.5))
    for y in (4.0, 8.0, 12.0):
        p.drawLine(QPointF(3.0, y), QPointF(13.0, y))


def _paint_view_mid(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.4))
    for x in (3.0, 8.8):
        for y in (3.0, 8.8):
            p.drawRoundedRect(QRectF(x, y, 4.2, 4.2), 1.0, 1.0)


def _paint_view_big(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.4))
    p.drawRoundedRect(QRectF(3.0, 3.0, 10.0, 10.0), 1.5, 1.5)


def _paint_save_all(p: QPainter, color: QColor) -> None:
    _paint_save(p, color)
    p.setPen(_pen(color, 1.4))
    p.drawLine(QPointF(6.5, 10.8), QPointF(9.5, 10.8))


def _paint_export(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.5))
    p.drawLine(QPointF(8.0, 2.5), QPointF(8.0, 10.0))
    p.drawLine(QPointF(5.0, 5.5), QPointF(8.0, 2.5))
    p.drawLine(QPointF(11.0, 5.5), QPointF(8.0, 2.5))
    p.drawLine(QPointF(3.0, 12.5), QPointF(13.0, 12.5))
    p.drawLine(QPointF(3.0, 12.5), QPointF(3.0, 10.0))
    p.drawLine(QPointF(13.0, 12.5), QPointF(13.0, 10.0))


def _paint_redo(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.5))
    path = QPainterPath(QPointF(13.5, 6.5))
    path.lineTo(6.5, 6.5)
    path.arcTo(QRectF(2.5, 6.5, 8.0, 8.0), 90.0, 180.0)
    path.lineTo(10.0, 14.5)
    p.drawPath(path)
    p.drawLine(QPointF(10.5, 3.5), QPointF(13.5, 6.5))
    p.drawLine(QPointF(13.5, 6.5), QPointF(10.5, 9.5))


def _paint_tools(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.5))
    p.drawRoundedRect(QRectF(2.5, 7.2, 11.0, 3.6), 1.2, 1.2)
    p.drawLine(QPointF(5.5, 7.2), QPointF(5.5, 4.0))
    p.drawLine(QPointF(5.5, 10.8), QPointF(5.5, 13.5))
    p.drawLine(QPointF(10.5, 7.2), QPointF(10.5, 4.0))
    p.drawLine(QPointF(10.5, 10.8), QPointF(10.5, 13.5))


def _paint_settings(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.5))
    p.drawEllipse(QRectF(5.2, 5.2, 5.6, 5.6))
    p.drawEllipse(QRectF(2.2, 2.2, 11.6, 11.6))


def _paint_apps(p: QPainter, color: QColor) -> None:
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    for row in (2.4, 6.6, 10.8):
        for col in (2.4, 6.6, 10.8):
            p.drawEllipse(QRectF(col, row, 2.8, 2.8))


def _paint_theme(p: QPainter, color: QColor) -> None:
    p.setPen(_pen(color, 1.5))
    p.drawEllipse(QRectF(4.5, 4.5, 7.0, 7.0))
    rays = (
        (8.0, 1.4, 8.0, 2.8),
        (8.0, 13.2, 8.0, 14.6),
        (1.4, 8.0, 2.8, 8.0),
        (13.2, 8.0, 14.6, 8.0),
        (3.3, 3.3, 4.3, 4.3),
        (11.7, 11.7, 12.7, 12.7),
        (11.7, 3.3, 12.7, 4.3),
        (3.3, 11.7, 4.3, 12.7),
    )
    for x1, y1, x2, y2 in rays:
        p.drawLine(QPointF(x1, y1), QPointF(x2, y2))


_ICON_PAINTERS: dict[str, Callable[[QPainter, QColor], None]] = {
    "folder": _paint_folder,
    "folder_open": _paint_folder_open,
    "refresh": _paint_refresh,
    "search": _paint_search,
    "undo": _paint_undo,
    "redo": _paint_redo,
    "chevron_left": _paint_chevron_left,
    "chevron_right": _paint_chevron_right,
    "copy": _paint_copy,
    "save": _paint_save,
    "save_all": _paint_save_all,
    "export": _paint_export,
    "tools": _paint_tools,
    "apps": _paint_apps,
    "settings": _paint_settings,
    "theme": _paint_theme,
    "view_list": _paint_view_list,
    "view_mid": _paint_view_mid,
    "view_big": _paint_view_big,
}

ICON_KINDS: tuple[str, ...] = tuple(_ICON_PAINTERS)


def make_icon(kind: str, color: str | QColor, size: int = 16) -> QIcon:
    """Painter-drawn line icon in one token color (design SVG equivalents)."""
    painter_fn = _ICON_PAINTERS[kind]
    pix = QPixmap(int(size * _ICON_DPR), int(size * _ICON_DPR))
    pix.setDevicePixelRatio(_ICON_DPR)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(size / _ICON_UNIT, size / _ICON_UNIT)
    painter_fn(painter, QColor(color))
    painter.end()
    return QIcon(pix)


def _draw_check(p: QPainter, box: QRectF, color: QColor, width: float) -> None:
    """Check mark polyline scaled into ``box`` (design M2.5 7.5l3 3 6-7)."""
    p.setPen(_pen(color, width))
    unit = box.width() / 14.0
    a = QPointF(box.left() + 2.5 * unit, box.top() + 7.5 * unit)
    b = QPointF(box.left() + 5.5 * unit, box.top() + 10.5 * unit)
    c = QPointF(box.left() + 11.5 * unit, box.top() + 3.5 * unit)
    p.drawLine(a, b)
    p.drawLine(b, c)


class WarnDot(QWidget):
    """Small round 未保存 indicator dot in the warn token color."""

    def __init__(
        self, diameter: int, tokens: TokensProvider, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._diameter = diameter
        self._tokens = tokens
        self.setFixedSize(diameter, diameter)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._tokens().warn))
        painter.drawEllipse(QRectF(0, 0, self._diameter, self._diameter))
        painter.end()


class _BaseCell(QWidget):
    """Shared behavior: controller-driven click semantics + thumb loading."""

    def __init__(
        self,
        key: str,
        controller: AppController,
        loader: ThumbnailLoader,
        tokens: TokensProvider,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._key = key
        self._controller = controller
        self._loader = loader
        self._tokens = tokens
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        loader.ready.connect(self._on_thumb_ready)

    @property
    def key(self) -> str:
        return self._key

    def checkbox_rect(self) -> QRect:
        raise NotImplementedError

    def _thumb(self, target_h: int) -> QPixmap | None:
        """Cached thumb or async request; None until the pixmap is ready."""
        try:
            path = self._controller.image_path(self._key)
        except NLaptError:
            return None
        return self._loader.request(self._key, path, target_h)

    def _on_thumb_ready(self, key: str, _pix: QPixmap) -> None:
        if key == self._key:
            self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        controller = self._controller
        if self.checkbox_rect().contains(event.position().toPoint()):
            controller.toggle_selected(self._key)  # design: stopPropagation
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            controller.select_range_to(self._key)
        elif event.modifiers() & Qt.KeyboardModifier.AltModifier:
            controller.deselect(self._key)
        else:
            controller.set_current(self._key)
            controller.set_anchor(self._key)
        event.accept()

    # -- shared paint helpers -------------------------------------------------------
    def _paint_glow(self, painter: QPainter, frame: QRectF, radius: float) -> None:
        soft = QColor(accent_soft(self._tokens()))
        painter.setPen(_pen(soft, GLOW_WIDTH))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        offset = GLOW_WIDTH / 2.0
        painter.drawRoundedRect(
            frame.adjusted(-offset, -offset, offset, offset), radius + offset, radius + offset
        )

    def _paint_checkbox(self, painter: QPainter, box: QRectF, radius: float) -> None:
        tokens = self._tokens()
        checked = self._controller.is_selected(self._key)
        fill = QColor(tokens.accent)
        if not checked:
            fill = QColor(tokens.surface)
            fill.setAlphaF(CHECKBOX_BG_ALPHA)
        border = QColor(tokens.accent if checked else tokens.bd2)
        painter.setPen(_pen(border, CHECKBOX_BORDER_W))
        painter.setBrush(fill)
        painter.drawRoundedRect(box, radius, radius)
        if checked:
            _draw_check(painter, box.adjusted(2, 2, -2, -2), QColor(tokens.onaccent), 1.8)


class ThumbCell(_BaseCell):
    """Grid cell: 3:4 cover thumbnail + name overlay + checkbox + dirty dot."""

    def checkbox_rect(self) -> QRect:
        offset = GLOW_MARGIN + GRID_CHECKBOX_MARGIN
        return QRect(offset, offset, GRID_CHECKBOX_SIZE, GRID_CHECKBOX_SIZE)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        tokens = self._tokens()
        margin = GLOW_MARGIN
        frame = QRectF(self.rect()).adjusted(margin, margin, -margin, -margin)
        is_current = self._controller.current_key == self._key

        if is_current:
            self._paint_glow(painter, frame, GRID_RADIUS)

        # thumbnail (cover-cropped) clipped to the rounded frame
        inner = frame.adjusted(GRID_BORDER_W, GRID_BORDER_W, -GRID_BORDER_W, -GRID_BORDER_W)
        clip = QPainterPath()
        clip.addRoundedRect(inner, GRID_RADIUS - 1.0, GRID_RADIUS - 1.0)
        painter.save()
        painter.setClipPath(clip)
        painter.fillRect(inner, QColor(tokens.surface2))
        pix = self._thumb(max(1, int(inner.height())))
        if pix is not None and not pix.isNull():
            painter.drawPixmap(inner, pix, cover_source_rect(pix, inner))

        # bottom gradient overlay + name + segment count
        overlay = QRectF(
            inner.x(), inner.bottom() - GRID_OVERLAY_H, inner.width(), GRID_OVERLAY_H
        )
        gradient = QLinearGradient(overlay.topLeft(), overlay.bottomLeft())
        gradient.setColorAt(0.0, scrim_color(0.0))
        gradient.setColorAt(1.0, scrim_color(SCRIM_ALPHA))
        painter.fillRect(overlay, gradient)

        segments = self._controller.segments(self._key)
        segs_text = f"{len(segments)}{SEGS_SUFFIX}"
        painter.setFont(mono_font(GRID_SEGS_PX))
        segs_w = QFontMetrics(painter.font()).horizontalAdvance(segs_text)
        baseline = inner.bottom() - GRID_OVERLAY_PAD_BOTTOM
        painter.setPen(overlay_text_color(SEGS_TEXT_ALPHA))
        painter.drawText(QPointF(inner.right() - GRID_OVERLAY_PAD_X - segs_w, baseline), segs_text)

        name = self._key.rsplit("/", 1)[-1]
        painter.setFont(mono_font(GRID_NAME_PX))
        name_w = int(inner.width()) - 2 * GRID_OVERLAY_PAD_X - segs_w - 4
        elided = QFontMetrics(painter.font()).elidedText(
            name, Qt.TextElideMode.ElideRight, max(8, name_w)
        )
        painter.setPen(overlay_text_color())
        painter.drawText(QPointF(inner.x() + GRID_OVERLAY_PAD_X, baseline), elided)
        painter.restore()

        # 2px border (accent when current, otherwise none per design)
        if is_current:
            painter.setPen(_pen(QColor(tokens.accent), GRID_BORDER_W))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            half = GRID_BORDER_W / 2.0
            painter.drawRoundedRect(
                frame.adjusted(half, half, -half, -half), GRID_RADIUS, GRID_RADIUS
            )

        self._paint_checkbox(painter, QRectF(self.checkbox_rect()), GRID_CHECKBOX_RADIUS)

        if self._controller.record(self._key).dirty:
            dot = QRectF(
                frame.right() - GRID_DIRTY_MARGIN - GRID_DIRTY_DOT,
                frame.top() + GRID_DIRTY_MARGIN,
                GRID_DIRTY_DOT,
                GRID_DIRTY_DOT,
            )
            painter.setPen(_pen(scrim_color(DIRTY_RING_ALPHA), GRID_DIRTY_RING_W))
            painter.setBrush(QColor(tokens.warn))
            painter.drawEllipse(dot)
        painter.end()


class ListRow(_BaseCell):
    """List row: checkbox + 34x44 thumb + name/dirty + caption preview + 段 count."""

    def __init__(
        self,
        key: str,
        controller: AppController,
        loader: ThumbnailLoader,
        tokens: TokensProvider,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(key, controller, loader, tokens, parent)
        self.setFixedHeight(LIST_ROW_HEIGHT)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def _frame(self) -> QRectF:
        margin = GLOW_MARGIN
        return QRectF(self.rect()).adjusted(margin, margin, -margin, -margin)

    def checkbox_rect(self) -> QRect:
        frame = self._frame()
        y = frame.center().y() - LIST_CHECKBOX_SIZE / 2.0
        return QRect(
            int(frame.x() + LIST_PAD_X), int(y), LIST_CHECKBOX_SIZE, LIST_CHECKBOX_SIZE
        )

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        tokens = self._tokens()
        frame = self._frame()
        is_current = self._controller.current_key == self._key

        if is_current:
            self._paint_glow(painter, frame, LIST_RADIUS)

        if is_current or self.underMouse():
            border = QColor(tokens.accent)
        else:
            border = QColor(tokens.bd)
        painter.setPen(_pen(border, LIST_BORDER_W))
        painter.setBrush(QColor(tokens.surface))
        half = LIST_BORDER_W / 2.0
        painter.drawRoundedRect(frame.adjusted(half, half, -half, -half), LIST_RADIUS, LIST_RADIUS)

        self._paint_checkbox(painter, QRectF(self.checkbox_rect()), LIST_CHECKBOX_RADIUS)

        # thumbnail
        thumb = QRectF(
            self.checkbox_rect().right() + LIST_GAP,
            frame.center().y() - LIST_THUMB_H / 2.0,
            LIST_THUMB_W,
            LIST_THUMB_H,
        )
        clip = QPainterPath()
        clip.addRoundedRect(thumb, LIST_THUMB_RADIUS, LIST_THUMB_RADIUS)
        painter.save()
        painter.setClipPath(clip)
        painter.fillRect(thumb, QColor(tokens.surface2))
        pix = self._thumb(LIST_THUMB_H)
        if pix is not None and not pix.isNull():
            painter.drawPixmap(thumb, pix, cover_source_rect(pix, thumb))
        painter.restore()

        # right-aligned segment count
        segments = self._controller.segments(self._key)
        segs_text = f"{len(segments)}{SEGS_SUFFIX}"
        painter.setFont(mono_font(LIST_SEGS_PX))
        segs_metrics = QFontMetrics(painter.font())
        segs_w = segs_metrics.horizontalAdvance(segs_text)
        painter.setPen(QColor(tokens.text3))
        painter.drawText(
            QPointF(
                frame.right() - LIST_PAD_X - segs_w,
                frame.center().y() + segs_metrics.ascent() / 2.0 - 1.0,
            ),
            segs_text,
        )

        # name + dirty dot + caption preview
        text_x = thumb.right() + LIST_GAP
        text_w = frame.right() - LIST_PAD_X - segs_w - 6 - text_x
        name = self._key.rsplit("/", 1)[-1]
        dirty = self._controller.record(self._key).dirty
        painter.setFont(mono_font(LIST_NAME_PX, QFont.Weight.DemiBold))
        name_metrics = QFontMetrics(painter.font())
        dot_space = LIST_DIRTY_DOT + 5 if dirty else 0
        elided = name_metrics.elidedText(
            name, Qt.TextElideMode.ElideRight, max(8, int(text_w) - dot_space)
        )
        name_baseline = frame.center().y() - 4.0
        painter.setPen(QColor(tokens.text))
        painter.drawText(QPointF(text_x, name_baseline), elided)
        if dirty:
            dot_x = text_x + name_metrics.horizontalAdvance(elided) + 5
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(tokens.warn))
            painter.drawEllipse(
                QRectF(dot_x, name_baseline - LIST_DIRTY_DOT + 1, LIST_DIRTY_DOT, LIST_DIRTY_DOT)
            )

        preview = ", ".join(segments[:LIST_PREVIEW_SEGS])
        painter.setFont(ui_font(LIST_PREVIEW_PX))
        preview_metrics = QFontMetrics(painter.font())
        painter.setPen(QColor(tokens.text3))
        painter.drawText(
            QPointF(text_x, name_baseline + preview_metrics.height() + 1.0),
            preview_metrics.elidedText(preview, Qt.TextElideMode.ElideRight, max(8, int(text_w))),
        )
        painter.end()

    def event(self, ev) -> bool:  # noqa: ANN001 - Qt override
        if ev.type() in (ev.Type.HoverEnter, ev.Type.HoverLeave):
            self.update()
        return super().event(ev)
