"""Single-image, compare-grid, and preview-header rendering widgets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt, QVariantAnimation
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import QFrame, QWidget

from nlapt_gui import anim
from nlapt_gui.theme.tokens import accent_soft
from nlapt_gui.widgets.thumb_cells import (
    mono_font,
    overlay_text_color,
    scrim_color,
    ui_font,
)
from nlapt_gui.widgets.window_chrome import WindowDragHelper

if TYPE_CHECKING:
    from nlapt_gui.widgets.preview_panel import PreviewPanel

PREVIEW_PAD = 22
IMAGE_RADIUS = 10.0
MULTI_RADIUS = 12.0
MULTI_BORDER_W = 2.0
MULTI_GLOW_W = 3.0
BADGE_MARGIN = 10
DIRTY_DOT_PX = 8
DIRTY_RING_W = 2.5
DIRTY_RING_ALPHA = 0.35
NAME_PILL_ALPHA = 0.55
WHEEL_NOTCH = 120  # one physical wheel notch == one ZOOM_STEP (finer devices accumulate)
IMAGE_FADE_MS = 170  # cross-fade on image change (skip-safe: content is correct meanwhile)
IMAGE_FADE_FROM = 0.55  # starting opacity of the fade-in (fully visible content, just faint)
TEXT_EDITING = "编辑中"


class _SingleView(QWidget):
    """Centered single-image view with cursor-anchored zoom and drag-to-pan.

    The widget lives inside the panel's non-resizable :class:`QScrollArea`; the
    panel drives its size via :meth:`sync_size` (viewport-sized at 适应, or the
    scaled image plus padding when zoomed so the scroll bars become the pan
    surface). Wheel notches accumulate so high-resolution devices feel smooth
    while a plain notch still equals one ``ZOOM_STEP``.
    """

    def __init__(self, panel: "PreviewPanel", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._panel = panel
        self._pixmap: QPixmap | None = None
        self._source_size: QSize | None = None
        self._zoom: int | None = None  # None = fit
        self._wheel_accum = 0
        self._pan_origin: QPointF | None = None
        self._pan_h0 = 0
        self._pan_v0 = 0
        # Paint-level fade alpha (1.0 = fully opaque). Deliberately NOT a
        # QGraphicsOpacityEffect: a graphics effect on a view that is resized
        # with the window re-renders through an effect buffer and hard-crashes
        # Qt during native resizes.
        self._fade_alpha = 1.0
        self._fade_anim: QVariantAnimation | None = None

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        changed = pixmap is not self._pixmap
        self._pixmap = pixmap
        if (
            pixmap is not None
            and self._source_size is not None
            and (pixmap.width() > self._source_size.width() * 2
                 or pixmap.height() > self._source_size.height() * 2)
        ):
            self._source_size = pixmap.size()
        if changed and pixmap is not None and not pixmap.isNull() and self._zoom is None:
            self._start_fade()
        self.update()

    def set_source_size(self, size: QSize | None) -> None:
        """Set the original image geometry used while a coarse frame is shown."""
        self._source_size = size
        self.update()

    def set_zoom(self, zoom: int | None) -> None:
        self._zoom = zoom
        self.update()

    # -- cross-fade on image change (skip-safe) ----------------------------------------
    def _start_fade(self) -> None:
        """Fade the freshly-loaded image in via painter opacity.

        Skip-safe: the new pixmap is already the painted content, so even if no
        event loop advances the animation the image is correct (just a touch
        faint). :meth:`finish_image_fade` jumps straight to the end state.
        """
        if not self.isVisible():
            return
        if not anim.animations_enabled():
            self.finish_image_fade()
            return
        if self._fade_anim is not None:
            self._fade_anim.stop()
        self._fade_alpha = float(IMAGE_FADE_FROM)
        fade = QVariantAnimation(self)
        fade.setStartValue(float(IMAGE_FADE_FROM))
        fade.setEndValue(1.0)
        fade.setDuration(IMAGE_FADE_MS)
        fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        fade.valueChanged.connect(self._on_fade_value)
        fade.finished.connect(self.finish_image_fade)
        self._fade_anim = fade
        fade.start()

    def _on_fade_value(self, value: object) -> None:
        self._fade_alpha = float(value)  # type: ignore[arg-type]
        self.update()

    def finish_image_fade(self) -> None:
        """Jump the fade to its end state synchronously (tests + safety net)."""
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim = None
        self._fade_alpha = 1.0
        self.update()

    # -- sizing ------------------------------------------------------------------------
    def fit_percent(self) -> int:
        """Effective fit percentage (100 when unknown; never upscales)."""
        if self._pixmap is None or self._pixmap.isNull():
            return 100
        avail_w = self.width() - 2 * PREVIEW_PAD
        avail_h = self.height() - 2 * PREVIEW_PAD
        if avail_w <= 0 or avail_h <= 0:
            return 100
        source = self._source_size or self._pixmap.size()
        scale = min(1.0, avail_w / source.width(), avail_h / source.height())
        return max(1, round(scale * 100))

    def _scaled_size(self) -> QSize | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        scale = (self._zoom / 100.0) if self._zoom is not None else self.fit_percent() / 100.0
        source = self._source_size or self._pixmap.size()
        return QSize(
            max(1, round(source.width() * scale)),
            max(1, round(source.height() * scale)),
        )

    def _content_size(self, viewport: QSize) -> QSize:
        """Widget size inside the scroll area: viewport at 适应, else scaled+pad."""
        vw = max(1, viewport.width())
        vh = max(1, viewport.height())
        scaled = self._scaled_size()
        if self._zoom is None or scaled is None:
            return QSize(vw, vh)
        return QSize(
            max(vw, scaled.width() + 2 * PREVIEW_PAD),
            max(vh, scaled.height() + 2 * PREVIEW_PAD),
        )

    def sync_size(self, viewport: QSize) -> None:
        """Resize to match the current zoom against ``viewport`` (panel-driven)."""
        size = self._content_size(viewport)
        if size != self.size():
            self.resize(size)
        self._update_cursor()
        self.update()

    def image_rect(self) -> QRectF | None:
        """Where the image is painted, in this widget's own coordinates."""
        size = self._scaled_size()
        if size is None:
            return None
        return QRectF(
            (self.width() - size.width()) / 2.0,
            (self.height() - size.height()) / 2.0,
            float(size.width()),
            float(size.height()),
        )

    def _update_cursor(self) -> None:
        if self._pan_origin is not None:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._panel.single_pannable():
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    # -- interaction -------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        """Cursor-anchored wheel zoom over the single image (up = in, down = out).

        Notches accumulate so trackpads step smoothly; a full ``WHEEL_NOTCH``
        still equals one ``ZOOM_STEP`` (from 适应 the first step lands on 120%).
        Ignored while the compare grid is showing.
        """
        if self._panel.controller.multi_mode():
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        self._wheel_accum += delta
        steps = int(self._wheel_accum / WHEEL_NOTCH)
        self._wheel_accum -= steps * WHEEL_NOTCH
        if steps != 0:
            self._panel.wheel_zoom(steps, event.position())
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        """Toggle 适应 <-> 100% on the single image (design 双击切换)."""
        if event.button() == Qt.MouseButton.LeftButton and not self._panel.controller.multi_mode():
            self._panel.toggle_actual_size(event.position())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton and self._panel.single_pannable():
            hbar, vbar = self._panel.single_scrollbars()
            self._pan_origin = event.globalPosition()
            self._pan_h0 = hbar.value()
            self._pan_v0 = vbar.value()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._pan_origin is not None:
            hbar, vbar = self._panel.single_scrollbars()
            delta = event.globalPosition() - self._pan_origin
            hbar.setValue(round(self._pan_h0 - delta.x()))
            vbar.setValue(round(self._pan_v0 - delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._pan_origin is not None and event.button() == Qt.MouseButton.LeftButton:
            self._pan_origin = None
            self._update_cursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        target = self.image_rect()
        if target is None or self._pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self._fade_alpha < 1.0:
            painter.setOpacity(self._fade_alpha)
        clip = QPainterPath()
        clip.addRoundedRect(target, IMAGE_RADIUS, IMAGE_RADIUS)
        painter.setClipPath(clip)
        painter.drawPixmap(target.toRect(), self._pixmap)
        painter.end()


class _MultiCell(QWidget):
    """One tile of the 2x2 compare grid (contain-fit, badges, click to edit)."""

    def __init__(self, key: str, panel: "PreviewPanel", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self._panel = panel
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(40, 40)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._panel.controller.set_current(self.key)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        """Open this tile as the standalone large single preview.

        Composes existing controller methods only: focus this image
        (``set_current``) and collapse the selection (``clear_selection``)
        so ``multi_mode()`` becomes false and the single large view shows it.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            controller = self._panel.controller
            controller.set_current(self.key)
            controller.clear_selection()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        tokens = self._panel.current_tokens()
        controller = self._panel.controller
        is_current = controller.current_key == self.key
        margin = MULTI_GLOW_W
        frame = QRectF(self.rect()).adjusted(margin, margin, -margin, -margin)

        if is_current:
            soft = QColor(accent_soft(tokens))
            pen = QPen(soft, MULTI_GLOW_W)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            offset = MULTI_GLOW_W / 2.0
            painter.drawRoundedRect(
                frame.adjusted(-offset, -offset, offset, offset),
                MULTI_RADIUS + offset,
                MULTI_RADIUS + offset,
            )

        border = QColor(tokens.accent if is_current else tokens.bd)
        painter.setPen(QPen(border, MULTI_BORDER_W))
        painter.setBrush(QColor(tokens.panel))
        half = MULTI_BORDER_W / 2.0
        painter.drawRoundedRect(frame.adjusted(half, half, -half, -half), MULTI_RADIUS, MULTI_RADIUS)

        inner = frame.adjusted(MULTI_BORDER_W, MULTI_BORDER_W, -MULTI_BORDER_W, -MULTI_BORDER_W)
        clip = QPainterPath()
        clip.addRoundedRect(inner, MULTI_RADIUS - 1.0, MULTI_RADIUS - 1.0)
        painter.save()
        painter.setClipPath(clip)
        pix = self._panel.pixmap_for(self.key)
        if pix is not None and not pix.isNull():
            scaled = pix.scaled(
                inner.size().toSize(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(
                QPointF(
                    inner.x() + (inner.width() - scaled.width()) / 2.0,
                    inner.y() + (inner.height() - scaled.height()) / 2.0,
                ),
                scaled,
            )

        # name pill (bottom-left, mono white on scrim)
        name = self.key.rsplit("/", 1)[-1]
        painter.setFont(mono_font(10.5))
        metrics = QFontMetrics(painter.font())
        pill_h = metrics.height() + 5
        pill_w = metrics.horizontalAdvance(name) + 18
        pill = QRectF(
            inner.x() + BADGE_MARGIN, inner.bottom() - 8 - pill_h, pill_w, pill_h
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(scrim_color(NAME_PILL_ALPHA))
        painter.drawRoundedRect(pill, pill_h / 2.0, pill_h / 2.0)
        painter.setPen(overlay_text_color())
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, name)

        # 编辑中 badge (top-right) on the current tile
        if is_current:
            painter.setFont(ui_font(10, QFont.Weight.Bold))
            badge_metrics = QFontMetrics(painter.font())
            badge_h = badge_metrics.height() + 5
            badge_w = badge_metrics.horizontalAdvance(TEXT_EDITING) + 18
            badge = QRectF(
                inner.right() - BADGE_MARGIN - badge_w, inner.y() + 8, badge_w, badge_h
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(tokens.accent))
            painter.drawRoundedRect(badge, badge_h / 2.0, badge_h / 2.0)
            painter.setPen(QColor(tokens.onaccent))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, TEXT_EDITING)

        # dirty dot (top-left)
        if controller.record(self.key).dirty:
            dot = QRectF(
                inner.x() + BADGE_MARGIN, inner.y() + BADGE_MARGIN, DIRTY_DOT_PX, DIRTY_DOT_PX
            )
            painter.setPen(QPen(scrim_color(DIRTY_RING_ALPHA), DIRTY_RING_W))
            painter.setBrush(QColor(tokens.warn))
            painter.drawEllipse(dot)
        painter.restore()
        painter.end()


class _InfoBar(QFrame):
    """46px preview header: empty space drags the frameless window."""

    def __init__(self, panel: "PreviewPanel") -> None:
        super().__init__(panel)
        self._drag = WindowDragHelper(self)

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
