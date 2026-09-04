"""Middle column 预览与编辑 info bar + single/multi image preview.

Header (46px): filename, format pill, dim / size / mtime, 未保存 pill,
multi compare pill, prev / next / position, and the frameless ─ □ ×
controls. Preview: single image with fit-or-percent zoom and a floating
zoom pill, or a 2x2 compare grid of the first four selected images with
an 编辑中 badge and an overflow pill. A thin :class:`SplitterHandle` is
provided for the main window to mount between this panel and the editor
area; it emits ``editor_h_changed`` while dragging and persists
``editor_h`` on release.
"""

from __future__ import annotations

from collections import OrderedDict

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
    QThreadPool,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QImageReader,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from shiboken6 import isValid

from nlapt.core.errors import NLaptError
from nlapt.diagnostics import get_logger

from nlapt_gui import anim
from nlapt_gui.controller import AppController, MULTI_PREVIEW_LIMIT
from nlapt_gui.theme.tokens import EDITOR_H_RANGE, ZOOM_RANGE, ZOOM_STEP, ThemeTokens, accent_soft
from nlapt_gui.widgets.thumb_cells import (
    make_icon,
    mono_font,
    overlay_text_color,
    scrim_color,
    tokens_for_settings,
    ui_font,
)
from nlapt_gui.widgets.window_chrome import (
    KIND_CLOSE,
    KIND_MAX,
    KIND_MIN,
    WindowButton,
    WindowDragHelper,
    wire_window_buttons,
)
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

HEADER_H = 46
NAV_BUTTON_PX = 28
POS_MIN_W = 44
DIVIDER_H = 18
PREVIEW_PAD = 22
IMAGE_RADIUS = 10.0
MULTI_GAP = 14
MULTI_RADIUS = 12.0
MULTI_BORDER_W = 2.0
MULTI_GLOW_W = 3.0
BADGE_MARGIN = 10
DIRTY_DOT_PX = 8
DIRTY_RING_W = 2.5
DIRTY_RING_ALPHA = 0.35
NAME_PILL_ALPHA = 0.55
ZOOM_BTN_PX = 26
ZOOM_LABEL_MIN_W = 42
ZOOM_PILL_MARGIN_X = 16
ZOOM_PILL_MARGIN_Y = 14
OVERFLOW_MARGIN_X = 16
OVERFLOW_MARGIN_Y = 12
PREVIEW_CACHE_LIMIT = 8
ACTUAL_SIZE_ZOOM = 100  # 双击在适配与 100% 原始尺寸间切换
WHEEL_NOTCH = 120  # one physical wheel notch == one ZOOM_STEP (finer devices accumulate)
IMAGE_FADE_MS = 170  # cross-fade on image change (skip-safe: content is correct meanwhile)
IMAGE_FADE_FROM = 0.55  # starting opacity of the fade-in (fully visible content, just faint)
SPLITTER_H = 8
GRIP_W = 44
GRIP_H = 3

# Exact strings from the design.
TIP_PREV = "上一张 (Alt+↑)"
TIP_NEXT = "下一张 (Alt+↓)"
TEXT_DIRTY = "未保存"
TEXT_EDITING = "编辑中"
TEXT_FIT = "适应"
TIP_ZOOM_IN = "放大"
TIP_ZOOM_OUT = "缩小"
TIP_SPLITTER = "拖动调整编辑区高度"
MULTI_PILL_FMT = "对比 · 已选 {n} · 切换锁定在选中集"
OVERFLOW_FMT = "已选 {n} 张 · 仅显示前 4 张"
ZOOM_FMT = "{n}%"


def _read_full(path: str) -> QImage:
    """Worker-side full decode of the preview image."""
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        raise OSError(f"could not read image {path!r}: {reader.errorString()}")
    return image


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
        if changed and pixmap is not None and not pixmap.isNull() and self._zoom is None:
            self._start_fade()
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
        scale = min(1.0, avail_w / self._pixmap.width(), avail_h / self._pixmap.height())
        return max(1, round(scale * 100))

    def _scaled_size(self) -> QSize | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        scale = (self._zoom / 100.0) if self._zoom is not None else self.fit_percent() / 100.0
        return QSize(
            max(1, round(self._pixmap.width() * scale)),
            max(1, round(self._pixmap.height() * scale)),
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


class PreviewPanel(QFrame):
    """Preview-only middle widget (editor area is mounted by the integrator)."""

    def __init__(
        self,
        controller: AppController,
        *,
        tokens: ThemeTokens | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller
        self._tokens = tokens if tokens is not None else tokens_for_settings(controller.settings)
        self._zoom: int | None = None
        self._pix_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._loading: set[str] = set()
        self._epoch = 0
        self._multi_cells: list[_MultiCell] = []
        self._scroll_viewport: QWidget | None = None
        self._build_ui()
        self._connect_controller()
        self._apply_icon_colors()
        self._refresh_all()

    # -- construction --------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.header = _InfoBar(self)
        header = self.header
        header.setProperty("panel", True)
        header.setFixedHeight(HEADER_H)
        bar = QHBoxLayout(header)
        bar.setContentsMargins(14, 0, 0, 0)
        bar.setSpacing(10)
        self.name_label = QLabel(header)
        self.name_label.setFont(mono_font(12.5, QFont.Weight.DemiBold))
        bar.addWidget(self.name_label)
        self.meta_pill = QLabel(header)
        self.meta_pill.setProperty("pill", True)
        self.meta_pill.setProperty("mono", True)
        bar.addWidget(self.meta_pill)
        self.dim_label = QLabel(header)
        self.dim_label.setProperty("muted", True)
        self.dim_label.setFont(ui_font(11))
        bar.addWidget(self.dim_label)
        self.size_label = QLabel(header)
        self.size_label.setProperty("muted", True)
        self.size_label.setFont(ui_font(11))
        bar.addWidget(self.size_label)
        self.mtime_label = QLabel(header)
        self.mtime_label.setProperty("muted", True)
        self.mtime_label.setFont(ui_font(11))
        bar.addWidget(self.mtime_label)
        self.dirty_pill = QLabel(TEXT_DIRTY, header)
        self.dirty_pill.setProperty("pill", "warn")
        self.dirty_pill.hide()
        bar.addWidget(self.dirty_pill)
        self.multi_pill = QLabel(header)
        self.multi_pill.setProperty("pill", "accentSoft")
        self.multi_pill.hide()
        bar.addWidget(self.multi_pill)
        bar.addStretch(1)
        self.prev_button = self._nav_button(header, TIP_PREV, lambda: self.controller.nav(-1))
        bar.addWidget(self.prev_button)
        self.pos_label = QLabel(header)
        self.pos_label.setFont(mono_font(11.5))
        self.pos_label.setProperty("secondary", True)
        self.pos_label.setMinimumWidth(POS_MIN_W)
        self.pos_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar.addWidget(self.pos_label)
        self.next_button = self._nav_button(header, TIP_NEXT, lambda: self.controller.nav(1))
        bar.addWidget(self.next_button)
        tokens = self._tokens
        self.min_button = WindowButton(KIND_MIN, tokens, header)
        self.max_button = WindowButton(KIND_MAX, tokens, header)
        self.close_button = WindowButton(KIND_CLOSE, tokens, header)
        wire_window_buttons(self.min_button, self.max_button, self.close_button, self)
        for button in (self.min_button, self.max_button, self.close_button):
            bar.addWidget(button)
        root.addWidget(header)

        divider = QFrame(self)
        divider.setProperty("divider", True)
        divider.setFixedHeight(1)
        root.addWidget(divider)

        # preview host: stacked single / multi views + floating overlays
        self._host = QWidget(self)
        self._host.installEventFilter(self)
        self._stack = QStackedLayout(self._host)
        self._scroll = QScrollArea(self._host)
        # Manual sizing (not widgetResizable) so cursor-anchored zoom + drag-pan
        # can address the scroll bars deterministically via the single view.
        self._scroll.setWidgetResizable(False)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.single_view = _SingleView(self)
        self._scroll.setWidget(self.single_view)
        self._scroll_viewport = self._scroll.viewport()
        self._scroll_viewport.installEventFilter(self)
        self._stack.addWidget(self._scroll)
        self._multi_host = QWidget(self._host)
        self._multi_grid = QGridLayout(self._multi_host)
        self._multi_grid.setContentsMargins(
            PREVIEW_PAD, PREVIEW_PAD, PREVIEW_PAD, PREVIEW_PAD
        )
        self._multi_grid.setSpacing(MULTI_GAP)
        for i in range(2):
            self._multi_grid.setRowStretch(i, 1)
            self._multi_grid.setColumnStretch(i, 1)
        self._stack.addWidget(self._multi_host)
        root.addWidget(self._host, 1)

        self._build_zoom_pill()
        self.overflow_pill = QLabel(self._host)
        self.overflow_pill.hide()

    def _nav_button(self, parent: QWidget, tip: str, slot) -> QPushButton:  # noqa: ANN001
        btn = QPushButton(parent)
        btn.setFixedSize(NAV_BUTTON_PX, NAV_BUTTON_PX)
        btn.setToolTip(tip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(slot)
        return btn

    def _divider(self, parent: QWidget) -> QFrame:
        line = QFrame(parent)
        line.setProperty("divider", True)
        line.setFixedSize(1, DIVIDER_H)
        return line

    def _build_zoom_pill(self) -> None:
        self.zoom_pill = QFrame(self._host)
        self.zoom_pill.setObjectName("zoomPill")
        lay = QHBoxLayout(self.zoom_pill)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self.zoom_out_button = QPushButton("−", self.zoom_pill)
        self.zoom_out_button.setFixedSize(ZOOM_BTN_PX, ZOOM_BTN_PX)
        self.zoom_out_button.setToolTip(TIP_ZOOM_OUT)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        lay.addWidget(self.zoom_out_button)
        self.zoom_label_widget = QLabel(TEXT_FIT, self.zoom_pill)
        self.zoom_label_widget.setFont(mono_font(11))
        self.zoom_label_widget.setProperty("secondary", True)
        self.zoom_label_widget.setMinimumWidth(ZOOM_LABEL_MIN_W)
        self.zoom_label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.zoom_label_widget)
        self.zoom_in_button = QPushButton("+", self.zoom_pill)
        self.zoom_in_button.setFixedSize(ZOOM_BTN_PX, ZOOM_BTN_PX)
        self.zoom_in_button.setToolTip(TIP_ZOOM_IN)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        lay.addWidget(self.zoom_in_button)
        inner_divider = QFrame(self.zoom_pill)
        inner_divider.setProperty("divider", True)
        inner_divider.setFixedSize(1, 16)
        lay.addWidget(inner_divider)
        self.zoom_fit_button = QPushButton(TEXT_FIT, self.zoom_pill)
        self.zoom_fit_button.setFixedHeight(ZOOM_BTN_PX)
        self.zoom_fit_button.clicked.connect(self.zoom_fit)
        lay.addWidget(self.zoom_fit_button)

    def _connect_controller(self) -> None:
        c = self.controller
        c.dataset_opened.connect(self._on_dataset_opened)
        c.current_changed.connect(self._on_current_changed)
        c.selection_changed.connect(self._on_selection_changed)
        c.caption_changed.connect(self._on_caption_changed)
        c.filter_changed.connect(lambda _t: self._refresh_pos())

    # -- tokens ----------------------------------------------------------------------
    def current_tokens(self) -> ThemeTokens:
        return self._tokens

    def apply_tokens(self, tokens: ThemeTokens) -> None:
        self._tokens = tokens
        self._apply_icon_colors()
        for button in (self.min_button, self.max_button, self.close_button):
            button.set_tokens(tokens)
        self.single_view.update()
        for cell in self._multi_cells:
            cell.update()

    def _apply_icon_colors(self) -> None:
        t = self._tokens
        self.prev_button.setIcon(make_icon("chevron_left", t.text2, 14))
        self.next_button.setIcon(make_icon("chevron_right", t.text2, 14))
        self.zoom_pill.setStyleSheet(
            f"QFrame#zoomPill {{ background: {t.surface}; border: 1px solid {t.bd};"
            f" border-radius: 9px; }}"
        )
        self.overflow_pill.setStyleSheet(
            f"QLabel {{ background: {t.surface}; border: 1px solid {t.bd};"
            f" border-radius: 12px; padding: 3px 10px; color: {t.text2};"
            f" font-size: 10.5px; font-weight: 600; }}"
        )

    # -- zoom ------------------------------------------------------------------------
    def zoom_label(self) -> str:
        return TEXT_FIT if self._zoom is None else ZOOM_FMT.format(n=self._zoom)

    def zoom_in(self) -> None:
        self._set_zoom(self._clamp_zoom(self._zoom_base() + ZOOM_STEP))

    def zoom_out(self) -> None:
        self._set_zoom(self._clamp_zoom(self._zoom_base() - ZOOM_STEP))

    def zoom_fit(self) -> None:
        self._set_zoom(None)

    def _zoom_base(self) -> int:
        # From 适应 (fit) the prototype steps relative to 100%, not the actual
        # fit scale: first + click on a fit image yields 120%.
        return self._zoom if self._zoom is not None else 100

    def _clamp_zoom(self, value: int) -> int:
        """Clamp a zoom percentage to the usable range.

        The floor additionally drops to the current fit percentage so zooming
        out always ends at (or below) a whole-image view, even for images whose
        fit scale is under ``ZOOM_RANGE[0]`` (e.g. a 2894x4093 photo fits at
        ~20%% in a typical viewport).
        """
        low, high = ZOOM_RANGE
        low = min(low, self.single_view.fit_percent())
        return max(low, min(high, value))

    def _set_zoom(self, zoom: int | None) -> None:
        self._zoom = zoom
        self.single_view.set_zoom(zoom)
        self._sync_single_size()
        self.zoom_label_widget.setText(self.zoom_label())

    # -- single-view sizing / scroll ---------------------------------------------------
    def _sync_single_size(self) -> None:
        """Resize the single view to the current zoom against the viewport."""
        self.single_view.sync_size(self._scroll.viewport().size())

    def single_scrollbars(self):  # noqa: ANN201 - Qt scrollbar pair
        """(horizontal, vertical) scroll bars backing the single-image pan."""
        return self._scroll.horizontalScrollBar(), self._scroll.verticalScrollBar()

    def single_pannable(self) -> bool:
        """True when the zoomed image overflows the viewport (drag can pan)."""
        if self._zoom is None or self.controller.multi_mode():
            return False
        hbar, vbar = self.single_scrollbars()
        return hbar.maximum() > 0 or vbar.maximum() > 0

    def wheel_zoom(self, steps: int, view_pos: QPointF) -> None:
        """Step the zoom by ``steps`` notches, keeping ``view_pos`` under the cursor."""
        target = self._clamp_zoom(self._zoom_base() + steps * ZOOM_STEP)
        if self._zoom is not None and target == self._zoom:
            return
        self._apply_zoom_anchored(target, view_pos)

    def toggle_actual_size(self, view_pos: QPointF) -> None:
        """Double-click toggle between 适应 (fit) and 100% original size."""
        if self._zoom is None:
            self._apply_zoom_anchored(self._clamp_zoom(ACTUAL_SIZE_ZOOM), view_pos)
        else:
            self.zoom_fit()

    def _apply_zoom_anchored(self, target: int | None, view_pos: QPointF) -> None:
        """Set the zoom while pinning the image point under ``view_pos`` in place.

        ``view_pos`` is in single-view coordinates. The image fraction beneath
        it is preserved by shifting the scroll offset toward the cursor instead
        of always anchoring the top-left corner.
        """
        old_rect = self.single_view.image_rect()
        hbar, vbar = self.single_scrollbars()
        # Viewport-relative cursor position (view coord = viewport coord + offset).
        vp_x = view_pos.x() - hbar.value()
        vp_y = view_pos.y() - vbar.value()
        if old_rect is not None and old_rect.width() > 0 and old_rect.height() > 0:
            fx = min(1.0, max(0.0, (view_pos.x() - old_rect.left()) / old_rect.width()))
            fy = min(1.0, max(0.0, (view_pos.y() - old_rect.top()) / old_rect.height()))
        else:
            fx = fy = 0.5
        self._set_zoom(target)  # resizes the view + refreshes scroll ranges
        if target is None:
            return
        new_rect = self.single_view.image_rect()
        if new_rect is None:
            return
        new_view_x = new_rect.left() + fx * new_rect.width()
        new_view_y = new_rect.top() + fy * new_rect.height()
        hbar.setValue(round(new_view_x - vp_x))
        vbar.setValue(round(new_view_y - vp_y))

    # -- image cache -------------------------------------------------------------------
    def pixmap_for(self, key: str) -> QPixmap | None:
        """Cached full preview pixmap for ``key`` (starts an async load)."""
        cached = self._pix_cache.get(key)
        if cached is not None:
            self._pix_cache.move_to_end(key)
            return cached
        self._ensure_image(key)
        return None

    def _ensure_image(self, key: str) -> None:
        if not key or key in self._pix_cache or key in self._loading:
            return
        try:
            path = self.controller.image_path(key)
        except NLaptError:
            return
        self._loading.add(key)
        epoch = self._epoch

        def done(image: object) -> None:
            # A queued reply can arrive after the panel's C++ object is gone
            # (e.g. window closed while a load was in flight); ignore it.
            if not isValid(self):
                return
            self._loading.discard(key)
            if epoch != self._epoch or not isinstance(image, QImage):
                return
            self._store_pixmap(key, QPixmap.fromImage(image))

        def failed(message: str) -> None:
            if not isValid(self):
                return
            self._loading.discard(key)
            _LOGGER.warning("preview load failed for %r: %s", key, message)

        run_async(
            QThreadPool.globalInstance(), _read_full, str(path), on_done=done, on_error=failed
        )

    def _store_pixmap(self, key: str, pixmap: QPixmap) -> None:
        self._pix_cache[key] = pixmap
        self._pix_cache.move_to_end(key)
        while len(self._pix_cache) > PREVIEW_CACHE_LIMIT:
            self._pix_cache.popitem(last=False)
        if key == self.controller.current_key:
            self.single_view.set_pixmap(pixmap)
            self._sync_single_size()
            self.zoom_label_widget.setText(self.zoom_label())
        for cell in self._multi_cells:
            if cell.key == key:
                cell.update()

    # -- controller reactions -------------------------------------------------------------
    def _on_dataset_opened(self, _result: object) -> None:
        self._epoch += 1
        self._pix_cache.clear()
        self._loading.clear()
        self.single_view.set_pixmap(None)
        self._refresh_all()

    def _on_current_changed(self, _key: str) -> None:
        self._refresh_header()
        self._refresh_pos()
        # Each new image starts at 适应 (fit), mirroring the prototype's pick().
        self._set_zoom(None)
        key = self.controller.current_key
        cached = self._pix_cache.get(key) if key else None
        if cached is not None or not key:
            self.single_view.set_pixmap(cached)
        self._sync_single_size()
        if key:
            self._ensure_image(key)
        for cell in self._multi_cells:
            cell.update()

    def _on_selection_changed(self) -> None:
        self._refresh_mode()
        self._refresh_pos()

    def _on_caption_changed(self, key: str) -> None:
        if key == self.controller.current_key:
            self._refresh_header()
        for cell in self._multi_cells:
            if cell.key == key:
                cell.update()

    # -- refreshers ------------------------------------------------------------------------
    def _refresh_all(self) -> None:
        self._refresh_header()
        self._refresh_mode()
        self._refresh_pos()
        self._on_current_changed(self.controller.current_key or "")

    def _refresh_header(self) -> None:
        controller = self.controller
        key = controller.current_key
        if key is None:
            self.name_label.setText("")
            self.meta_pill.setText("")
            self.meta_pill.hide()
            self.dim_label.setText("")
            self.size_label.setText("")
            self.mtime_label.setText("")
            self.dirty_pill.hide()
            return
        self.name_label.setText(key.rsplit("/", 1)[-1])
        try:
            self.meta_pill.setText(controller.image_format(key))
            self.meta_pill.show()
            meta = controller.image_meta(key)
            parts = [part.strip() for part in meta.split("·")]
            self.dim_label.setText(parts[0] if parts else "")
            self.size_label.setText(parts[-1] if len(parts) > 2 else "")
            self.mtime_label.setText(controller.image_modified_label(key))
        except NLaptError:
            self.meta_pill.hide()
            self.dim_label.setText("")
            self.size_label.setText("")
            self.mtime_label.setText("")
        self.dirty_pill.setVisible(controller.record(key).dirty)

    def _refresh_mode(self) -> None:
        controller = self.controller
        multi = controller.multi_mode()
        selected = len(controller.selected_keys())
        self._stack.setCurrentIndex(1 if multi else 0)
        self.zoom_pill.setVisible(not multi)
        self.multi_pill.setVisible(multi)
        if multi:
            self.multi_pill.setText(MULTI_PILL_FMT.format(n=selected))
            self._rebuild_multi_cells()
        overflow = multi and selected > MULTI_PREVIEW_LIMIT
        self.overflow_pill.setVisible(overflow)
        if overflow:
            self.overflow_pill.setText(OVERFLOW_FMT.format(n=selected))
            self.overflow_pill.adjustSize()
        self._position_overlays()

    def _refresh_pos(self) -> None:
        self.pos_label.setText(self.controller.pos_label())

    def _rebuild_multi_cells(self) -> None:
        keys = self.controller.editor_keys()
        if [cell.key for cell in self._multi_cells] == list(keys):
            for cell in self._multi_cells:
                cell.update()
            return
        for cell in self._multi_cells:
            cell.hide()
            cell.deleteLater()
        self._multi_cells = []
        for index, key in enumerate(keys):
            cell = _MultiCell(key, self, self._multi_host)
            row, col = divmod(index, 2)
            self._multi_grid.addWidget(cell, row, col)
            cell.show()
            self._multi_cells.append(cell)
            self._ensure_image(key)

    def multi_cells(self) -> tuple[_MultiCell, ...]:
        """Live compare-grid tiles (test/integration helper)."""
        return tuple(self._multi_cells)

    def current_view(self) -> str:
        """Which preview page is active: ``single`` or ``multi``."""
        return "multi" if self._stack.currentIndex() == 1 else "single"

    def set_busy(self, busy: bool) -> None:
        """Kept for the main-window busy hook; save now lives on the rail."""
        del busy

    # -- overlays --------------------------------------------------------------------------
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self._host and event.type() == QEvent.Type.Resize:
            self._position_overlays()
        elif obj is self._scroll_viewport and event.type() == QEvent.Type.Resize:
            # Keep the single view filling the viewport at 适应 and at least the
            # scaled image plus padding when zoomed.
            self._sync_single_size()
        return super().eventFilter(obj, event)

    def _position_overlays(self) -> None:
        host = self._host
        pill = self.zoom_pill
        pill.adjustSize()
        pill.move(
            host.width() - pill.width() - ZOOM_PILL_MARGIN_X,
            host.height() - pill.height() - ZOOM_PILL_MARGIN_Y,
        )
        pill.raise_()
        overflow = self.overflow_pill
        overflow.adjustSize()
        overflow.move(host.width() - overflow.width() - OVERFLOW_MARGIN_X, OVERFLOW_MARGIN_Y)
        overflow.raise_()


class SplitterHandle(QFrame):
    """8px drag handle between preview and editor (design 拖动分隔条).

    Emits ``editor_h_changed`` (clamped to EDITOR_H_RANGE) while dragging;
    persists the final value into ``UISettings.editor_h`` on release. The
    main window owns the actual layout and applies the emitted height to
    the editor area.
    """

    editor_h_changed = Signal(int)

    def __init__(
        self,
        controller: AppController,
        *,
        tokens: ThemeTokens | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._tokens = tokens if tokens is not None else tokens_for_settings(controller.settings)
        self._value = controller.settings.editor_h
        self._drag_origin: float | None = None
        self._drag_start_h = self._value
        self.setFixedHeight(SPLITTER_H)
        self.setCursor(Qt.CursorShape.SplitVCursor)
        self.setToolTip(TIP_SPLITTER)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def editor_height(self) -> int:
        return self._value

    def apply_tokens(self, tokens: ThemeTokens) -> None:
        self._tokens = tokens
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().y()
            self._drag_start_h = self._value
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag_origin is None:
            super().mouseMoveEvent(event)
            return
        delta = event.globalPosition().y() - self._drag_origin
        low, high = EDITOR_H_RANGE
        new_h = int(max(low, min(high, self._drag_start_h - delta)))
        if new_h != self._value:
            self._value = new_h
            self.editor_h_changed.emit(new_h)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if self._drag_origin is not None:
            self._drag_origin = None
            self._controller.update_settings(editor_h=self._value)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        tokens = self._tokens
        bg = QColor(tokens.surface2 if self.underMouse() else tokens.panel)
        painter.fillRect(self.rect(), bg)
        painter.setPen(QPen(QColor(tokens.bd), 1))
        painter.drawLine(0, 0, self.width(), 0)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(tokens.bd2))
        grip = QRectF(
            (self.width() - GRIP_W) / 2.0,
            (self.height() - GRIP_H) / 2.0 + 0.5,
            GRIP_W,
            GRIP_H,
        )
        painter.drawRoundedRect(grip, GRIP_H / 2.0, GRIP_H / 2.0)
        painter.end()

    def event(self, ev: QEvent) -> bool:  # noqa: N802 - Qt override
        if ev.type() in (QEvent.Type.HoverEnter, QEvent.Type.HoverLeave):
            self.update()
        return super().event(ev)
