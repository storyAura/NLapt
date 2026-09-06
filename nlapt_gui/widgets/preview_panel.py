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
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QImageReader,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPixmap,
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

from nlapt_gui.controller import AppController, MULTI_PREVIEW_LIMIT
from nlapt_gui.theme.tokens import EDITOR_H_RANGE, ZOOM_RANGE, ZOOM_STEP, ThemeTokens
from nlapt_gui.widgets.preview_views import (
    PREVIEW_PAD as PREVIEW_PAD,
    IMAGE_RADIUS as IMAGE_RADIUS,
    MULTI_RADIUS as MULTI_RADIUS,
    MULTI_BORDER_W as MULTI_BORDER_W,
    MULTI_GLOW_W as MULTI_GLOW_W,
    BADGE_MARGIN as BADGE_MARGIN,
    DIRTY_DOT_PX as DIRTY_DOT_PX,
    DIRTY_RING_W as DIRTY_RING_W,
    DIRTY_RING_ALPHA as DIRTY_RING_ALPHA,
    NAME_PILL_ALPHA as NAME_PILL_ALPHA,
    WHEEL_NOTCH as WHEEL_NOTCH,
    IMAGE_FADE_MS as IMAGE_FADE_MS,
    IMAGE_FADE_FROM as IMAGE_FADE_FROM,
    TEXT_EDITING as TEXT_EDITING,
    _InfoBar,
    _MultiCell,
    _SingleView,
)
from nlapt_gui.widgets.thumb_cells import (
    make_icon,
    mono_font,
    tokens_for_settings,
    ui_font,
)
from nlapt_gui.widgets.thumbnails import get_decode_pool
from nlapt_gui.widgets.window_chrome import (
    KIND_CLOSE,
    KIND_MAX,
    KIND_MIN,
    WindowButton,
    wire_window_buttons,
)
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

HEADER_H = 46
NAV_BUTTON_PX = 28
POS_MIN_W = 44
DIVIDER_H = 18
MULTI_GAP = 14
ZOOM_BTN_PX = 26
ZOOM_LABEL_MIN_W = 42
ZOOM_PILL_MARGIN_X = 16
ZOOM_PILL_MARGIN_Y = 14
OVERFLOW_MARGIN_X = 16
OVERFLOW_MARGIN_Y = 12
PREVIEW_CACHE_LIMIT = 8
PREVIEW_CACHE_PIXEL_BUDGET = 64 * 1024 * 1024
COARSE_PREVIEW_LIMIT = 16
COARSE_PREVIEW_EDGE = 512
PREVIEW_ACTIVE_LIMIT = 1
ACTUAL_SIZE_ZOOM = 100  # 双击在适配与 100% 原始尺寸间切换
SPLITTER_H = 8
GRIP_W = 44
GRIP_H = 3

# Exact strings from the design.
TIP_PREV = "上一张 (Alt+↑)"
TIP_NEXT = "下一张 (Alt+↓)"
TEXT_DIRTY = "未保存"
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


def _read_coarse(path: str, max_edge: int) -> QImage:
    """Worker-side bounded decode used as an immediate preview frame."""
    reader = QImageReader(path)
    reader.setAutoTransform(True)
    size = reader.size()
    if size.isValid() and max(size.width(), size.height()) > max_edge:
        scale = max_edge / max(size.width(), size.height())
        reader.setScaledSize(QSize(max(1, round(size.width() * scale)), max(1, round(size.height() * scale))))
    image = reader.read()
    if image.isNull():
        raise OSError(f"could not read image {path!r}: {reader.errorString()}")
    return image


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
        self._pix_cache_pixels = 0
        self._coarse_cache: OrderedDict[str, QPixmap] = OrderedDict()
        self._loading: set[str] = set()
        self._loading_epoch: dict[str, int] = {}
        self._preview_pending: OrderedDict[str, tuple[str, str, int]] = OrderedDict()
        self._preview_active = False
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
        if not key or key in self._pix_cache:
            return
        try:
            path = self.controller.image_path(key)
        except NLaptError:
            return
        if key in self._preview_pending or self._loading_epoch.get(key) == self._epoch:
            return
        stage = "native" if key in self._coarse_cache else "coarse"
        self._preview_pending[key] = (str(path), stage, self._epoch)
        self._loading.add(key)
        self._loading_epoch[key] = self._epoch
        self._pump_preview()

    def _pump_preview(self) -> None:
        if self._preview_active or not self._preview_pending:
            return
        key = self.controller.current_key
        chosen = key if key in self._preview_pending else next(iter(self._preview_pending))
        path, stage, epoch = self._preview_pending.pop(chosen)
        self._preview_active = True
        reader = _read_full if stage == "native" else _read_coarse
        args = (path,) if stage == "native" else (path, COARSE_PREVIEW_EDGE)

        def release() -> bool:
            if not isValid(self):
                return False
            self._preview_active = False
            current = epoch == self._epoch
            if current:
                self._loading.discard(chosen)
                self._loading_epoch.pop(chosen, None)
            return current

        def done(image: object) -> None:
            if not release() or not isinstance(image, QImage):
                return
            pixmap = QPixmap.fromImage(image)
            if stage == "coarse":
                self._coarse_cache[chosen] = pixmap
                self._coarse_cache.move_to_end(chosen)
                while len(self._coarse_cache) > COARSE_PREVIEW_LIMIT:
                    self._coarse_cache.popitem(last=False)
                if chosen == self.controller.current_key:
                    self.single_view.set_pixmap(pixmap)
                    self._sync_single_size()
                self._ensure_image(chosen)
                self._pump_preview()
            else:
                self._store_pixmap(chosen, pixmap)
                self._pump_preview()

        def failed(message: str) -> None:
            if release():
                _LOGGER.warning("preview load failed for %r: %s", chosen, message)
            self._pump_preview()

        run_async(get_decode_pool(), reader, *args, on_done=done, on_error=failed)

    def _store_pixmap(self, key: str, pixmap: QPixmap) -> None:
        previous = self._pix_cache.pop(key, None)
        if previous is not None:
            self._pix_cache_pixels -= previous.width() * previous.height()
        self._pix_cache[key] = pixmap
        self._pix_cache.move_to_end(key)
        self._pix_cache_pixels += pixmap.width() * pixmap.height()
        while (
            len(self._pix_cache) > PREVIEW_CACHE_LIMIT
            or self._pix_cache_pixels > PREVIEW_CACHE_PIXEL_BUDGET
        ):
            _old_key, old = self._pix_cache.popitem(last=False)
            self._pix_cache_pixels -= old.width() * old.height()
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
        self._pix_cache_pixels = 0
        self._coarse_cache.clear()
        self._loading.clear()
        self._loading_epoch.clear()
        self._preview_pending.clear()
        self.single_view.set_pixmap(None)
        self.single_view.set_source_size(None)
        self._refresh_all()

    def _on_current_changed(self, _key: str) -> None:
        self._refresh_header()
        self._refresh_pos()
        # Each new image starts at 适应 (fit), mirroring the prototype's pick().
        self._set_zoom(None)
        key = self.controller.current_key
        self._set_source_size(key)
        cached = self._pix_cache.get(key) if key else None
        coarse = self._coarse_cache.get(key) if key else None
        self.single_view.set_pixmap(cached or coarse)
        self._sync_single_size()
        if key:
            self._ensure_image(key)
        for cell in self._multi_cells:
            cell.update()

    def _set_source_size(self, key: str | None) -> None:
        """Use the cached header dimensions to keep coarse/native geometry stable."""
        if not key:
            self.single_view.set_source_size(None)
            return
        try:
            first = self.controller.image_meta(key).split("·", 1)[0].strip()
            width_text, height_text = first.split("×", 1)
            self.single_view.set_source_size(QSize(int(width_text), int(height_text)))
        except (NLaptError, ValueError):
            self.single_view.set_source_size(None)

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
        enabled = self.controller.can_navigate()
        self.prev_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)

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
