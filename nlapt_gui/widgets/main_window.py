"""MainWindow - frameless assembly of the whole CaptionForge layout.

Structure (per design): 40px title bar / three columns (file panel |
preview + splitter + editor | tools panel) held in a horizontal
:class:`QSplitter` with two draggable dividers / 26px status bar, plus the
toast overlay.

Window resizing uses Qt's cross-platform ``startSystemResize``: hovering near
an edge shows a resize cursor and pressing there starts the OS resize loop.
This deliberately avoids ``WM_NCHITTEST`` (which turns pixels into a non-client
area and, on HiDPI, silently swallowed clicks over whole panels). The window
also owns window-level shortcuts, a first-show fade-in and the real
:class:`TranslateBridge` wired into the editor panel.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPoint,
    QPropertyAnimation,
    Qt,
)
from PySide6.QtGui import (
    QCloseEvent,
    QCursor,
    QKeySequence,
    QMouseEvent,
    QResizeEvent,
    QShortcut,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLineEdit,
    QPlainTextEdit,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text

from nlapt_gui import anim
from nlapt_gui.controller import AppController
from nlapt_gui.resources import app_data_dir
from nlapt_gui.theme.logo import load_app_icon
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import EDITOR_H_RANGE, MIN_WINDOW, ThemeTokens
from nlapt_gui.translate_bridge import TranslateBridge
from nlapt_gui.vision_bridge import VisionBridge
from nlapt_gui.widgets.color_dialog import ColorSettingsDialog
from nlapt_gui.widgets.editor_panel import EditorPanel
from nlapt_gui.widgets.file_panel import FilePanel
from nlapt_gui.widgets.preview_panel import PreviewPanel, SplitterHandle
from nlapt_gui.widgets.status_bar import StatusBar
from nlapt_gui.widgets.title_bar import TitleBar
from nlapt_gui.widgets.toast import ToastOverlay
from nlapt_gui.widgets.tools_panel import ToolsPanel

_LOGGER = get_logger(__name__)

# Frameless edge-resize grab margin (px). Wide enough to grab comfortably; a
# hover cursor makes the edge discoverable.
EDGE_MARGIN_PX = 8

WINDOW_TITLE = "NLapt"
OPEN_FOLDER_CAPTION = "打开数据集文件夹"

# Body splitter: thin themed handle + default / min / max panel widths.
SPLITTER_HANDLE_W = 4
FILE_PANEL_DEFAULT_W = 300
TOOLS_PANEL_DEFAULT_W = 340
MIDDLE_DEFAULT_W = 760
MIDDLE_MIN_W = 360
PANEL_MIN_W = 240
PANEL_MAX_W = 16_777_215  # Qt QWIDGETSIZE_MAX - "large value" per contract
SPLITTER_SIZES_FILE = "body_splitter.json"

# Startup fade-in / close fade-out durations (skip-safe: end state is
# reachable synchronously and both collapse to no-ops in tests).
FADE_IN_MS = 240
FADE_OUT_MS = 140

# Text-input widget types whose own undo wins over the global Ctrl+Z.
_TEXT_INPUT_TYPES = (QLineEdit, QPlainTextEdit, QTextEdit)


def _clamp_editor_h(value: int) -> int:
    low, high = EDITOR_H_RANGE
    return max(low, min(high, value))


def _splitter_sizes_path() -> Path:
    return app_data_dir() / SPLITTER_SIZES_FILE


def load_splitter_sizes(path: Path | None = None) -> list[int] | None:
    """Load the persisted body-splitter pane sizes; missing/corrupt -> None."""
    target = path if path is not None else _splitter_sizes_path()
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _LOGGER.warning("corrupt splitter sizes %s (%s); ignoring", target, exc)
        return None
    if (
        not isinstance(data, list)
        or len(data) != 3
        or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in data)
    ):
        return None
    return [int(v) for v in data]


def save_splitter_sizes(sizes: list[int], path: Path | None = None) -> None:
    """Persist the body-splitter pane sizes atomically as UTF-8 JSON."""
    target = path if path is not None else _splitter_sizes_path()
    atomic_write_text(target, json.dumps(list(sizes), ensure_ascii=False, indent=2))
    _LOGGER.debug("splitter sizes saved to %s", target)


class _BodySplitter(QSplitter):
    """Horizontal 3-pane splitter that pins exact side widths on first layout.

    ``QSplitter.setSizes`` scales the requested sizes proportionally when their
    sum differs from the current width, so it cannot guarantee an exact initial
    side-panel width. This subclass records the desired left/right widths and
    applies them (middle absorbs the remainder) on the first layout wide enough
    to honour them, then leaves the panes fully user-draggable.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._pending: tuple[int, int] | None = None

    def set_initial_side_widths(self, left: int, right: int) -> None:
        """Request exact left/right pane widths for the first valid layout."""
        self._pending = (left, right)

    def _handles_width(self) -> int:
        return self.handleWidth() * max(0, self.count() - 1)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self._pending is None or self.count() != 3:
            return
        left, right = self._pending
        needed = left + right + self._handles_width() + PANEL_MIN_W
        if self.width() >= needed:
            middle = self.width() - left - right - self._handles_width()
            self.setSizes([left, middle, right])
            self._pending = None


class MainWindow(QWidget):
    """Top-level frameless window assembling all panels."""

    def __init__(
        self,
        controller: AppController,
        theme_manager: ThemeManager,
        parent: QWidget | None = None,
        *,
        splitter_sizes_path: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._theme_manager = theme_manager
        self._splitter_sizes_path = splitter_sizes_path
        self._resize_cursor_active = False
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(*MIN_WINDOW)
        self.setAutoFillBackground(True)
        # Window / taskbar icon: packaging/icon.ico when present, else the
        # theme-aware painted mark (refreshed on theme changes below).
        self.setWindowIcon(load_app_icon(theme_manager.tokens))

        # Keep strong references: the bridges own async LLM work.
        self.translate_bridge = TranslateBridge(controller, parent=self)
        self.vision_bridge = VisionBridge(controller, parent=self)

        self.title_bar = TitleBar(controller, theme_manager, self)
        self.file_panel = FilePanel(controller, tokens=theme_manager.tokens, parent=self)
        self.preview_panel = PreviewPanel(controller, tokens=theme_manager.tokens, parent=self)
        self.splitter = SplitterHandle(controller, tokens=theme_manager.tokens, parent=self)
        self.editor_panel = EditorPanel(
            controller, self.translate_bridge, self, vision_bridge=self.vision_bridge
        )
        self.tools_panel = ToolsPanel(controller, self)
        self.status_bar = StatusBar(controller, theme_manager, self)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.title_bar)
        root.addWidget(self._build_body(), 1)
        root.addWidget(self.status_bar)

        self.toast_overlay = ToastOverlay(self, tokens=theme_manager.tokens)
        controller.toast_requested.connect(self.toast_overlay.show_toast)

        self.editor_panel.setFixedHeight(_clamp_editor_h(controller.settings.editor_h))
        self.splitter.editor_h_changed.connect(self._apply_editor_height)

        # Disable batch/save buttons while an async batch or save runs.
        controller.busy_changed.connect(self._on_busy_changed)

        self.file_panel.open_folder_requested.connect(self.pick_folder)
        self.title_bar.open_folder_requested.connect(self.pick_folder)
        self.title_bar.settings_requested.connect(self.tools_panel.open_settings_dialog)
        self.title_bar.colors_requested.connect(self.open_color_settings)

        theme_manager.theme_changed.connect(self._on_theme_changed)
        self._install_shortcuts()

        # Fade-in / fade-out state (skip-safe for tests).
        self._did_fade_in = False
        self._fade_anim: QPropertyAnimation | None = None
        self._close_fade_done = False

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # -- body assembly -----------------------------------------------------------------
    def _build_body(self) -> QSplitter:
        """Three columns in a horizontal QSplitter with two draggable dividers."""
        middle = QWidget(self)
        middle_lay = QVBoxLayout(middle)
        middle_lay.setContentsMargins(0, 0, 0, 0)
        middle_lay.setSpacing(0)
        middle_lay.addWidget(self.preview_panel, 1)
        middle_lay.addWidget(self.splitter)
        middle_lay.addWidget(self.editor_panel)
        # Explicit minimum overrides the container's large content-driven
        # minimumSizeHint so the QSplitter honours the side-panel widths
        # instead of forcing the middle column wide (contract 3.1).
        middle.setMinimumWidth(MIDDLE_MIN_W)

        splitter = _BodySplitter(self)
        splitter.setObjectName("bodySplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(SPLITTER_HANDLE_W)
        splitter.addWidget(self.file_panel)
        splitter.addWidget(middle)
        splitter.addWidget(self.tools_panel)

        # The panels lock their own width in their constructors; release that
        # lock so the splitter can resize them (contract 3.1).
        for panel in (self.file_panel, self.tools_panel):
            panel.setMinimumWidth(PANEL_MIN_W)
            panel.setMaximumWidth(PANEL_MAX_W)

        # Only the middle column stretches; the side panels keep their sizes.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        saved = load_splitter_sizes(self._splitter_sizes_path)
        if saved is not None:
            left, _mid, right = saved
        else:
            left, right = FILE_PANEL_DEFAULT_W, TOOLS_PANEL_DEFAULT_W
        splitter.setSizes([left, MIDDLE_DEFAULT_W, right])
        splitter.set_initial_side_widths(left, right)
        splitter.splitterMoved.connect(self._on_splitter_moved)

        self.body_splitter = splitter
        self._style_splitter(self._theme_manager.tokens)
        return splitter

    def _style_splitter(self, tokens: ThemeTokens) -> None:
        """Subtle themed divider (thin border color, accent on hover)."""
        self.body_splitter.setStyleSheet(
            f"QSplitter#bodySplitter::handle {{ background: {tokens.bd}; }}"
            f"QSplitter#bodySplitter::handle:hover {{ background: {tokens.accent}; }}"
        )

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        self._persist_splitter_sizes()

    def _persist_splitter_sizes(self) -> None:
        sizes = self.body_splitter.sizes()
        if len(sizes) != 3 or any(size <= 0 for size in sizes):
            return  # never laid out yet; don't clobber a good saved layout
        try:
            save_splitter_sizes(sizes, self._splitter_sizes_path)
        except OSError:
            _LOGGER.exception("could not persist splitter sizes")

    # -- layout ----------------------------------------------------------------------
    def _apply_editor_height(self, value: int) -> None:
        self.editor_panel.setFixedHeight(_clamp_editor_h(value))

    def _on_busy_changed(self, busy: bool) -> None:
        self.preview_panel.set_busy(busy)
        self.tools_panel.set_busy(busy)

    # -- folder picking ---------------------------------------------------------------
    def pick_folder(self) -> None:
        """Directory dialog -> controller.open_dataset."""
        start = str(self._controller.root) if self._controller.root else str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, OPEN_FOLDER_CAPTION, start)
        if chosen:
            self._controller.open_dataset(Path(chosen))

    # -- theme ------------------------------------------------------------------------
    def open_color_settings(self) -> None:
        """色彩设置 window (theme + custom accent; applies live)."""
        dialog = ColorSettingsDialog(self._theme_manager, parent=self)
        dialog.exec()

    def _on_theme_changed(self, tokens: ThemeTokens) -> None:
        """Propagate new tokens to token-painted children + persisted settings."""
        self._controller.update_settings(
            theme=self._theme_manager.theme_name, accent=self._theme_manager.accent
        )
        self.file_panel.apply_tokens(tokens)
        self.preview_panel.apply_tokens(tokens)
        self.splitter.apply_tokens(tokens)
        self.toast_overlay.set_tokens(tokens)
        self._style_splitter(tokens)
        self.setWindowIcon(load_app_icon(tokens))

    # -- shortcuts ---------------------------------------------------------------------
    def _install_shortcuts(self) -> None:
        self._shortcuts: list[QShortcut] = []

        def add(sequence: str, handler) -> None:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(handler)
            self._shortcuts.append(shortcut)

        add("Ctrl+S", lambda: self._controller.save_current())
        add("Ctrl+Shift+S", lambda: self._controller.save_all())
        add("Ctrl+Z", self._undo_shortcut)
        add("Alt+Up", lambda: self._controller.nav(-1))
        add("Alt+Down", lambda: self._controller.nav(1))

    def _undo_shortcut(self) -> None:
        """Ctrl+Z: text inputs keep their own undo; otherwise undo the caption."""
        focus = QApplication.focusWidget()
        if isinstance(focus, _TEXT_INPUT_TYPES):
            focus.undo()
            return
        self._controller.undo_current()

    # -- first-show fade-in ------------------------------------------------------------
    # Implemented with windowOpacity (like the theme transition), NOT a
    # QGraphicsOpacityEffect: a graphics effect on a top-level window forces
    # effect-buffer re-rendering during native resizes and hard-crashes Qt.
    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._did_fade_in:
            self._did_fade_in = True
            self._start_fade_in()

    def _start_fade_in(self) -> None:
        self.setWindowOpacity(0.0)
        anim = QPropertyAnimation(self, b"windowOpacity", self)
        anim.setDuration(FADE_IN_MS)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.finished.connect(self._clear_fade_effect)
        self._fade_anim = anim
        anim.start()

    def finish_fade_in(self) -> None:
        """Jump the first-show fade-in to its end state (skip-safe for tests)."""
        anim = self._fade_anim
        if anim is not None:
            anim.stop()
        self._clear_fade_effect()

    def _clear_fade_effect(self) -> None:
        self._fade_anim = None
        self.setWindowOpacity(1.0)

    # -- frameless edge resize (cross-platform, click-safe) ----------------------------
    # Deliberately NOT implemented via WM_NCHITTEST: a native hit-test that returns
    # resize codes turns those pixels into a non-client area, and on a HiDPI display
    # (physical vs logical mouse coords) that silently swallowed clicks over large
    # interior regions. startSystemResize only runs on an actual edge press, so an
    # interior click can never be blocked. A hover cursor makes the edge discoverable.
    def edge_at(self, pos: QPoint) -> Qt.Edges:
        """Resize edges hit by a window-local point (6px margins)."""
        edges = Qt.Edges()
        if pos.x() <= EDGE_MARGIN_PX:
            edges |= Qt.Edge.LeftEdge
        elif pos.x() >= self.width() - EDGE_MARGIN_PX:
            edges |= Qt.Edge.RightEdge
        if pos.y() <= EDGE_MARGIN_PX:
            edges |= Qt.Edge.TopEdge
        elif pos.y() >= self.height() - EDGE_MARGIN_PX:
            edges |= Qt.Edge.BottomEdge
        return edges

    def edge_cursor_shape(self, edges: Qt.Edges) -> Qt.CursorShape | None:
        """Resize cursor for the given edges, or None when not on an edge."""
        left = bool(edges & Qt.Edge.LeftEdge)
        right = bool(edges & Qt.Edge.RightEdge)
        top = bool(edges & Qt.Edge.TopEdge)
        bottom = bool(edges & Qt.Edge.BottomEdge)
        if (top and left) or (bottom and right):
            return Qt.CursorShape.SizeFDiagCursor
        if (top and right) or (bottom and left):
            return Qt.CursorShape.SizeBDiagCursor
        if left or right:
            return Qt.CursorShape.SizeHorCursor
        if top or bottom:
            return Qt.CursorShape.SizeVerCursor
        return None

    def _update_resize_cursor(self, edges: Qt.Edges) -> None:
        """Show/clear the resize cursor as the pointer hovers near an edge."""
        app = QApplication.instance()
        if app is None:
            return
        shape = None if self.isMaximized() else self.edge_cursor_shape(edges)
        if shape is None:
            self._clear_resize_cursor()
            return
        cursor = QCursor(shape)
        if self._resize_cursor_active:
            app.changeOverrideCursor(cursor)
        else:
            app.setOverrideCursor(cursor)
            self._resize_cursor_active = True

    def _clear_resize_cursor(self) -> None:
        if not self._resize_cursor_active:
            return
        app = QApplication.instance()
        if app is not None:
            app.restoreOverrideCursor()
        self._resize_cursor_active = False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        # Cross-platform frameless resize: hover near an edge shows the resize
        # cursor; pressing there starts the OS resize loop. This never touches
        # WM_NCHITTEST, so it cannot turn interior pixels into a non-client area
        # (the HiDPI bug that made whole panels unclickable).
        et = event.type()
        if (
            isinstance(watched, QWidget)
            and watched.window() is self
            and isinstance(event, QMouseEvent)
            and not self.isMaximized()
        ):
            if (
                et == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                edges = self.edge_at(self.mapFromGlobal(event.globalPosition().toPoint()))
                if edges:
                    self._clear_resize_cursor()
                    handle = self.windowHandle()
                    if handle is not None:
                        handle.startSystemResize(edges)
                        return True
            elif (
                et == QEvent.Type.MouseMove
                and event.buttons() == Qt.MouseButton.NoButton
            ):
                self._update_resize_cursor(
                    self.edge_at(self.mapFromGlobal(event.globalPosition().toPoint()))
                )
        if et == QEvent.Type.Leave and self._resize_cursor_active:
            # Pointer left the window entirely -> drop the resize cursor.
            if not self.frameGeometry().contains(QCursor.pos()):
                self._clear_resize_cursor()
        return super().eventFilter(watched, event)

    # -- lifecycle -------------------------------------------------------------------
    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        # Close transition (spec module 4.3): fade the window out instead of
        # an abrupt disappearance. The first close request is deferred while a
        # short windowOpacity animation runs, then close() is re-issued and
        # the real teardown below executes. Skip-safe: with animations
        # disabled (tests) the fade branch never runs.
        if not self._close_fade_done and anim.animations_enabled() and self.isVisible():
            self._close_fade_done = True
            event.ignore()
            fade = QPropertyAnimation(self, b"windowOpacity", self)
            fade.setDuration(FADE_OUT_MS)
            fade.setStartValue(self.windowOpacity())
            fade.setEndValue(0.0)
            fade.setEasingCurve(QEasingCurve.Type.OutCubic)
            fade.finished.connect(self.close)
            self._fade_anim = fade
            fade.start()
            return
        self._clear_resize_cursor()
        self._persist_splitter_sizes()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._controller.close()
        _LOGGER.info("main window closed")
        super().closeEvent(event)
