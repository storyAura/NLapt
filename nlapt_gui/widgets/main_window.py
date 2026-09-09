"""MainWindow - frameless assembly of the prototype rail + two-column body.

Structure (v1.14): 44px left icon rail | file list | preview + editor.
The old title bar and status bar are gone; window drag / ─ □ × live on the
preview info bar. The tools panel is a right-edge overlay toggled from the
rail.

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
    QRect,
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
    QHBoxLayout,
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
from nlapt_gui.controller import TOAST_NO_DATASET, TOAST_NO_SELECTION, TOAST_WARN, AppController
from nlapt_gui.image_tools_bridge import ImageToolsBridge
from nlapt_gui.resources import app_data_dir
from nlapt_gui.theme.logo import load_app_icon
from nlapt_gui.theme.manager import ThemeManager
from nlapt_gui.theme.tokens import EDITOR_H_RANGE, MIN_WINDOW, ThemeTokens
from nlapt_gui.translate_bridge import TranslateBridge
from nlapt_gui.vision_bridge import VisionBridge
from nlapt_gui.prompt_store import ENGINE_LLM, ENGINE_LOCAL
from nlapt_gui.widgets.batch_progress_dialog import BatchProgressDialog
from nlapt_gui.widgets.batch_scope_dialog import pick_scope_keys
from nlapt_gui.widgets.color_dialog import ColorSettingsDialog
from nlapt_gui.widgets.dialogs import ask_confirm
from nlapt_gui.widgets.duplicate_review_dialog import DuplicateReviewDialog
from nlapt_gui.widgets.editor_panel import EditorPanel
from nlapt_gui.widgets.file_panel import FilePanel
from nlapt_gui.widgets.flatten_alpha_dialog import FlattenAlphaDialog
from nlapt_gui.widgets.layered_infer_dialog import LayeredInferDialog
from nlapt_gui.widgets.preview_panel import HEADER_H, PreviewPanel, SplitterHandle
from nlapt_gui.widgets.toast import ToastOverlay
from nlapt_gui.widgets.toolbar_rail import ToolbarRail
from nlapt_gui.widgets.compare_infer_dialog import open_compare_infer
from nlapt_gui.widgets.tools_menu import (
    ACTION_FIND_DUPLICATES,
    ACTION_FLATTEN_ALPHA,
    ACTION_INFER_CHA,
    ACTION_INFER_COMPARE,
    ACTION_INFER_LLM,
    ACTION_INFER_LOCAL,
    ACTION_UNDO_IMAGE_OP,
    LABEL_INFER_CHA,
    LABEL_INFER_COMPARE,
    LABEL_INFER_LLM,
    LABEL_INFER_LOCAL,
)
from nlapt_gui.widgets.tools_panel import ToolsPanel

_LOGGER = get_logger(__name__)

# Frameless edge-resize grab margin (px). Wide enough to grab comfortably; a
# hover cursor makes the edge discoverable.
EDGE_MARGIN_PX = 8

WINDOW_TITLE = "NLapt"
OPEN_FOLDER_CAPTION = "打开数据集文件夹"
EXPORT_CAPTION = "导出数据集"
EXPORT_FILTER = "ZIP (*.zip)"
EXPORT_DIRTY_TITLE = "导出数据集"
EXPORT_DIRTY_TEXT = (
    "有未保存的标注。导出只打包磁盘上的文件。是否先全部保存再导出？"
)

# Body splitter: thin themed handle + default / min / max panel widths.
SPLITTER_HANDLE_W = 4
FILE_PANEL_DEFAULT_W = 260
TOOLS_PANEL_DEFAULT_W = 340
MIDDLE_DEFAULT_W = 760
MIDDLE_MIN_W = 360
PANEL_MIN_W = 200
PANEL_MAX_W = 16_777_215  # Qt QWIDGETSIZE_MAX - "large value" per contract
SPLITTER_SIZES_FILE = "body_splitter.json"

# Startup fade-in / close fade-out durations (skip-safe: end state is
# reachable synchronously and both collapse to no-ops in tests).
FADE_IN_MS = 240
FADE_OUT_MS = 140

# Maximize/restore transition: a quick windowOpacity dip masks the abrupt
# native geometry jump (frameless windows get no OS zoom animation).
STATE_FADE_OUT_MS = 90
STATE_FADE_IN_MS = 130
STATE_FADE_LOW = 0.55
# Restored window size as a share of the available screen when the saved
# normal geometry would still cover the whole screen (small displays).
RESTORE_SCREEN_SHARE = 0.86
# States treated as "zoomed" by the toggle (维持全屏也能退出).
_ZOOMED_STATES = Qt.WindowState.WindowMaximized | Qt.WindowState.WindowFullScreen

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
    if not isinstance(data, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in data
    ):
        return None
    if len(data) == 2:
        return [int(v) for v in data]
    if len(data) == 3:
        # v1.13 three-column file: keep the left width, drop the tools column.
        return [int(data[0]), int(data[1])]
    return None


def save_splitter_sizes(sizes: list[int], path: Path | None = None) -> None:
    """Persist the body-splitter pane sizes atomically as UTF-8 JSON."""
    target = path if path is not None else _splitter_sizes_path()
    atomic_write_text(target, json.dumps(list(sizes), ensure_ascii=False, indent=2))
    _LOGGER.debug("splitter sizes saved to %s", target)


class _BodySplitter(QSplitter):
    """Horizontal 2-pane splitter that pins the file-panel width on first layout.

    ``QSplitter.setSizes`` scales the requested sizes proportionally when their
    sum differs from the current width, so it cannot guarantee an exact initial
    side-panel width. This subclass records the desired left width and applies
    it (middle absorbs the remainder) on the first layout wide enough to honour
    it, then leaves the panes fully user-draggable.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._pending_left: int | None = None

    def set_initial_side_widths(self, left: int, _right: int = 0) -> None:
        """Request an exact left pane width for the first valid layout."""
        self._pending_left = left

    def _handles_width(self) -> int:
        return self.handleWidth() * max(0, self.count() - 1)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self._pending_left is None or self.count() != 2:
            return
        left = self._pending_left
        needed = left + self._handles_width() + MIDDLE_MIN_W
        if self.width() >= needed:
            middle = self.width() - left - self._handles_width()
            self.setSizes([left, middle])
            self._pending_left = None


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
        self.image_tools = ImageToolsBridge(controller, parent=self)

        self.rail = ToolbarRail(controller, theme_manager, self)
        self.file_panel = FilePanel(controller, tokens=theme_manager.tokens, parent=self)
        self.preview_panel = PreviewPanel(controller, tokens=theme_manager.tokens, parent=self)
        self.splitter = SplitterHandle(controller, tokens=theme_manager.tokens, parent=self)
        self.editor_panel = EditorPanel(
            controller, self.translate_bridge, self, vision_bridge=self.vision_bridge
        )
        self.tools_panel = ToolsPanel(controller, self)
        self.tools_panel.setAutoFillBackground(True)
        self.tools_panel.hide()

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.rail)
        root.addWidget(self._build_body(), 1)

        self.toast_overlay = ToastOverlay(self, tokens=theme_manager.tokens)
        controller.toast_requested.connect(self.toast_overlay.show_toast)

        # 推标进度窗口: self-wired to batch_started/progress/finished.
        self.batch_progress_dialog = BatchProgressDialog(controller, parent=self)

        self.editor_panel.setFixedHeight(_clamp_editor_h(controller.settings.editor_h))
        self.splitter.editor_h_changed.connect(self._apply_editor_height)

        # Disable batch/save buttons while an async batch or save runs.
        controller.busy_changed.connect(self._on_busy_changed)

        self.file_panel.open_folder_requested.connect(self.pick_folder)
        # 文件夹右键推标 -> batch captioning through the vision bridge.
        self.file_panel.infer_requested.connect(
            lambda keys, engine: self.vision_bridge.request_batch(tuple(keys), engine)
        )
        self.file_panel.layered_infer_requested.connect(self._open_layered_infer)
        self.file_panel.compare_infer_requested.connect(self._open_compare_infer)
        self._layered_dialog: LayeredInferDialog | None = None
        self.rail.open_folder_requested.connect(self.pick_folder)
        self.rail.settings_requested.connect(self.tools_panel.open_settings_dialog)
        self.rail.colors_requested.connect(self.open_color_settings)
        self.rail.export_requested.connect(self.export_dataset)
        self.rail.tools_toggled.connect(self.set_tools_open)
        self.rail.tool_action_requested.connect(self._dispatch_tool_action)
        self.rail.model_switch_requested.connect(self._controller.set_model_target)
        self.tools_panel.close_requested.connect(lambda: self.rail.set_tools_open(False))

        theme_manager.theme_changed.connect(self._on_theme_changed)
        self._install_shortcuts()

        # Fade-in / fade-out state (skip-safe for tests).
        self._did_fade_in = False
        self._fade_anim: QPropertyAnimation | None = None
        self._state_anim: QPropertyAnimation | None = None
        self._close_fade_done = False
        self._tools_anim: QPropertyAnimation | None = None
        self._tools_want_open = False

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    # -- body assembly -----------------------------------------------------------------
    def _build_body(self) -> QSplitter:
        """File list + preview/editor in a horizontal QSplitter."""
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

        # The file panel locks its own width in the constructor; release that
        # lock so the splitter can resize it (contract 3.1).
        self.file_panel.setMinimumWidth(PANEL_MIN_W)
        self.file_panel.setMaximumWidth(PANEL_MAX_W)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        saved = load_splitter_sizes(self._splitter_sizes_path)
        left = saved[0] if saved is not None else FILE_PANEL_DEFAULT_W
        splitter.setSizes([left, MIDDLE_DEFAULT_W])
        splitter.set_initial_side_widths(left)
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
        if len(sizes) != 2 or any(size <= 0 for size in sizes):
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
    def _dispatch_tool_action(self, action_id: str) -> None:
        """Launch a 工具-menu action (image tools or one of the 推标 flavours)."""
        if action_id == ACTION_FLATTEN_ALPHA:
            FlattenAlphaDialog(self._controller, self.image_tools, parent=self).exec()
            return
        if action_id == ACTION_FIND_DUPLICATES:
            DuplicateReviewDialog(
                self._controller,
                self.image_tools,
                self.file_panel.thumbnail_loader,
                parent=self,
            ).exec()
            return
        if action_id == ACTION_UNDO_IMAGE_OP:
            self.image_tools.restore_last_backup()
            return
        title = {
            ACTION_INFER_LLM: LABEL_INFER_LLM,
            ACTION_INFER_LOCAL: LABEL_INFER_LOCAL,
            ACTION_INFER_CHA: LABEL_INFER_CHA,
            ACTION_INFER_COMPARE: LABEL_INFER_COMPARE,
        }.get(action_id)
        if title is None:
            return
        keys = pick_scope_keys(self._controller, title, parent=self)
        if keys is None:
            return
        if not keys:
            self._controller.toast_requested.emit(TOAST_NO_SELECTION, TOAST_WARN)
            return
        if action_id == ACTION_INFER_CHA:
            self._open_layered_infer(keys)
            return
        if action_id == ACTION_INFER_COMPARE:
            self._open_compare_infer(keys)
            return
        engine = ENGINE_LOCAL if action_id == ACTION_INFER_LOCAL else ENGINE_LLM
        self.vision_bridge.request_batch(keys, engine)

    def _open_compare_infer(self, keys: object) -> None:
        """Show the 多对比推标 review window for a key list (right-click or 工具 popup)."""
        batch = tuple(keys) if isinstance(keys, (list, tuple)) else ()
        open_compare_infer(self._controller, batch, self.file_panel.thumbnail_loader, self)

    def _open_layered_infer(self, keys: object) -> None:
        """Show the 分层推标 wizard for the file-panel menu's key list."""
        batch = tuple(keys) if isinstance(keys, (list, tuple)) else ()
        if self._layered_dialog is not None:
            self._layered_dialog.close()
        dialog = LayeredInferDialog(
            self._controller,
            self.vision_bridge,
            batch,
            loader=self.file_panel.thumbnail_loader,
            parent=self,
        )
        self._layered_dialog = dialog
        dialog.exec()

    def pick_folder(self) -> None:
        """Directory dialog -> controller.open_dataset."""
        start = str(self._controller.root) if self._controller.root else str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, OPEN_FOLDER_CAPTION, start)
        if chosen:
            self._controller.open_dataset(Path(chosen))

    def export_dataset(self) -> None:
        """Ask for a zip path, optionally save dirty files, then export."""
        if self._controller.root is None:
            self._controller.toast_requested.emit(TOAST_NO_DATASET, TOAST_WARN)
            return
        save_first = False
        if self._controller.dirty_count() > 0:
            if not ask_confirm(self, EXPORT_DIRTY_TITLE, EXPORT_DIRTY_TEXT):
                return
            save_first = True
        suggested = str(self._controller.root) + ".zip"
        path, _filter = QFileDialog.getSaveFileName(
            self, EXPORT_CAPTION, suggested, EXPORT_FILTER
        )
        if path:
            self._controller.export_dataset(Path(path), save_first=save_first)

    def set_tools_open(self, open_: bool) -> None:
        """Show or hide the right-edge tools overlay (geometry slide)."""
        self._tools_want_open = open_
        self._stop_tools_anim()
        docked = self._tools_drawer_rect()
        offscreen = self._tools_offscreen_rect()
        if open_:
            start = self.tools_panel.geometry() if self.tools_panel.isVisible() else offscreen
            self.tools_panel.show()
            self._tools_anim = anim.slide_geometry(self.tools_panel, start, docked)
            if self._tools_anim is not None:
                self._tools_anim.finished.connect(self._on_tools_anim_finished)
            self.tools_panel.raise_()
            self.toast_overlay.raise_()
            return
        if not self.tools_panel.isVisible():
            return
        if not anim.animations_enabled():
            self.tools_panel.hide()
            return
        self._tools_anim = anim.slide_geometry(
            self.tools_panel, self.tools_panel.geometry(), offscreen
        )
        if self._tools_anim is None:
            self.tools_panel.hide()
            return
        self._tools_anim.finished.connect(self._on_tools_anim_finished)

    def _tools_drawer_rect(self) -> QRect:
        # Sit below the 46px preview info bar so ─ □ × stay visible and clickable.
        width = TOOLS_PANEL_DEFAULT_W
        top = HEADER_H
        return QRect(self.width() - width, top, width, max(0, self.height() - top))

    def _tools_offscreen_rect(self) -> QRect:
        return self._tools_drawer_rect().translated(TOOLS_PANEL_DEFAULT_W, 0)

    def _stop_tools_anim(self) -> None:
        animation = self._tools_anim
        self._tools_anim = None
        if animation is not None:
            animation.stop()

    def _finish_tools_anim(self) -> None:
        """Jump a running drawer animation to its end state (resize-safe)."""
        animation = self._tools_anim
        if animation is None:
            return
        animation.setCurrentTime(animation.duration())
        self._tools_anim = None

    def _on_tools_anim_finished(self) -> None:
        animation = self.sender()
        if animation is None or animation is not self._tools_anim:
            return
        self._tools_anim = None
        if not self._tools_want_open:
            self.tools_panel.hide()

    def _position_tools_drawer(self) -> None:
        self._finish_tools_anim()
        if not self.tools_panel.isVisible():
            return
        self.tools_panel.setGeometry(self._tools_drawer_rect())
        self.tools_panel.raise_()
        self.toast_overlay.raise_()

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
        add("Ctrl+Y", self._redo_shortcut)
        add("Ctrl+Shift+Z", self._redo_shortcut)
        add("Ctrl+Shift+C", lambda: self._controller.copy_caption())
        add("Alt+Up", lambda: self._controller.nav(-1))
        add("Alt+Down", lambda: self._controller.nav(1))
        add("Escape", self._close_tools_if_open)

    def _redo_shortcut(self) -> None:
        focus = QApplication.focusWidget()
        if isinstance(focus, _TEXT_INPUT_TYPES) and hasattr(focus, "redo"):
            focus.redo()
            return
        self._controller.redo_current()

    def _close_tools_if_open(self) -> None:
        if self.rail.tools_open():
            self.rail.set_tools_open(False)

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

    # -- maximize / restore with a transition ------------------------------------------
    def toggle_max_restore(self) -> None:
        """Maximize ⇄ restore with a short fade transition.

        Checks the raw window state so a window stuck in FULLSCREEN is also
        brought back to normal (``isMaximized()`` alone misses that state and
        made restore impossible).
        """
        if self.windowState() & _ZOOMED_STATES:
            self._animate_state_switch(self._restore_small)
        else:
            self._animate_state_switch(self.showMaximized)

    def _restore_small(self) -> None:
        self.showNormal()
        self._ensure_normal_fits_screen()

    def _ensure_normal_fits_screen(self) -> None:
        """Guarantee the restored window is visibly smaller than the screen.

        On small logical resolutions the fixed minimum size can equal the
        available screen, making 还原 look like a no-op; lower the minimum
        and center a RESTORE_SCREEN_SHARE-sized window instead.
        """
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        frame = self.frameGeometry()
        if frame.width() < avail.width() - 4 or frame.height() < avail.height() - 4:
            return  # already visibly smaller
        width = int(avail.width() * RESTORE_SCREEN_SHARE)
        height = int(avail.height() * RESTORE_SCREEN_SHARE)
        self.setMinimumSize(
            min(MIN_WINDOW[0], width), min(MIN_WINDOW[1], height)
        )
        self.resize(width, height)
        self.move(
            avail.x() + (avail.width() - width) // 2,
            avail.y() + (avail.height() - height) // 2,
        )

    def _animate_state_switch(self, apply_state) -> None:
        """Dip windowOpacity, switch the window state, fade back in."""
        if not anim.animations_enabled() or not self.isVisible():
            apply_state()
            return
        fade_out = QPropertyAnimation(self, b"windowOpacity", self)
        fade_out.setDuration(STATE_FADE_OUT_MS)
        fade_out.setStartValue(self.windowOpacity())
        fade_out.setEndValue(STATE_FADE_LOW)
        fade_out.setEasingCurve(QEasingCurve.Type.OutCubic)

        def switch() -> None:
            apply_state()
            fade_in = QPropertyAnimation(self, b"windowOpacity", self)
            fade_in.setDuration(STATE_FADE_IN_MS)
            fade_in.setStartValue(STATE_FADE_LOW)
            fade_in.setEndValue(1.0)
            fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._state_anim = fade_in
            fade_in.start()

        fade_out.finished.connect(switch)
        self._state_anim = fade_out
        fade_out.start()

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

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self.tools_panel.isVisible():
            self._position_tools_drawer()

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
