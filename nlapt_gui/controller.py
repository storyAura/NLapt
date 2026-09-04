"""AppController - the single bridge between the GUI and the nlapt core.

Widgets never import core packages or touch the filesystem directly: they
call controller methods and react to controller signals. Disk and batch
work runs on a QThreadPool through :mod:`nlapt_gui.workers`; core EventBus
events (which may fire on worker threads) are re-emitted as queued Qt
signals so every handler runs on the GUI thread.
"""

from __future__ import annotations

import time
from dataclasses import replace as _dc_replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

from PySide6.QtCore import QObject, QThreadPool, Signal
from PySide6.QtGui import QGuiApplication, QImageReader

from nlapt.app import NLaptApp
from nlapt.batch.progress import BatchController, BatchReport
from nlapt.captions.chips import join_chips, split_chips
from nlapt.captions.store import CaptionRecord
from nlapt.core.config import AppConfig, get_active_profile
from nlapt.core.errors import LLMConfigError, NLaptError, ValidationError
from nlapt.core.events import EVT_ENCODING_ISSUES, EVT_TXT_CONFLICT, Event
from nlapt.core.models import DatasetScanResult, ImageFile
from nlapt.core.states import CaptionState
from nlapt.diagnostics import get_logger
from nlapt.llm.base import LLMMessage, LLMRequest, create_client
from nlapt.llm.cleaning import clean_llm_output, ensure_not_refusal
from nlapt.llm.retry import MinIntervalLimiter
# Shared spec-8 pacing/retry helper; also reused by nlapt.llm.translate.
from nlapt.llm.rewrite import _paced_complete
from nlapt.llm.translate import TranslationCache, Translator
from nlapt.llm.vision import prepare_image
from nlapt.ops.base import TextOperation
from nlapt.ops.find_replace import FindReplaceOperation, FindReplaceSpec, scope_stats
from nlapt.ops.prefix_suffix import PrefixSuffixOperation, PrefixSuffixSpec

from nlapt_gui.history_model import FileHistory
from nlapt_gui.layered_prompts import count_words, estimate_tokens
from nlapt_gui.settings import UISettings, load_ui_settings, save_ui_settings
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

# -- vocabulary ----------------------------------------------------------------
FOLDER_ROOT_LABEL = "根目录"
EDITOR_MODES: tuple[str, ...] = ("chips", "sents", "text")
VIEW_MODES: tuple[str, ...] = ("list", "mid", "big")
SCOPES: tuple[str, ...] = ("current", "selected", "all")
MULTI_PREVIEW_LIMIT = 4
AS_TAG_JOINER = ", "

# -- toast kinds ---------------------------------------------------------------
TOAST_OK = "ok"
TOAST_WARN = "warn"
TOAST_ERR = "err"
TOAST_INFO = "info"

# -- exact UI strings (design/contract) ------------------------------------------
TOAST_NO_UNSAVED = "没有未保存的更改"
TOAST_UNDONE = "已撤销"
TOAST_NO_UNDO = "没有可撤销的操作"
TOAST_REDONE = "已重做"
TOAST_NO_REDO = "没有可重做的操作"
TOAST_NO_DATASET = "请先打开数据集"
TOAST_EXPORTED = "已导出 {n} 个文件"
TOAST_COPIED = "已复制标注文本"
TOAST_COPY_FAILED = "复制失败"
TOAST_FIND_EMPTY = "请输入查找内容"
TOAST_NO_SELECTION = "尚未选择任何图片"
TOAST_NO_MATCH = "没有找到匹配内容"
TOAST_PS_EMPTY = "请输入前缀/后缀内容"
LABEL_UNDO = "撤销"
LABEL_FIND_REPLACE = "查找替换"
POSITION_PREFIX = "prefix"
POSITION_SUFFIX = "suffix"
WORD_PREFIX = "添加前缀"
WORD_SUFFIX = "添加后缀"

# stat/meta separators from the design.
_DOT = " · "


def _format_size(size: int) -> str:
    """Human file size like the design's meta pill ('1.2 MB')."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


class AppController(QObject):
    """UI-facing facade over :class:`NLaptApp` plus pure-UI state.

    Owns: current key, ordered multi-selection (+ shift anchor), filter
    text, editor/view modes, the labeled UI history, and the async batch
    entry points used by the right-panel tools.
    """

    # -- signals -----------------------------------------------------------------
    dataset_opened = Signal(object)  # DatasetScanResult
    dataset_open_failed = Signal(str)
    current_changed = Signal(str)  # key ('' when none)
    caption_changed = Signal(str)  # key - text/dirty changed (any source)
    selection_changed = Signal()
    filter_changed = Signal(str)
    mode_changed = Signal(str)  # chips|sents|text
    view_mode_changed = Signal(str)  # list|mid|big
    files_saved = Signal(tuple)  # keys just saved
    batch_started = Signal(str, int)  # description, total — caption batch (推标) began
    batch_finished = Signal(str, object)  # description, BatchReport
    batch_progress = Signal(str, int, int)  # description, done, total
    toast_requested = Signal(str, str)  # text, kind: ok|warn|err|info
    busy_changed = Signal(bool)
    save_state_changed = Signal(str)  # "saving" | "saved" | "failed" (spec 2.3 三态)

    # Internal bridge: core EventBus events hop onto the GUI thread here.
    _core_event = Signal(str, object)

    def __init__(
        self,
        app: NLaptApp | None = None,
        *,
        settings: UISettings | None = None,
        pool: QThreadPool | None = None,
    ) -> None:
        super().__init__()
        self._app = app if app is not None else NLaptApp()
        self._settings = settings if settings is not None else load_ui_settings()
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._history = FileHistory(self)
        self._root: Path | None = None
        self._keys: tuple[str, ...] = ()
        self._image_files: dict[str, ImageFile] = {}
        self._meta_cache: dict[str, str] = {}
        self._filter = ""
        self._current: str | None = None
        self._anchor: str | None = None
        self._selected: set[str] = set()
        self._mode: str = EDITOR_MODES[0]
        self._view_mode: str = self._settings.view_mode
        self._translator: Translator | None = None
        self._translator_resolved = False
        self._rescanning = False
        self._batch_in_flight = False
        self._batch_controller: BatchController | None = None
        self._core_event.connect(self._on_core_event)
        self._unsubscribes = [
            self._app.bus.subscribe(EVT_ENCODING_ISSUES, self._bridge_event),
            self._app.bus.subscribe(EVT_TXT_CONFLICT, self._bridge_event),
        ]

    # -- foundation accessors (used by panels; not part of the widget contract) ----
    @property
    def app(self) -> NLaptApp:
        """The core facade (read-only access for advanced panel needs)."""
        return self._app

    @property
    def settings(self) -> UISettings:
        return self._settings

    @property
    def history(self) -> FileHistory:
        return self._history

    def update_settings(self, **changes: object) -> None:
        """Immutably merge persisted-UI-state changes (folder_open, editor_h...)."""
        self._settings = _dc_replace(self._settings, **changes)  # type: ignore[arg-type]

    # -- dataset -------------------------------------------------------------------
    def open_dataset(self, root: Path) -> None:
        """Scan + load asynchronously; emits dataset_opened / dataset_open_failed."""
        target = Path(root)
        self._rescanning = True
        self.busy_changed.emit(True)
        run_async(
            self._pool,
            self._app.open_dataset,
            target,
            on_done=self._finish_open,
            on_error=self._open_failed,
        )

    def refresh(self) -> None:
        """Rescan the currently open root (no-op when nothing is open).

        Unsaved drafts are persisted to the session file first so the core
        ``open_dataset`` restores them as dirty records instead of silently
        reverting to the on-disk text; the labeled UI history is preserved in
        :meth:`_finish_open`.
        """
        if self._root is None:
            return
        try:
            self._app.save_session()
        except NLaptError:
            _LOGGER.exception("could not persist session before refresh")
        self.open_dataset(self._root)

    @property
    def root(self) -> Path | None:
        return self._root

    def dataset_label(self) -> tuple[str, str]:
        """(folder name, full path string) for the left-panel header."""
        if self._root is None:
            return ("", "")
        return (self._root.name, str(self._root))

    def _finish_open(self, result: DatasetScanResult) -> None:
        previous_root = self._root
        self._root = result.root
        self._keys = self._app.store.keys()
        self._image_files = {img.key: img for img in result.images}
        self._meta_cache.clear()
        self._selected = {k for k in self._selected if k in self._image_files}
        if self._current not in self._keys:
            self._current = self._keys[0] if self._keys else None
        self._anchor = self._current
        # A rescan of the SAME root preserves each file's labeled history; a
        # fresh dataset (re)seeds every key with the 载入原始标注 entry.
        is_refresh = previous_root is not None and previous_root == result.root
        for key in self._keys:
            if is_refresh and self._history.entries(key):
                continue
            self._history.seed(key, self._app.caption(key).text)
        self._settings = _dc_replace(self._settings, last_root=str(result.root))
        self._rescanning = False
        self.busy_changed.emit(False)
        self.dataset_opened.emit(result)
        self.current_changed.emit(self._current or "")
        self.selection_changed.emit()

    def _open_failed(self, message: str) -> None:
        self._rescanning = False
        self.busy_changed.emit(False)
        self.dataset_open_failed.emit(message)
        self.toast_requested.emit(message, TOAST_ERR)

    def _bridge_event(self, event: Event) -> None:
        """EventBus handler (may run on a worker thread) -> queued Qt signal."""
        self._core_event.emit(event.name, dict(event.payload))

    def _on_core_event(self, name: str, payload: dict) -> None:
        if name == EVT_ENCODING_ISSUES:
            count = len(payload.get("keys", ()))
            self.toast_requested.emit(
                f"{count} 个文件使用非 UTF-8 编码,保存后将转换为 UTF-8", TOAST_WARN
            )
        elif name == EVT_TXT_CONFLICT:
            excluded = payload.get("excluded", ())
            self.toast_requested.emit(
                f"{len(excluded) + 1} 个同名图片共用标注文件,仅编辑 {payload.get('key', '')}",
                TOAST_WARN,
            )

    # -- data access -----------------------------------------------------------------
    def keys(self) -> tuple[str, ...]:
        return self._keys

    def filtered_keys(self) -> tuple[str, ...]:
        """Keys whose file name OR caption contains the filter (case-insensitive)."""
        query = self._filter.strip().lower()
        if not query:
            return self._keys
        matches = []
        for key in self._keys:
            name = key.rsplit("/", 1)[-1]
            if query in name.lower() or query in self.record(key).text.lower():
                matches.append(key)
        return tuple(matches)

    def set_filter(self, text: str) -> None:
        if text == self._filter:
            return
        self._filter = text
        self.filter_changed.emit(text)

    @property
    def filter_text(self) -> str:
        return self._filter

    def folders(self) -> tuple[str, ...]:
        """Distinct relative folders in key (natural) order; root -> 根目录."""
        seen: dict[str, None] = {}
        for key in self._keys:
            seen.setdefault(self.folder_of(key))
        return tuple(seen)

    def folder_of(self, key: str) -> str:
        return key.rsplit("/", 1)[0] if "/" in key else FOLDER_ROOT_LABEL

    def record(self, key: str) -> CaptionRecord:
        """The caption record for a key; placeholder while a rescan is in flight.

        Painting (thumbnail overlays, dirty dots) keeps running while
        ``open_dataset``/``refresh`` reload the core store on a worker
        thread; a temporarily missing key must not crash the paint path.
        """
        try:
            return self._app.caption(key)
        except KeyError:
            _LOGGER.debug("record(%r) requested mid-rescan; returning placeholder", key)
            return CaptionRecord(key=key, text="", state=CaptionState.UNLABELED)

    def image_path(self, key: str) -> Path:
        file = self._image_files.get(key)
        if file is None:
            raise ValidationError(f"unknown image key: {key!r}")
        return file.image_path

    def image_meta(self, key: str) -> str:
        """Cached '480 × 640 · PNG · 1.2 MB' meta string for the header pill."""
        cached = self._meta_cache.get(key)
        if cached is not None:
            return cached
        path = self.image_path(key)
        fmt = path.suffix.lstrip(".").upper()
        parts: list[str] = []
        size = QImageReader(str(path)).size()
        if size.isValid():
            parts.append(f"{size.width()} × {size.height()}")
        parts.append(fmt)
        try:
            parts.append(_format_size(path.stat().st_size))
        except OSError:
            _LOGGER.warning("could not stat image %s", path)
        meta = _DOT.join(parts)
        self._meta_cache[key] = meta
        return meta

    def image_format(self, key: str) -> str:
        """Uppercase extension without the dot (``PNG``)."""
        return self.image_path(key).suffix.lstrip(".").upper()

    def image_modified_label(self, key: str) -> str:
        """``修改于 yyyy-mm-dd`` from the image mtime, or empty on stat failure."""
        try:
            stamp = datetime.fromtimestamp(self.image_path(key).stat().st_mtime)
        except (OSError, ValidationError):
            return ""
        return f"修改于 {stamp.strftime('%Y-%m-%d')}"

    def segments(self, key: str) -> tuple[str, ...]:
        return split_chips(self.record(key).text)

    def char_seg_info(self, key: str) -> str:
        text = self.record(key).text
        return (
            f"{len(text)} 字符{_DOT}{len(split_chips(text))} 段"
            f"{_DOT}约 {estimate_tokens(text)} tokens{_DOT}{count_words(text)} 词"
        )

    # -- current / navigation -----------------------------------------------------------
    @property
    def current_key(self) -> str | None:
        return self._current

    def set_current(self, key: str) -> None:
        if key == self._current:
            return
        if key not in self._image_files:
            _LOGGER.warning("set_current ignored unknown key %r", key)
            return
        self._current = key
        self.current_changed.emit(key)

    def _nav_order(self) -> tuple[str, ...]:
        if self.multi_mode():
            return self.selected_keys()
        filtered = self.filtered_keys()
        return filtered if filtered else self._keys

    def nav(self, delta: int) -> None:
        """Prev/next within selection (multi) or filtered list; wraps around."""
        order = self._nav_order()
        if not order:
            return
        index = order.index(self._current) if self._current in order else 0
        self.set_current(order[(index + delta) % len(order)])

    def pos_label(self) -> str:
        """'i / n' per design (multi: index within the selection or '-')."""
        if self.multi_mode():
            selected = self.selected_keys()
            if self._current in selected:
                return f"{selected.index(self._current) + 1} / {len(selected)}"
            return f"- / {len(selected)}"
        filtered = self.filtered_keys()
        if self._current is None:
            return f"- / {len(filtered)}"
        if self._current in filtered:
            position = filtered.index(self._current) + 1
        elif self._current in self._keys:
            position = self._keys.index(self._current) + 1
        else:
            return f"- / {len(filtered)}"
        return f"{position} / {len(filtered)}"

    # -- selection --------------------------------------------------------------------
    def selected_keys(self) -> tuple[str, ...]:
        """Selected keys ordered by keys() order."""
        return tuple(k for k in self._keys if k in self._selected)

    def is_selected(self, key: str) -> bool:
        return key in self._selected

    def set_anchor(self, key: str) -> None:
        self._anchor = key

    def toggle_selected(self, key: str) -> None:
        """Checkbox toggle; selecting sets the shift anchor."""
        if key in self._selected:
            self._selected.discard(key)
        else:
            self._anchor = key
            self._selected.add(key)
        self._after_selection_change()

    def select_range_to(self, key: str) -> None:
        """Select anchor..key across the filtered list (Shift+click)."""
        flat = self.filtered_keys()
        if key not in flat:
            return
        anchor = self._anchor if self._anchor is not None else self._current
        anchor_index = flat.index(anchor) if anchor in flat else 0
        key_index = flat.index(key)
        low, high = sorted((anchor_index, key_index))
        self._selected.update(flat[low : high + 1])
        self._after_selection_change()

    def deselect(self, key: str) -> None:
        if key not in self._selected:
            return
        self._selected.discard(key)
        self.selection_changed.emit()

    def select_all(self) -> None:
        self._selected = set(self._keys)
        self._after_selection_change()

    def folder_keys(self, folder: str) -> tuple[str, ...]:
        """All keys inside one relative folder (根目录 for root files)."""
        return tuple(k for k in self._keys if self.folder_of(k) == folder)

    def unlabeled_keys(self, keys: Sequence[str]) -> tuple[str, ...]:
        """Subset of ``keys`` whose caption is 未标注 (empty body text)."""
        return tuple(
            k for k in keys if self.record(k).state is CaptionState.UNLABELED
        )

    def folder_selection_state(self, folder: str) -> str:
        """'all' | 'some' | 'none' — selection coverage of one folder."""
        keys = self.folder_keys(folder)
        selected = sum(1 for k in keys if k in self._selected)
        if keys and selected == len(keys):
            return "all"
        return "some" if selected else "none"

    def set_folder_selected(self, folder: str, selected: bool) -> None:
        """Select/deselect every file of a folder (文件夹多选 checkbox)."""
        keys = self.folder_keys(folder)
        if not keys:
            return
        if selected:
            self._anchor = keys[0]
            self._selected.update(keys)
            self._after_selection_change()
        else:
            self._selected.difference_update(keys)
            self.selection_changed.emit()

    def clear_selection(self) -> None:
        if not self._selected:
            return
        self._selected.clear()
        self.selection_changed.emit()

    def multi_mode(self) -> bool:
        return len(self._selected) >= 2

    def editor_keys(self) -> tuple[str, ...]:
        """Keys shown as editor blocks / preview grid (first 4 when multi)."""
        if self.multi_mode():
            return self.selected_keys()[:MULTI_PREVIEW_LIMIT]
        return (self._current,) if self._current is not None else ()

    def _after_selection_change(self) -> None:
        selected = self.selected_keys()
        if len(selected) >= 2 and self._current not in selected:
            self.set_current(selected[0])
        self.selection_changed.emit()

    # -- editing --------------------------------------------------------------------
    def set_caption(self, key: str, text: str, label: str) -> None:
        """Edit + push a labeled UI history entry + notify (no-op if unchanged)."""
        if self._rescanning or self.record(key).text == text:
            return
        self._app.edit(key, text)
        self._history.push(key, label, text)
        self.caption_changed.emit(key)

    def set_caption_no_history(self, key: str, text: str) -> None:
        """Edit without a history entry (history revert / live text typing)."""
        if self._rescanning or self.record(key).text == text:
            return
        self._app.edit(key, text)
        self.caption_changed.emit(key)

    def commit_segments(self, key: str, segs: Sequence[str], label: str) -> None:
        self.set_caption(key, join_chips(segs), label)

    def undo_current(self) -> None:
        """Undo = step the history cursor one entry older (nothing is deleted).

        The labeled UI history is the design's undo model; with the cursor
        semantics the newer entries stay listed in 历史记录 so the user can
        jump forward again from the panel.
        """
        key = self._current
        if key is None:
            return
        previous = self._history.step_older(key)
        if previous is None:
            self.toast_requested.emit(TOAST_NO_UNDO, TOAST_INFO)
            return
        self._app.edit(key, previous)
        self.caption_changed.emit(key)
        self.toast_requested.emit(TOAST_UNDONE, TOAST_INFO)

    def redo_current(self) -> None:
        """Redo = step the history cursor one entry newer."""
        key = self._current
        if key is None:
            return
        nxt = self._history.step_newer(key)
        if nxt is None:
            self.toast_requested.emit(TOAST_NO_REDO, TOAST_INFO)
            return
        self._app.edit(key, nxt)
        self.caption_changed.emit(key)
        self.toast_requested.emit(TOAST_REDONE, TOAST_INFO)

    def can_undo(self) -> bool:
        key = self._current
        if key is None:
            return False
        entries = self._history.entries(key)
        return self._history.current_index(key) + 1 < len(entries)

    def can_redo(self) -> bool:
        key = self._current
        if key is None:
            return False
        return self._history.current_index(key) > 0

    def export_dataset(self, dest: Path, *, save_first: bool = False) -> None:
        """Zip images + existing txts off-thread; toast on success or failure.

        ``save_first`` writes dirty captions to disk before packing so the
        archive matches the editor.
        """
        if self._root is None:
            self.toast_requested.emit(TOAST_NO_DATASET, TOAST_WARN)
            return
        if self._batch_in_flight or self._rescanning:
            self.toast_requested.emit(self.TOAST_BUSY, TOAST_WARN)
            return
        self._batch_in_flight = True
        self.busy_changed.emit(True)

        def job() -> object:
            saved: tuple[str, ...] = ()
            if save_first:
                saved = self._app.save_all_dirty()
            count = self._app.export_dataset(Path(dest))
            return (count, saved)

        def done(result: object) -> None:
            self._batch_in_flight = False
            self.busy_changed.emit(False)
            count, saved = result if isinstance(result, tuple) else (0, ())
            if saved:
                self.save_state_changed.emit("saved")
                self.files_saved.emit(tuple(saved))
                for key in saved:
                    self.caption_changed.emit(key)
            self.toast_requested.emit(TOAST_EXPORTED.format(n=int(count)), TOAST_OK)

        def failed(message: str) -> None:
            self._batch_in_flight = False
            self.busy_changed.emit(False)
            self.toast_requested.emit(message, TOAST_ERR)

        run_async(self._pool, job, on_done=done, on_error=failed)

    def save_current(self) -> None:
        """Save the current file off-thread (atomic disk write on the pool)."""
        key = self._current
        if key is None or self._rescanning:
            return
        if not self.record(key).dirty:
            self.toast_requested.emit(TOAST_NO_UNSAVED, TOAST_INFO)
            return
        txt_name = self._image_files[key].txt_path.name
        self.busy_changed.emit(True)
        self.save_state_changed.emit("saving")

        def job() -> object:
            self._app.save(key)
            return key

        def done(_saved: object) -> None:
            self.busy_changed.emit(False)
            self.save_state_changed.emit("saved")
            self.files_saved.emit((key,))
            self.toast_requested.emit(f"已保存 {txt_name}", TOAST_OK)
            self.caption_changed.emit(key)

        def failed(message: str) -> None:
            self.busy_changed.emit(False)
            self.save_state_changed.emit("failed")
            self.toast_requested.emit(message, TOAST_ERR)

        run_async(self._pool, job, on_done=done, on_error=failed)

    def save_all(self) -> None:
        """Save every dirty file off-thread; never blocks the GUI thread."""
        if self._rescanning:
            return
        if not self._app.store.dirty_keys():
            self.toast_requested.emit(TOAST_NO_UNSAVED, TOAST_INFO)
            return
        self.busy_changed.emit(True)
        self.save_state_changed.emit("saving")

        def done(saved: object) -> None:
            self.busy_changed.emit(False)
            self.save_state_changed.emit("saved")
            keys = tuple(saved) if isinstance(saved, (list, tuple)) else ()
            self.files_saved.emit(keys)
            self.toast_requested.emit(f"已保存 {len(keys)} 个文件", TOAST_OK)
            for key in keys:
                self.caption_changed.emit(key)

        def failed(message: str) -> None:
            self.busy_changed.emit(False)
            self.save_state_changed.emit("failed")
            self.toast_requested.emit(message, TOAST_ERR)

        run_async(self._pool, self._app.save_all_dirty, on_done=done, on_error=failed)

    def copy_caption(self, key: str | None = None) -> None:
        target = key if key is not None else self._current
        if target is None:
            return
        try:
            QGuiApplication.clipboard().setText(self._app.caption(target).text)
        except Exception:
            _LOGGER.exception("clipboard copy failed")
            self.toast_requested.emit(TOAST_COPY_FAILED, TOAST_ERR)
            return
        self.toast_requested.emit(TOAST_COPIED, TOAST_OK)

    # -- editor mode / view mode / counts ------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in EDITOR_MODES:
            raise ValidationError(f"mode must be one of {EDITOR_MODES}, got {mode!r}")
        if mode == self._mode:
            return
        self._mode = mode
        self.mode_changed.emit(mode)

    @property
    def view_mode(self) -> str:
        return self._view_mode

    def set_view_mode(self, view_mode: str) -> None:
        if view_mode not in VIEW_MODES:
            raise ValidationError(f"view_mode must be one of {VIEW_MODES}, got {view_mode!r}")
        if view_mode == self._view_mode:
            return
        self._view_mode = view_mode
        self._settings = _dc_replace(self._settings, view_mode=view_mode)
        self.view_mode_changed.emit(view_mode)

    def dirty_count(self) -> int:
        return len(self._app.store.dirty_keys())

    def stat_line(self) -> str:
        return (
            f"{len(self._keys)} 张图片{_DOT}已选 {len(self._selected)}"
            f"{_DOT}未保存 {self.dirty_count()}"
        )

    # -- tool scopes -------------------------------------------------------------------
    def scope_keys(self, scope: str) -> tuple[str, ...]:
        if scope not in SCOPES:
            raise ValidationError(f"scope must be one of {SCOPES}, got {scope!r}")
        if scope == "selected":
            return self.selected_keys()
        if scope == "all":
            return self._keys
        return (self._current,) if self._current is not None else ()

    def count_matches(
        self, find: str, case_sensitive: bool, scope: str, *, whole_word: bool = False
    ) -> tuple[int, int]:
        """(total hits, files with hits) over in-memory captions in scope.

        Matching is substring-level over the full caption (sentences
        included, not per comma segment); ``whole_word`` narrows hits to
        ``\\b`` word boundaries so "hair" no longer matches "hairband".
        """
        if not find:
            return (0, 0)
        texts = {key: self.record(key).text for key in self.scope_keys(scope)}
        spec = FindReplaceSpec(
            find=find, replace="", case_sensitive=case_sensitive, whole_word=whole_word
        )
        stats = scope_stats(spec, self._current, texts)
        return (stats.total_hits, stats.files_with_hits)

    def replace_all(
        self,
        find: str,
        replace: str,
        case_sensitive: bool,
        scope: str,
        *,
        whole_word: bool = False,
    ) -> None:
        """Batch find/replace via the core engine (snapshot + oplog), async."""
        if not find:
            self.toast_requested.emit(TOAST_FIND_EMPTY, TOAST_WARN)
            return
        if scope == "selected" and not self._selected:
            self.toast_requested.emit(TOAST_NO_SELECTION, TOAST_WARN)
            return
        keys = self.scope_keys(scope)
        total, _files = self.count_matches(
            find, case_sensitive, scope, whole_word=whole_word
        )
        if not keys or total == 0:
            self.toast_requested.emit(TOAST_NO_MATCH, TOAST_WARN)
            return
        spec = FindReplaceSpec(
            find=find, replace=replace, case_sensitive=case_sensitive, whole_word=whole_word
        )
        operation = FindReplaceOperation(spec)
        hits = {key: len(operation.preview(self.record(key).text)) for key in keys}
        description = f"{LABEL_FIND_REPLACE}「{find}」→「{replace}」"
        self._run_batch(
            operation,
            keys,
            description=description,
            make_label=lambda key: f"{LABEL_FIND_REPLACE} ×{hits[key]}",
            make_toast=lambda changed: f"已在 {len(changed)} 个文件中替换 {total} 处",
        )

    def apply_prefix_suffix(self, text: str, position: str, as_tag: bool, scope: str) -> None:
        """Batch prefix/suffix via the core engine (skip-if-present when 独立标签)."""
        if position not in (POSITION_PREFIX, POSITION_SUFFIX):
            raise ValidationError(f"position must be prefix|suffix, got {position!r}")
        stripped = text.strip()
        if not stripped:
            self.toast_requested.emit(TOAST_PS_EMPTY, TOAST_WARN)
            return
        if scope == "selected" and not self._selected:
            self.toast_requested.emit(TOAST_NO_SELECTION, TOAST_WARN)
            return
        keys = self.scope_keys(scope)
        if not keys:
            self.toast_requested.emit(TOAST_NO_SELECTION, TOAST_WARN)
            return
        is_prefix = position == POSITION_PREFIX
        spec = PrefixSuffixSpec(
            prefix=stripped if is_prefix else "",
            suffix="" if is_prefix else stripped,
            joiner=AS_TAG_JOINER if as_tag else "",
            skip_if_present=as_tag,
        )
        word = WORD_PREFIX if is_prefix else WORD_SUFFIX
        label = f"{word}「{stripped}」"
        self._run_batch(
            PrefixSuffixOperation(spec),
            keys,
            description=label,
            make_label=lambda _key: label,
            make_toast=lambda changed: f"已为 {len(changed)} 个文件{word}",
        )

    # Rejection toast when a batch/save is already running (re-entrancy guard).
    TOAST_BUSY = "正在处理,请稍候"

    def _run_batch(
        self,
        operation: TextOperation,
        keys: tuple[str, ...],
        *,
        description: str,
        make_label: Callable[[str], str],
        make_toast: Callable[[tuple[str, ...]], str],
    ) -> None:
        """Run app.apply_operation off-thread; refresh history/signals when done.

        Only one batch runs at a time: a second request while one is in flight
        is rejected with a toast, so concurrent QThreadPool workers never mutate
        the core store in parallel.
        """
        if self._batch_in_flight or self._rescanning:
            self.toast_requested.emit(self.TOAST_BUSY, TOAST_WARN)
            return
        before: Mapping[str, str] = {key: self.record(key).text for key in keys}
        self._batch_in_flight = True
        self.busy_changed.emit(True)

        def job() -> object:
            return self._app.apply_operation(operation, keys, description=description)

        def done(report: object) -> None:
            changed = tuple(
                key for key in keys if self.record(key).text != before[key]
            )
            for key in changed:
                self._history.push(key, make_label(key), self.record(key).text)
                self.caption_changed.emit(key)
            self._batch_in_flight = False
            self.busy_changed.emit(False)
            self.batch_finished.emit(description, report)
            self.toast_requested.emit(make_toast(changed), TOAST_OK)

        def failed(message: str) -> None:
            self._batch_in_flight = False
            self.busy_changed.emit(False)
            self.toast_requested.emit(message, TOAST_ERR)

        run_async(self._pool, job, on_done=done, on_error=failed)

    # -- batch vision captioning (推标) -------------------------------------------------
    TOAST_INFER_STARTED = "开始推标 {n} 张(右键文件夹可取消)"
    TOAST_INFER_DONE = "推标完成:成功 {ok} / 失败 {fail}(可在 历史记录 回滚)"
    TOAST_INFER_CANCELLED = "推标已取消:本次完成 {ok} 张(重新开始可续跑)"
    TOAST_INFER_CANCELLING = "正在取消推标(等待进行中的请求完成)…"

    def batch_running(self) -> bool:
        """Whether an async batch (推标 or text op) is currently in flight."""
        return self._batch_in_flight

    def cancel_batch(self) -> None:
        """Cooperatively cancel the running 推标 batch (completed items kept)."""
        controller = self._batch_controller
        if controller is None:
            return
        controller.cancel()
        self.toast_requested.emit(self.TOAST_INFER_CANCELLING, TOAST_INFO)

    def run_caption_batch(
        self,
        keys: Sequence[str],
        caption_fn: Callable[[str, Path], str],
        *,
        description: str,
        history_label: str,
        engine: str = "",
        concurrency: int = 1,
        on_finished: Callable[[object], None] | None = None,
    ) -> bool:
        """Batch 推标 via :meth:`NLaptApp.run_caption_batch`, off-thread.

        Results are written directly (snapshot + oplog rollback in the
        core); per-file labeled history entries are pushed here so every
        image keeps its undo trail. ``batch_started`` fires on the GUI
        thread before the work is queued (the progress window's show
        trigger); progress is re-emitted through ``batch_progress``;
        ``on_finished(report_or_None)`` always runs last (engine cleanup
        hook, e.g. stopping a batch-started server).
        """
        keys = tuple(keys)
        if self._batch_in_flight or self._rescanning:
            self.toast_requested.emit(self.TOAST_BUSY, TOAST_WARN)
            return False
        if not keys:
            self.toast_requested.emit(TOAST_NO_SELECTION, TOAST_WARN)
            return False
        before: Mapping[str, str] = {key: self.record(key).text for key in keys}
        batch_controller = BatchController()
        self._batch_controller = batch_controller
        self._batch_in_flight = True
        self.busy_changed.emit(True)
        self.toast_requested.emit(
            self.TOAST_INFER_STARTED.format(n=len(keys)), TOAST_INFO
        )
        self.batch_started.emit(description, len(keys))

        def progress(done: int, total: int, _key: str) -> None:
            # Worker-thread emit; Qt auto-queues delivery to GUI receivers.
            self.batch_progress.emit(description, done, total)

        def job() -> object:
            return self._app.run_caption_batch(
                keys,
                caption_fn,
                description=description,
                engine=engine,
                concurrency=concurrency,
                controller=batch_controller,
                on_progress=progress,
            )

        def finish_common() -> None:
            self._batch_in_flight = False
            self._batch_controller = None
            self.busy_changed.emit(False)

        def done(report: object) -> None:
            changed = tuple(key for key in keys if self.record(key).text != before[key])
            for key in changed:
                self._history.push(key, history_label, self.record(key).text)
                self.caption_changed.emit(key)
            finish_common()
            if isinstance(report, BatchReport) and report.status.value == "cancelled":
                self.toast_requested.emit(
                    self.TOAST_INFER_CANCELLED.format(ok=report.succeeded), TOAST_WARN
                )
            else:
                ok = report.succeeded if isinstance(report, BatchReport) else len(changed)
                fail = report.failed if isinstance(report, BatchReport) else 0
                self.toast_requested.emit(
                    self.TOAST_INFER_DONE.format(ok=ok, fail=fail),
                    TOAST_OK if fail == 0 else TOAST_WARN,
                )
            self.batch_finished.emit(description, report)
            if on_finished is not None:
                on_finished(report)

        def failed(message: str) -> None:
            finish_common()
            self.toast_requested.emit(message, TOAST_ERR)
            if on_finished is not None:
                on_finished(None)

        run_async(self._pool, job, on_done=done, on_error=failed)
        return True

    # -- services ---------------------------------------------------------------------
    def make_translator_or_none(self) -> Translator | None:
        """Translator from the active LLM profile; None when unconfigured. Cached."""
        if self._translator_resolved:
            return self._translator
        try:
            self._translator = self._app.make_translator(cache=TranslationCache())
        except LLMConfigError:
            _LOGGER.info("translation unavailable: no active LLM profile")
            self._translator = None
        self._translator_resolved = True
        return self._translator

    def invalidate_translator(self) -> None:
        """Drop the cached translator (call after the LLM profile changes)."""
        self._translator = None
        self._translator_resolved = False

    def make_vision_captioner_or_none(
        self, *, retry_sleep: Callable[[float], None] = time.sleep
    ) -> Callable[[Path, str, str], str] | None:
        """A ``(image_path, system, user_prompt) -> caption`` callable, or None.

        None when no active profile / base URL / vision model is configured
        (统一模式 saves the shared model into ``vision_model`` too, so that
        field alone decides vision availability). The callable blocks on the
        HTTP round-trip — run it on the worker pool.

        Requests run under the spec-8 controls (``config.request``): retries
        with exponential backoff on transient request errors and one shared
        min-interval limiter across every call of this captioner (a batch
        builds ONE captioner, so pacing spans the whole batch). Replies that
        read like a safety refusal raise ``LLMOutputError`` instead of being
        written as captions. ``retry_sleep`` is the backoff sleep, injectable
        so tests never really wait.
        """
        config = self._app.config
        profile = get_active_profile(config)
        if profile is None or not profile.base_url or not profile.vision_model:
            return None
        request_control = config.request
        max_edge = config.image_max_edge
        limiter = MinIntervalLimiter(request_control.min_interval)

        def caption(image_path: Path, system: str, user_prompt: str) -> str:
            client = create_client(profile)
            image = prepare_image(image_path, max_edge=max_edge)
            request = LLMRequest(
                messages=(
                    LLMMessage(role="user", text=user_prompt, images=(image,)),
                ),
                model=profile.vision_model,
                system=system if system else profile.system_prompt,
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                timeout=request_control.timeout,
            )
            response = _paced_complete(
                client,
                request,
                control=request_control,
                limiter=limiter,
                retry_sleep=retry_sleep,
            )
            return ensure_not_refusal(clean_llm_output(response.text))

        return caption

    def reload_config(self, config: AppConfig) -> None:
        """Point the core app at a freshly saved config (used by 设置).

        The controller owns the core-app boundary, so it applies the config
        through the public :meth:`NLaptApp.reload_config` seam and drops the
        cached translator — widgets must not touch core state directly.
        """
        self._app.reload_config(config)
        self.invalidate_translator()

    # -- lifecycle ---------------------------------------------------------------------
    def close(self) -> None:
        """Persist the crash-safe session + UI settings (no txt writes)."""
        if self._root is not None:
            try:
                self._app.save_session()
            except NLaptError:
                _LOGGER.exception("could not save session on close")
        try:
            # Theme/accent are persisted separately by ThemeManager; merge the
            # on-disk values so this save does not clobber a newer choice.
            disk = load_ui_settings()
            merged = _dc_replace(self._settings, theme=disk.theme, accent=disk.accent)
            save_ui_settings(merged)
        except NLaptError:
            _LOGGER.exception("could not save UI settings on close")
        for unsubscribe in self._unsubscribes:
            unsubscribe()
        self._unsubscribes = []
        _LOGGER.info("controller closed")
