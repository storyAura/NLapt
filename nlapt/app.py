"""NLaptApp facade: UI-agnostic application core (contract section "nlapt.app").

Wires every subsystem together — EventBus, DebugManager tap, CaptionStore,
SearchIndex, SnapshotManager, SessionStore, CheckpointStore, BatchEngine,
OperationLog, per-file UndoStack, TokenEstimator — so a future UI only binds
to :attr:`NLaptApp.bus` events and calls facade methods.

Auto-save semantics (spec 2.3): :meth:`confirm` and :meth:`accept_suggestion`
persist the caption to disk; :meth:`open_dataset` restores crash-recovery
session state (unsaved drafts, pending suggestions, confirmed states);
:meth:`close` saves dirty files plus the session. After
:meth:`rollback_operation` the affected captions are reloaded from disk and
the search index refreshed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from nlapt.batch.checkpoint import CheckpointStore, make_checkpoint_id
from nlapt.batch.engine import BatchEngine, ProgressCallback
from nlapt.batch.progress import BatchController, BatchItemResult, BatchReport
from nlapt.captions.store import CaptionRecord, CaptionStore, PendingSuggestion
from nlapt.captions.tokens import TokenEstimator, get_default_estimator
from nlapt.core.config import (
    AppConfig,
    LLMProfile,
    resolve_text_profile,
    resolve_vision_profile,
)
from nlapt.core.errors import (
    LLMConfigError,
    SessionError,
    StorageError,
    ValidationError,
)
from nlapt.core.events import (
    EVT_ENCODING_ISSUES,
    EVT_FILE_SAVED,
    EVT_SAVE_FAILED,
    EVT_TXT_CONFLICT,
    EventBus,
)
from nlapt.core.models import DatasetScanResult, ImageFile
from nlapt.core.states import CaptionState, FileStatus
from nlapt.diagnostics import configure_logging, get_debug_manager, get_logger
from nlapt.history.oplog import ROLLBACK_KIND, OperationLog, OperationRecord
from nlapt.history.undo import UndoStack
from nlapt.indexing.search_index import SearchIndex
from nlapt.llm.base import create_client
from nlapt.llm.rewrite import RewriteService, RewriteSpec
from nlapt.llm.translate import TranslationCache, Translator
from nlapt.ops.base import MatchPreview, TextOperation
from nlapt.storage.export import export_dataset_zip
from nlapt.storage.paths import (
    STATE_BACKUPS_DIR_NAME,
    dataset_state_dir,
    migrate_legacy_dataset_state,
)
from nlapt.storage.scanner import scan_dataset
from nlapt.storage.session import SessionSnapshot, SessionStore
from nlapt.storage.snapshots import RestoreResult, SnapshotManager
from nlapt.storage.text_io import read_text_detect, write_caption
from nlapt.workflow.diff import DiffSegment, word_diff
from nlapt.workflow.review import next_pending, next_unconfirmed
from nlapt.workflow.suggestions import accept_for_edit as _wf_accept_for_edit
from nlapt.workflow.suggestions import accept_suggestion as _wf_accept
from nlapt.workflow.suggestions import reject_suggestion as _wf_reject

_LOGGER = get_logger(__name__)

# Operation-log record kinds written by the facade.
BATCH_KIND = "batch"
# Oplog kind for AI rewrite batches: no snapshot (suggestions never touch txt
# files); rollback clears the pending suggestions instead of restoring files.
AI_BATCH_KIND = "ai_batch"
# Pending-suggestion source used when the original source label is unknown.
SESSION_PENDING_SOURCE = "session"
# Batch operation name prefix for AI rewrite runs (also the pending source).
REWRITE_OPERATION_PREFIX = "rewrite"
# Batch operation name prefix for vision caption (推标) runs.
CAPTION_OPERATION_PREFIX = "caption"
# Item error when a captioner returns empty text (must never blank captions).
EMPTY_CAPTION_ERROR = "empty caption from model"
# Detail strings for batch item results.
NO_CHANGE_DETAIL = "no change"
# Max characters of suggestion text quoted in a batch item detail.
DETAIL_TEXT_LIMIT = 80

# Review-navigation selector signature (next_unconfirmed / next_pending).
_NextSelector = Callable[
    [Sequence[str], Mapping[str, FileStatus], str | None], str | None
]


class ScopeType(str, Enum):
    """Batch scope selector (spec 9: 选定作用域)."""

    CURRENT = "current"
    SELECTED = "selected"
    FILTERED = "filtered"
    ALL = "all"


@dataclass(frozen=True)
class SuggestionRollbackResult:
    """RestoreResult-equivalent summary of an AI-batch rollback.

    ``restored_files`` holds the caption keys whose pending suggestion was
    cleared. ``pre_restore_snapshot`` is always None: suggestions never touch
    txt files, so there is nothing to snapshot or restore on disk.
    """

    restored_files: tuple[str, ...]
    pre_restore_snapshot: None = None


class NLaptApp:
    """UI-agnostic application core.

    A future UI binds to :attr:`bus` events and calls these methods. All
    dataset-scoped collaborators are (re)created by :meth:`open_dataset`.
    """

    def __init__(self, *, config: AppConfig | None = None, log_dir: Path | None = None) -> None:
        if config is not None and not isinstance(config, AppConfig):
            raise ValidationError(f"config must be an AppConfig, got {type(config).__name__}")
        self._config = config if config is not None else AppConfig()
        if log_dir is not None:
            configure_logging(Path(log_dir))
        self.bus = EventBus()
        self._debug = get_debug_manager()
        self._untap = self._debug.tap_events(self.bus)
        self._estimator: TokenEstimator = get_default_estimator()
        self._store = CaptionStore(
            self.bus, revert_confirmed_on_edit=self._config.revert_confirmed_on_edit
        )
        self._index = SearchIndex()
        self.oplog = OperationLog(self.bus)
        self._lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._root: Path | None = None
        self._files: dict[str, ImageFile] = {}
        self._txt_to_key: dict[str, str] = {}
        self._undo: dict[str, UndoStack] = {}
        self._txt_conflicts: dict[str, tuple[str, ...]] = {}
        self._needs_conversion: dict[str, str] = {}
        self._snapshots: SnapshotManager | None = None
        self._session: SessionStore | None = None
        self._checkpoints: CheckpointStore | None = None
        self._engine: BatchEngine | None = None

    @property
    def config(self) -> AppConfig:
        return self._config

    def reload_config(self, config: AppConfig) -> None:
        """Replace the active configuration (e.g. after the user edits settings).

        Services built on demand (translator, rewrite) read the new config on
        their next construction. The caption store's edit-revert rule and an
        already-open dataset's snapshot retention keep their construction-time
        values until the next dataset is opened.
        """
        if not isinstance(config, AppConfig):
            raise ValidationError(f"config must be an AppConfig, got {type(config).__name__}")
        self._config = config

    @property
    def store(self) -> CaptionStore:
        """The in-memory caption store (read it; mutate via facade methods)."""
        return self._store

    @property
    def index(self) -> SearchIndex:
        return self._index

    @property
    def snapshots(self) -> SnapshotManager:
        """Snapshot manager of the open dataset (spec 2.4 snapshot management)."""
        self._require_dataset()
        assert self._snapshots is not None
        return self._snapshots

    # -- dataset lifecycle ---------------------------------------------------

    def open_dataset(self, root: Path) -> DatasetScanResult:
        """Scan ``root``, load captions, build the index, restore the session.

        Re-opening replaces all dataset-scoped state (fresh store, index,
        undo stacks, operation log). A corrupt session file is logged and
        reported to the DebugManager but never blocks opening the dataset.
        """
        result = scan_dataset(Path(root))
        self._root = result.root
        state_dir = dataset_state_dir(result.root)
        migrate_legacy_dataset_state(result.root, state_dir)
        self._snapshots = SnapshotManager(
            result.root,
            retention=self._config.snapshot_retention,
            backup_dir=state_dir / STATE_BACKUPS_DIR_NAME,
        )
        self._session = SessionStore(result.root, state_dir=state_dir)
        self._checkpoints = CheckpointStore(result.root, state_dir=state_dir)
        self._engine = BatchEngine(
            snapshots=self._snapshots, bus=self.bus, checkpoints=self._checkpoints
        )
        self._store = CaptionStore(
            self.bus, revert_confirmed_on_edit=self._config.revert_confirmed_on_edit
        )
        self._index = SearchIndex()
        self.oplog = OperationLog(self.bus)
        self._files = {}
        self._txt_to_key = {}
        self._undo = {}
        self._txt_conflicts = {}
        self._needs_conversion = {}
        # Group images by their resolved txt so several same-stem images can
        # never alias one caption file as independent editable records; the
        # first key in natural order (scan results are natural-sorted) becomes
        # the canonical record, the rest are reported via txt_conflicts().
        grouped: dict[str, list[ImageFile]] = {}
        for file in result.images:
            grouped.setdefault(self._txt_key(file), []).append(file)
        for txt_key, group in grouped.items():
            canonical = group[0]
            if len(group) > 1:
                excluded = tuple(item.key for item in group[1:])
                self._txt_conflicts[canonical.key] = excluded
                _LOGGER.warning(
                    "images %s share caption txt %r; only %r is editable",
                    (canonical.key, *excluded), txt_key, canonical.key,
                )
            self._load_caption(canonical, txt_key)
        self._restore_session()
        self._index.build(dict(self._store.texts()))
        self._announce_open_issues()
        _LOGGER.info(
            "dataset opened: %s (%d images, %d orphan txts)",
            result.root, len(result.images), len(result.orphan_txts),
        )
        return result

    def close(self) -> None:
        """Save dirty captions and the session, then release the event tap."""
        if self._root is not None:
            self.save_all_dirty()
            self.save_session()
        if self._untap is not None:
            self._untap()
            self._untap = None
        _LOGGER.info("app closed")

    def txt_conflicts(self) -> Mapping[str, tuple[str, ...]]:
        """Same-txt image groups found by :meth:`open_dataset`.

        Maps the canonical (editable) caption key to the same-stem image keys
        that were excluded from editing because they resolve to the same txt
        file (announced via ``EVT_TXT_CONFLICT`` at open time).
        """
        self._require_dataset()
        return dict(self._txt_conflicts)

    def keys_needing_conversion(self) -> Mapping[str, str]:
        """Caption keys whose txt file is not plain UTF-8 (spec 2.2).

        Maps the caption key to the detected source encoding; announced via
        ``EVT_ENCODING_ISSUES`` at open time. Convert with
        :meth:`convert_to_utf8`.
        """
        self._require_dataset()
        return dict(self._needs_conversion)

    def convert_to_utf8(self, keys: Sequence[str]) -> tuple[str, ...]:
        """Re-save the given captions as UTF-8 (no BOM); returns converted keys.

        Keys that do not need conversion are skipped; unknown keys raise
        ``ValidationError``. Each converted file is written atomically via
        the normal save path and removed from :meth:`keys_needing_conversion`.
        """
        self._require_dataset()
        if isinstance(keys, str) or not isinstance(keys, Sequence):
            raise ValidationError(
                f"keys must be a sequence of strings, got {type(keys).__name__}"
            )
        converted: list[str] = []
        for key in keys:
            self._require_file(key)
            if key not in self._needs_conversion:
                continue
            self.save(key)  # write_caption: atomic UTF-8, no BOM
            del self._needs_conversion[key]
            converted.append(key)
        if converted:
            _LOGGER.info("converted %d caption file(s) to UTF-8", len(converted))
        return tuple(converted)

    # -- editing ---------------------------------------------------------------

    def caption(self, key: str) -> CaptionRecord:
        """Current record for ``key`` (KeyError if unknown)."""
        self._require_dataset()
        return self._store.get(key)

    def edit(self, key: str, text: str) -> CaptionRecord:
        """Apply a manual edit: store transition + undo push + index update."""
        self._require_dataset()
        record = self._store.set_text(key, text)
        self._push_undo(key, text)
        self._index.update(key, text)
        return record

    def undo(self, key: str) -> str | None:
        """Undo the last edit of ``key``; returns the restored text or None."""
        return self._apply_history(key, "undo")

    def redo(self, key: str) -> str | None:
        """Redo the last undone edit of ``key``."""
        return self._apply_history(key, "redo")

    def confirm(self, key: str) -> str | None:
        """Confirm + auto-save, then return the next unconfirmed key (spec 6.4)."""
        self._require_dataset()
        self._store.confirm(key)
        self.save(key)
        return self._next(next_unconfirmed, key)

    def save(self, key: str) -> None:
        """Write the caption to disk atomically and clear its dirty flag.

        Publishes ``EVT_FILE_SAVED`` on success; on failure publishes
        ``EVT_SAVE_FAILED`` (with the reason), records the error, re-raises.
        """
        file = self._require_file(key)
        record = self._store.get(key)
        try:
            write_caption(file.txt_path, record.text)
        except StorageError as exc:
            self._debug.record_error("app.save", exc)
            self.bus.publish(EVT_SAVE_FAILED, key=key, reason=str(exc))
            raise
        self._store.mark_saved(key)
        self.bus.publish(EVT_FILE_SAVED, key=key)

    def save_all_dirty(self) -> tuple[str, ...]:
        """Save every dirty caption; returns the keys actually saved.

        Individual failures are logged and published as ``EVT_SAVE_FAILED``
        (inside :meth:`save`); the failing captions stay dirty so the session
        snapshot still carries them as recoverable drafts.
        """
        self._require_dataset()
        saved: list[str] = []
        for key in self._store.dirty_keys():
            try:
                self.save(key)
            except StorageError:
                _LOGGER.exception("could not save caption %r; kept as dirty draft", key)
            else:
                saved.append(key)
        return tuple(saved)

    def export_dataset(self, dest: Path) -> int:
        """Zip the open dataset's images and existing caption txts to ``dest``."""
        self._require_dataset()
        assert self._root is not None
        return export_dataset_zip(self._root, Path(dest), tuple(self._files.values()))

    def token_count(self, key: str) -> int:
        """Estimated CLIP token count of the current caption text (spec 6.5)."""
        self._require_dataset()
        return self._estimator.estimate(self._store.get(key).text)

    # -- search / scope ----------------------------------------------------------

    def search(self, query: str) -> tuple[str, ...]:
        """Full-text search over captions + filenames (spec 4.2)."""
        self._require_dataset()
        return self._index.query(query)

    def resolve_scope(
        self,
        scope: ScopeType,
        *,
        current: str | None = None,
        selected: Sequence[str] = (),
        filtered: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """Resolve a batch scope selector to concrete caption keys."""
        self._require_dataset()
        if not isinstance(scope, ScopeType):
            raise ValidationError(f"scope must be a ScopeType, got {scope!r}")
        if scope is ScopeType.ALL:
            return self._store.keys()
        if scope is ScopeType.CURRENT:
            if current is None:
                raise ValidationError("scope CURRENT requires a current key")
            self._require_file(current)
            return (current,)
        keys = tuple(selected) if scope is ScopeType.SELECTED else tuple(filtered)
        for key in keys:
            self._require_file(key)
        return keys

    # -- batch text operations -----------------------------------------------------

    def preview_operation(
        self, op: TextOperation, keys: Sequence[str]
    ) -> Mapping[str, tuple[MatchPreview, ...]]:
        """Per-file match previews; only keys with at least one match appear."""
        self._require_dataset()
        previews: dict[str, tuple[MatchPreview, ...]] = {}
        for key in keys:
            matches = op.preview(self._store.get(key).text)
            if matches:
                previews[key] = matches
        return previews

    def apply_operation(
        self,
        op: TextOperation,
        keys: Sequence[str],
        *,
        description: str,
        controller: BatchController | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> BatchReport:
        """Snapshot, apply ``op`` per file, save, and append to the oplog."""
        self._require_dataset()
        if not isinstance(description, str) or not description.strip():
            raise ValidationError(f"description must be a non-empty string, got {description!r}")
        changed: list[str] = []

        def worker(key: str) -> BatchItemResult:
            old = self._store.get(key).text
            hits = len(op.preview(old))
            new = op.apply(old)
            if new == old:
                return BatchItemResult(key=key, ok=True, detail=NO_CHANGE_DETAIL)
            self._store.set_text(key, new)
            self._push_undo(key, new)
            self._index.update(key, new)
            self.save(key)
            with self._lock:
                changed.append(key)
            return BatchItemResult(key=key, ok=True, detail=f"{hits} replacement(s)")

        assert self._engine is not None
        report = self._engine.run(
            operation=op.name,
            keys=keys,
            worker=worker,
            concurrency=1,
            controller=controller,
            on_progress=on_progress,
            take_snapshot=True,
        )
        self.oplog.append(
            description=description,
            affected_keys=self._txt_keys(changed),
            snapshot=report.snapshot,
            kind=BATCH_KIND,
        )
        return report

    # -- suggestions / review -------------------------------------------------------

    def set_suggestion(self, key: str, text: str, source: str) -> CaptionRecord:
        """Attach a pending AI suggestion (body text untouched, spec 10)."""
        self._require_dataset()
        if not isinstance(source, str) or not source.strip():
            raise ValidationError(f"source must be a non-empty string, got {source!r}")
        suggestion = PendingSuggestion(text=text, source=source, created_at=time.time())
        return self._store.set_pending(key, suggestion)

    def accept_suggestion(self, key: str) -> str | None:
        """Accept: pending becomes body, auto-save, return next pending key."""
        self._require_dataset()
        record = _wf_accept(self._store, key)
        self._push_undo(key, record.text)
        self._index.update(key, record.text)
        self.save(key)
        return self._next(next_pending, key)

    def reject_suggestion(self, key: str) -> str | None:
        """Reject: discard the pending suggestion, return next pending key."""
        self._require_dataset()
        _wf_reject(self._store, key)
        return self._next(next_pending, key)

    def accept_for_edit(self, key: str) -> CaptionRecord:
        """"编辑后接受": pending becomes a DRAFT body for manual editing.

        Not part of the minimal contract surface but required by spec 10;
        keeps undo history and search index consistent, does NOT save.
        """
        self._require_dataset()
        record = _wf_accept_for_edit(self._store, key)
        self._push_undo(key, record.text)
        self._index.update(key, record.text)
        return record

    def suggestion_diff(self, key: str) -> tuple[DiffSegment, ...]:
        """Word-level diff between the body text and the pending suggestion."""
        self._require_dataset()
        record = self._store.get(key)
        if record.pending is None:
            raise ValidationError(f"no pending suggestion for {key!r}")
        return word_diff(record.text, record.pending.text)

    # -- LLM batches ---------------------------------------------------------------

    def run_rewrite_batch(
        self,
        spec: RewriteSpec,
        keys: Sequence[str],
        *,
        service: RewriteService,
        controller: BatchController | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> BatchReport:
        """Run an AI rewrite over ``keys``; results land as pending suggestions.

        The body text is never overwritten (review happens via accept/reject),
        so no disk snapshot is needed; the run is checkpointed for resume
        (spec 9: 断点续跑).
        """
        self._require_dataset()
        if not isinstance(spec, RewriteSpec):
            raise ValidationError(f"spec must be a RewriteSpec, got {type(spec).__name__}")
        if not isinstance(service, RewriteService):
            raise ValidationError(f"service must be a RewriteService, got {type(service).__name__}")
        operation = f"{REWRITE_OPERATION_PREFIX}:{spec.type.value}"
        checkpoint_id = make_checkpoint_id(operation, keys, repr(spec))

        def worker(key: str) -> BatchItemResult:
            file = self._files[key]
            caption = self._store.get(key).text
            image: bytes | None = None
            if spec.use_vision:
                from nlapt.llm.vision import prepare_image

                image = prepare_image(file.image_path, max_edge=self._config.image_max_edge)
            result = service.run_one(
                spec, key=key, caption=caption, filename=file.image_path.name, image=image
            )
            self._store.set_pending(
                key,
                PendingSuggestion(text=result.result, source=operation, created_at=time.time()),
            )
            # Persist the suggestion BEFORE reporting success: the engine marks
            # the resume checkpoint only after the worker returns ok, so the
            # suggestion is always at least as durable as its checkpoint mark.
            self.save_session()
            return BatchItemResult(key=key, ok=True, detail=result.result[:DETAIL_TEXT_LIMIT])

        assert self._engine is not None
        report = self._engine.run(
            operation=operation,
            keys=keys,
            worker=worker,
            concurrency=self._config.request.concurrency,
            controller=controller,
            on_progress=on_progress,
            take_snapshot=False,
            checkpoint_id=checkpoint_id,
        )
        # Persist the session however the batch ended (completed/cancelled):
        # best-effort, the per-item saves above already provided durability.
        try:
            self.save_session()
        except SessionError as exc:
            self._debug.record_error("app.rewrite_batch", exc)
            _LOGGER.exception("could not save session after batch %r", operation)
        self.oplog.append(
            description=f"AI {operation} · {report.succeeded}/{len(keys)} files",
            affected_keys=tuple(r.key for r in report.results if r.ok),
            snapshot=None,
            kind=AI_BATCH_KIND,
        )
        return report

    def run_caption_batch(
        self,
        keys: Sequence[str],
        caption_fn: Callable[[str, Path], str],
        *,
        description: str,
        engine: str = "",
        concurrency: int = 1,
        controller: BatchController | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> BatchReport:
        """Batch 推标: caption every image and WRITE the results to disk.

        Unlike :meth:`run_rewrite_batch` (pending suggestions) this replaces
        caption bodies directly, so the run follows the destructive-batch
        safety model: snapshot before executing, per-file save, one
        ``kind="batch"`` oplog record for whole-batch rollback, and a resume
        checkpoint keyed by ``engine`` + keys. ``caption_fn(key, image_path)``
        runs concurrently (the slow LLM round-trips); store/index/disk
        mutations are serialized under the app lock. An empty result marks
        the item failed and leaves the caption untouched.
        """
        self._require_dataset()
        if not callable(caption_fn):
            raise ValidationError(
                f"caption_fn must be callable, got {type(caption_fn).__name__}"
            )
        if not isinstance(description, str) or not description.strip():
            raise ValidationError(
                f"description must be a non-empty string, got {description!r}"
            )
        for key in keys:
            self._require_file(key)
        operation = f"{CAPTION_OPERATION_PREFIX}:{engine or description}"
        checkpoint_id = make_checkpoint_id(operation, keys, engine)
        changed: list[str] = []

        def worker(key: str) -> BatchItemResult:
            text = caption_fn(key, self._files[key].image_path).strip()
            if not text:
                return BatchItemResult(key=key, ok=False, error=EMPTY_CAPTION_ERROR)
            with self._lock:
                if self._store.get(key).text == text:
                    return BatchItemResult(key=key, ok=True, detail=NO_CHANGE_DETAIL)
                self._store.set_text(key, text)
                self._push_undo(key, text)
                self._index.update(key, text)
                self.save(key)
                changed.append(key)
            return BatchItemResult(key=key, ok=True, detail=text[:DETAIL_TEXT_LIMIT])

        assert self._engine is not None
        report = self._engine.run(
            operation=operation,
            keys=keys,
            worker=worker,
            concurrency=concurrency,
            controller=controller,
            on_progress=on_progress,
            take_snapshot=True,
            checkpoint_id=checkpoint_id,
        )
        self.oplog.append(
            description=description,
            affected_keys=self._txt_keys(changed),
            snapshot=report.snapshot,
            kind=BATCH_KIND,
        )
        return report

    # -- llm service factories ---------------------------------------------------------

    TRANSLATE_TEMPLATE_KEYS = ("translate_en_zh", "translate_zh_en")

    def make_rewrite_service(self, *, trigger: str = "") -> RewriteService:
        """Build a :class:`RewriteService` from the current text / vision models.

        Wires the spec-8 request controls (``config.request``: timeout,
        retries with backoff, min interval) into every LLM call. The text
        and vision clients come from ``config.text_target`` /
        ``config.vision_target`` and may live on different APIs; a vision
        client is attached only when a vision model is chosen. Raises
        :class:`LLMConfigError` when no text model is configured.
        """
        text_profile = self._require_profile()
        vision_profile = resolve_vision_profile(self._config)
        vision_client = create_client(vision_profile) if vision_profile is not None else None
        merged = replace(
            text_profile,
            vision_model=vision_profile.vision_model if vision_profile is not None else "",
        )
        return RewriteService(
            text_client=create_client(text_profile),
            vision_client=vision_client,
            profile=merged,
            trigger=trigger,
            request=self._config.request,
        )

    def make_translator(self, *, cache: TranslationCache | None = None) -> Translator:
        """Build a :class:`Translator` from the current text model.

        Wires the spec-8 request controls and any user-edited translation
        prompt templates saved in ``config.custom_templates`` under the
        :data:`TRANSLATE_TEMPLATE_KEYS` names. Raises
        :class:`LLMConfigError` when no text model is configured.
        """
        profile = self._require_profile()
        overrides = {
            name: text
            for name, text in self._config.custom_templates.items()
            if name in self.TRANSLATE_TEMPLATE_KEYS
        }
        return Translator(
            create_client(profile),
            profile,
            cache=cache,
            templates=overrides or None,
            request=self._config.request,
        )

    def _require_profile(self) -> LLMProfile:
        profile = resolve_text_profile(self._config)
        if profile is None:
            raise LLMConfigError(
                "no text model configured — add an API profile, enable a model "
                "and set AppConfig.text_target before using LLM features"
            )
        return profile

    # -- history / snapshots ----------------------------------------------------------

    def rollback_operation(self, op_id: int) -> RestoreResult | SuggestionRollbackResult:
        """Roll back one logged operation.

        Snapshot-backed records restore the affected txt files from the
        snapshot and reload them into the store. AI-batch records
        (``kind="ai_batch"``, no snapshot — suggestions never touch txt files)
        clear the pending suggestions they produced instead and return a
        :class:`SuggestionRollbackResult`.
        """
        self._require_dataset()
        if not isinstance(op_id, int) or isinstance(op_id, bool):
            raise ValidationError(f"op_id must be an int, got {op_id!r}")
        record = next((r for r in self.oplog.records() if r.op_id == op_id), None)
        if record is not None and record.kind == AI_BATCH_KIND:
            return self._rollback_ai_batch(record)
        assert self._snapshots is not None
        result = self.oplog.rollback(op_id, self._snapshots)
        self._reload_from_disk(result.restored_files)
        return result

    def save_session(self) -> None:
        """Persist crash-recovery state: drafts, pending, per-file states.

        Serialized under a lock: concurrent batch workers persist after every
        item, and interleaved writers must not race the atomic session write.
        """
        self._require_dataset()
        assert self._session is not None
        with self._session_lock:
            drafts: dict[str, str] = {}
            pending: dict[str, str] = {}
            pending_sources: dict[str, str] = {}
            states: dict[str, str] = {}
            for key in self._store.keys():
                record = self._store.get(key)
                if record.dirty:
                    drafts[key] = record.text
                if record.pending is not None:
                    pending[key] = record.pending.text
                    pending_sources[key] = record.pending.source
                states[key] = record.state.value
            self._session.save(
                SessionSnapshot(
                    drafts=drafts,
                    pending=pending,
                    pending_sources=pending_sources,
                    states=states,
                    saved_at=time.time(),
                )
            )

    # -- internals ----------------------------------------------------------------

    def _load_caption(self, file: ImageFile, txt_key: str) -> None:
        """Load one canonical image/txt pair into store, undo, files, index maps."""
        text = ""
        if file.txt_exists:
            read = read_text_detect(file.txt_path)
            text = read.text
            if read.needs_conversion:
                self._needs_conversion[file.key] = read.encoding
        self._store.load(file.key, text)
        self._undo[file.key] = UndoStack(text)
        self._files[file.key] = file
        self._txt_to_key[txt_key] = file.key

    def _announce_open_issues(self) -> None:
        """Publish txt-conflict and encoding issues collected by open_dataset."""
        for canonical, excluded in self._txt_conflicts.items():
            self.bus.publish(
                EVT_TXT_CONFLICT,
                key=canonical,
                excluded=excluded,
                txt=self._txt_key(self._files[canonical]),
            )
        if self._needs_conversion:
            self.bus.publish(
                EVT_ENCODING_ISSUES,
                keys=tuple(self._needs_conversion),
                encodings=dict(self._needs_conversion),
            )

    def _rollback_ai_batch(self, record: OperationRecord) -> SuggestionRollbackResult:
        """Undo an AI batch: clear its pending suggestions (no disk restore).

        ``ai_batch`` records carry caption keys (not txt keys) because the
        suggestions only ever lived in the store/session, never in txt files.
        """
        cleared: list[str] = []
        for key in record.affected_keys:
            if key not in self._files:
                continue
            if self._store.get(key).pending is None:
                continue
            self._store.clear_pending(key)
            cleared.append(key)
        self.oplog.append(
            description=f"rollback of #{record.op_id}: {record.description}",
            affected_keys=tuple(cleared),
            snapshot=None,
            kind=ROLLBACK_KIND,
        )
        self.save_session()
        _LOGGER.info(
            "rolled back AI batch #%d: cleared %d pending suggestion(s)",
            record.op_id, len(cleared),
        )
        return SuggestionRollbackResult(restored_files=tuple(cleared))

    def _require_dataset(self) -> None:
        if self._root is None:
            raise ValidationError("no dataset is open; call open_dataset() first")

    def _require_file(self, key: str) -> ImageFile:
        self._require_dataset()
        file = self._files.get(key)
        if file is None:
            raise ValidationError(f"unknown caption key: {key!r}")
        return file

    def _push_undo(self, key: str, text: str) -> None:
        stack = self._undo.get(key)
        if stack is None:  # defensive: keys always get a stack at load time
            stack = UndoStack(text)
            self._undo[key] = stack
            return
        stack.push(text)

    def _apply_history(self, key: str, direction: str) -> str | None:
        """Shared undo/redo: move the stack, then sync store + index."""
        self._require_dataset()
        stack = self._undo.get(key)
        if stack is None:
            raise ValidationError(f"unknown caption key: {key!r}")
        text = stack.undo() if direction == "undo" else stack.redo()
        if text is None:
            return None
        self._store.set_text(key, text)
        self._index.update(key, text)
        return text

    def _next(self, selector: _NextSelector, current: str) -> str | None:
        keys = self._store.keys()
        states: dict[str, FileStatus] = {k: self._store.status(k) for k in keys}
        return selector(keys, states, current)

    def _txt_key(self, file: ImageFile) -> str:
        assert self._root is not None
        return file.txt_path.relative_to(self._root).as_posix()

    def _txt_keys(self, keys: Sequence[str]) -> tuple[str, ...]:
        """Map caption keys to root-relative txt keys (snapshot entry names)."""
        return tuple(self._txt_key(self._files[key]) for key in keys)

    def _reload_from_disk(self, txt_keys: Sequence[str]) -> None:
        """Re-read restored txts into the store and refresh the index."""
        for txt_key in txt_keys:
            key = self._txt_to_key.get(txt_key)
            if key is None:
                _LOGGER.debug("restored txt %r has no matching image; skipped", txt_key)
                continue
            file = self._files[key]
            text = read_text_detect(file.txt_path).text if file.txt_path.exists() else ""
            self._store.load(key, text)  # resets dirty/pending/state from disk truth
            self._undo[key] = UndoStack(text)
            self._index.update(key, text)
        _LOGGER.info("reloaded %d caption(s) after restore", len(txt_keys))

    def _restore_session(self) -> None:
        """Apply a saved session snapshot; corruption never blocks opening."""
        assert self._session is not None
        try:
            snapshot = self._session.load()
        except SessionError as exc:
            self._debug.record_error("app.session", exc)
            _LOGGER.warning("session restore skipped (corrupt file): %s", exc)
            return
        if snapshot is None:
            return
        known = set(self._store.keys())
        for key, text in snapshot.drafts.items():
            if key in known:
                self._store.set_text(key, text)
                self._push_undo(key, text)
        for key, state in snapshot.states.items():
            if key not in known or state != CaptionState.CONFIRMED.value:
                continue
            if self._store.get(key).text.strip():
                self._store.confirm(key)
        for key, text in snapshot.pending.items():
            if key not in known:
                continue
            source = snapshot.pending_sources.get(key, SESSION_PENDING_SOURCE)
            self._store.set_pending(
                key, PendingSuggestion(text=text, source=source, created_at=snapshot.saved_at)
            )
        _LOGGER.info(
            "session restored: %d draft(s), %d pending suggestion(s)",
            len(snapshot.drafts), len(snapshot.pending),
        )
