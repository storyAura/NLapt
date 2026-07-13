"""Batch resume checkpoints (spec 9).

Long batch runs (notably LLM batches) record every successfully completed key
under a stable checkpoint id in ``<root>/.nlapt/checkpoints.json``. When the
same operation is started again after an interruption (including a crash),
the engine skips the recorded keys ("从断点继续"). The file is written
atomically after every change so a crash never corrupts it.

``make_checkpoint_id`` derives the id from the operation name, the *sorted*
key set, and a parameter repr — the same logical operation always maps to the
same id, while different parameters produce a different id.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path

from nlapt.core.errors import StorageError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.storage.atomic import atomic_write_text
from nlapt.storage.session import SESSION_DIR_NAME

_LOGGER = get_logger(__name__)

CHECKPOINT_FILE_NAME = "checkpoints.json"
CHECKPOINT_ENCODING = "utf-8"
JSON_INDENT = 2
# Separator between hashed fields; keys are sorted first for order independence.
HASH_FIELD_SEPARATOR = "\n"
# A corrupt checkpoint file is moved aside under this suffix instead of
# blocking dataset opening (checkpoints are an auxiliary resume cache).
CORRUPT_FILE_SUFFIX = ".corrupt"


def make_checkpoint_id(operation: str, keys: Sequence[str], params_repr: str) -> str:
    """Stable sha1 id for one logical batch operation.

    The id is independent of key order (keys are sorted before hashing) and
    stable across process restarts; different ``params_repr`` values yield
    different ids so changed parameters never resume a stale checkpoint.
    """
    if not isinstance(operation, str) or not operation.strip():
        raise ValidationError(f"operation must be a non-empty string, got {operation!r}")
    if isinstance(keys, str) or not isinstance(keys, Sequence):
        raise ValidationError(f"keys must be a sequence of strings, got {type(keys).__name__}")
    for key in keys:
        if not isinstance(key, str):
            raise ValidationError(f"keys must all be strings, got {key!r}")
    if not isinstance(params_repr, str):
        raise ValidationError(f"params_repr must be a string, got {type(params_repr).__name__}")
    material = HASH_FIELD_SEPARATOR.join([operation, *sorted(keys), params_repr])
    return hashlib.sha1(material.encode(CHECKPOINT_ENCODING)).hexdigest()


def _parse_checkpoint_payload(raw: object) -> dict[str, frozenset[str]]:
    """Convert decoded JSON to the internal mapping; ValueError when malformed."""
    if not isinstance(raw, dict):
        raise ValueError(f"expected a JSON object, got {type(raw).__name__}")
    data: dict[str, frozenset[str]] = {}
    for cid, keys in raw.items():
        if (
            not isinstance(cid, str)
            or not isinstance(keys, list)
            or not all(isinstance(k, str) for k in keys)
        ):
            raise ValueError(f"malformed entry for {cid!r}")
        data[cid] = frozenset(keys)
    return data


def _validate_id(checkpoint_id: str) -> None:
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValidationError(
            f"checkpoint_id must be a non-empty string, got {checkpoint_id!r}"
        )


class CheckpointStore:
    """Persist completed-key sets per checkpoint id (``.nlapt/checkpoints.json``).

    Thread-safe: the engine may mark keys from its collection loop while a UI
    thread queries progress. Internal state is replaced, never mutated in
    place, and every change is persisted atomically.
    """

    def __init__(self, root: Path) -> None:
        base = Path(root)
        if not base.is_dir():
            raise ValidationError(f"checkpoint root is not a directory: {base}")
        if not base.is_absolute():
            base = base.absolute()
        self._path = base / SESSION_DIR_NAME / CHECKPOINT_FILE_NAME
        self._lock = threading.RLock()
        self._data: dict[str, frozenset[str]] = self._load()

    # -- queries ---------------------------------------------------------------

    def completed(self, checkpoint_id: str) -> frozenset[str]:
        """Keys already completed under ``checkpoint_id`` (empty set if none)."""
        _validate_id(checkpoint_id)
        with self._lock:
            return self._data.get(checkpoint_id, frozenset())

    def has(self, checkpoint_id: str) -> bool:
        """True when a checkpoint is recorded for ``checkpoint_id``."""
        _validate_id(checkpoint_id)
        with self._lock:
            return checkpoint_id in self._data

    # -- updates ---------------------------------------------------------------

    def mark(self, checkpoint_id: str, key: str) -> None:
        """Record ``key`` as completed and persist immediately (crash-safe)."""
        _validate_id(checkpoint_id)
        if not isinstance(key, str) or not key:
            raise ValidationError(f"checkpoint key must be a non-empty string, got {key!r}")
        with self._lock:
            current = self._data.get(checkpoint_id, frozenset())
            if key in current:
                return
            self._data = {**self._data, checkpoint_id: current | {key}}
            self._persist()

    def clear(self, checkpoint_id: str) -> None:
        """Drop the checkpoint entirely (called after full completion). Idempotent."""
        _validate_id(checkpoint_id)
        with self._lock:
            if checkpoint_id not in self._data:
                return
            self._data = {
                cid: keys for cid, keys in self._data.items() if cid != checkpoint_id
            }
            self._persist()
        _LOGGER.debug("checkpoint %s cleared", checkpoint_id)

    # -- persistence -------------------------------------------------------------

    def _load(self) -> dict[str, frozenset[str]]:
        """Read the checkpoint file; missing file means no checkpoints.

        A corrupt or unreadable file never blocks construction (checkpoints
        are an auxiliary resume cache — losing them only forces a batch
        re-run): the store self-heals by logging a warning, moving the bad
        file aside to ``checkpoints.json.corrupt`` (best-effort), and
        starting empty.
        """
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding=CHECKPOINT_ENCODING))
            return _parse_checkpoint_payload(raw)
        except (OSError, ValueError) as exc:
            _LOGGER.warning(
                "checkpoint file %s is unreadable or corrupt (%s); "
                "moving it aside and starting with an empty checkpoint store",
                self._path,
                exc,
            )
            self._quarantine_corrupt_file()
            return {}

    def _quarantine_corrupt_file(self) -> None:
        """Best-effort rename of the corrupt file to ``*.corrupt`` so the next
        write starts fresh while the evidence stays inspectable."""
        target = self._path.with_name(self._path.name + CORRUPT_FILE_SUFFIX)
        try:
            target.unlink(missing_ok=True)
            self._path.replace(target)
        except OSError:
            _LOGGER.exception(
                "could not move corrupt checkpoint file %s aside", self._path
            )

    def _persist(self) -> None:
        """Atomically write current state as sorted, human-inspectable JSON."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(
                f"cannot create checkpoint directory {self._path.parent}: {exc}"
            ) from exc
        payload = {cid: sorted(keys) for cid, keys in sorted(self._data.items())}
        atomic_write_text(
            self._path, json.dumps(payload, ensure_ascii=False, indent=JSON_INDENT)
        )
