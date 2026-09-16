"""CHA 角色识别 bridge: gated CL Tagger download + async identify.

Downloads go through the process-wide download hub (Bearer token on the
gated repo). Inference runs on the worker pool; the ONNX engine is a
process-wide singleton released when the CHA wizard closes.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import shiboken6
from PySide6.QtCore import QObject, QThreadPool, Signal

from nlapt.core.errors import NLaptError
from nlapt.diagnostics import get_logger
from nlapt.local.catalog import (
    CHARACTER_CSV_FILENAME,
    CHARACTER_CSV_URL,
    TAGGER_FAMILY,
    TAGGER_FAMILY_ID,
    TAGGER_QUANT_LABEL,
    download_url,
    family_dir,
    find_family,
    quant_path,
)
from nlapt.local.character_index import (
    CharacterEntry,
    CharacterIndex,
    display_name,
    display_series,
    normalize_tag,
)
from nlapt.local.settings import LocalSettings, load_local_settings
from nlapt.local.tagger import REQUIRED_FILES, TagResult, TaggerEngine

from nlapt_gui.download_hub import (
    DOWNLOAD_OK,
    active_download,
    get_download_hub,
    launch_download_jobs,
)
from nlapt_gui.local_bridge import local_settings_path, models_dir_for
from nlapt_gui.workers import run_async

_LOGGER = get_logger(__name__)

MSG_NEED_HF_TOKEN = "请先填写 Hugging Face Token"
MSG_NOT_READY = "请先在 设置▸CHA标注 下载 CL Tagger 模型"
CHARACTER_MIN_PROB = 0.35
MAX_DETECTED = 8
REL_DETECTED = "detected"
REL_PARENT = "parent"
REL_CHILD = "child"
REL_SIBLING = "sibling"

_TAGGER_ENGINE: TaggerEngine | None = None
_TAGGER_KEY: tuple[tuple[str, str], ...] = ()
_TAGGER_LOCK = threading.Lock()
_CHAR_INDEX: CharacterIndex | None = None
_CHAR_INDEX_KEY: tuple[str, int, int] = ("", 0, 0)
_INDEX_LOCK = threading.Lock()


@dataclass(frozen=True)
class CharacterCandidate:
    """One pick in the 识别角色 menu (detected tag or a parent/child/sibling)."""

    tag: str
    display_name: str
    series_tag: str
    series_display: str
    prob: float
    relation: str
    post_count: int = 0


@dataclass(frozen=True)
class IdentifyResult:
    """Candidates for one reference image, detected first then relatives."""

    candidates: tuple[CharacterCandidate, ...] = ()


def _settings() -> LocalSettings:
    return load_local_settings(local_settings_path())


def tagger_files(settings: LocalSettings | None = None) -> dict[str, Path] | None:
    """basename → path when every required file + the CSV exist and are non-empty."""
    resolved = settings if settings is not None else _settings()
    family = find_family(TAGGER_FAMILY_ID)
    dests: dict[str, Path] = {}
    for spec in (family.quants[0], *family.extra_files):
        path = quant_path(models_dir_for(resolved), family, spec)
        if not path.is_file() or path.stat().st_size <= 0:
            return None
        dests[path.name] = path
    csv_path = family_dir(models_dir_for(resolved), family) / CHARACTER_CSV_FILENAME
    if not csv_path.is_file() or csv_path.stat().st_size <= 0:
        return None
    dests[CHARACTER_CSV_FILENAME] = csv_path
    if any(name not in dests for name in REQUIRED_FILES):
        return None
    return dests


def is_tagger_ready() -> bool:
    """True when the ONNX trio and the character CSV are on disk."""
    return tagger_files() is not None


def start_tagger_download(token: str) -> bool:
    """Download missing tagger files + CSV. False when busy or token empty."""
    if not token.strip():
        return False
    if active_download() is not None:
        return False
    family = TAGGER_FAMILY
    settings = _settings()
    base = models_dir_for(settings)

    def present(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > 0

    jobs: list[tuple[str, Path, int, str]] = []
    for spec in (family.quants[0], *family.extra_files):
        dest = quant_path(base, family, spec)
        if not present(dest):
            jobs.append((download_url(family.repo_id, spec.filename), dest, 0, ""))
    csv_dest = family_dir(base, family) / CHARACTER_CSV_FILENAME
    if not present(csv_dest):
        jobs.append((CHARACTER_CSV_URL, csv_dest, 0, ""))
    if not jobs:
        get_download_hub().finished.emit(
            TAGGER_FAMILY_ID, TAGGER_QUANT_LABEL, DOWNLOAD_OK, ""
        )
        return True
    headers = {"Authorization": f"Bearer {token.strip()}"}
    return launch_download_jobs(
        TAGGER_FAMILY_ID, TAGGER_QUANT_LABEL, jobs, None, headers=headers
    )


def get_tagger_engine(
    files: dict[str, Path],
    *,
    session_factory: object | None = None,
) -> TaggerEngine:
    """Process-wide engine; replaced when the file set changes."""
    global _TAGGER_ENGINE, _TAGGER_KEY
    model_files = {name: files[name] for name in REQUIRED_FILES}
    key = tuple(sorted((name, str(path)) for name, path in model_files.items()))
    factory = session_factory if callable(session_factory) else None
    with _TAGGER_LOCK:
        if _TAGGER_ENGINE is None or _TAGGER_KEY != key:
            _TAGGER_ENGINE = TaggerEngine(model_files, session_factory=factory)
            _TAGGER_KEY = key
        return _TAGGER_ENGINE


def get_character_index(path: Path) -> CharacterIndex:
    """Process-wide CSV index; reloaded when path / mtime / size change."""
    global _CHAR_INDEX, _CHAR_INDEX_KEY
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = (str(path), 0, 0)
    with _INDEX_LOCK:
        if _CHAR_INDEX is None or _CHAR_INDEX_KEY != key:
            _CHAR_INDEX = CharacterIndex.load(path)
            _CHAR_INDEX_KEY = key
        return _CHAR_INDEX


def release_tagger_engine() -> None:
    """Unload the ~2.5 GB ONNX session (wizard close)."""
    global _TAGGER_ENGINE, _TAGGER_KEY
    with _TAGGER_LOCK:
        engine = _TAGGER_ENGINE
        _TAGGER_ENGINE = None
        _TAGGER_KEY = ()
    if engine is not None:
        engine.unload()


def identify_characters(
    image_path: Path,
    *,
    engine: TaggerEngine | None = None,
    index: CharacterIndex | None = None,
) -> IdentifyResult:
    """Tag ``image_path`` and expand each character through the CSV family."""
    files = tagger_files()
    if files is None:
        raise NLaptError(MSG_NOT_READY)
    tagger = engine if engine is not None else get_tagger_engine(files)
    roster = index if index is not None else get_character_index(
        files[CHARACTER_CSV_FILENAME]
    )
    result = tagger.tag(image_path, threshold=CHARACTER_MIN_PROB)
    return _expand_candidates(result, roster)


def _expand_candidates(result: TagResult, index: CharacterIndex) -> IdentifyResult:
    fallback_series = result.copyrights[0].tag if result.copyrights else ""
    seen: set[str] = set()
    candidates: list[CharacterCandidate] = []
    for score in result.characters[:MAX_DETECTED]:
        detected = normalize_tag(score.tag)
        members = index.family(detected)
        if not members:
            if detected in seen:
                continue
            seen.add(detected)
            series = normalize_tag(fallback_series)
            candidates.append(
                CharacterCandidate(
                    tag=detected,
                    display_name=display_name(score.tag),
                    series_tag=series,
                    series_display=display_series(series),
                    prob=score.prob,
                    relation=REL_DETECTED,
                )
            )
            continue
        for entry in members:
            if entry.tag in seen:
                continue
            seen.add(entry.tag)
            series = entry.copyright or normalize_tag(fallback_series)
            candidates.append(
                CharacterCandidate(
                    tag=entry.tag,
                    display_name=display_name(entry.tag),
                    series_tag=series,
                    series_display=display_series(series),
                    prob=score.prob if entry.tag == detected else 0.0,
                    relation=_relation(entry, detected, index),
                    post_count=entry.post_count,
                )
            )
    return IdentifyResult(tuple(candidates))


def _relation(entry: CharacterEntry, detected: str, index: CharacterIndex) -> str:
    if entry.tag == detected:
        return REL_DETECTED
    cursor: CharacterEntry | None = index.lookup(detected)
    while cursor is not None and cursor.parent_tag:
        if cursor.parent_tag == entry.tag:
            return REL_PARENT
        cursor = index.lookup(cursor.parent_tag)
    cursor = entry
    while cursor.parent_tag:
        if cursor.parent_tag == detected:
            return REL_CHILD
        parent = index.lookup(cursor.parent_tag)
        if parent is None:
            break
        cursor = parent
    return REL_SIBLING


class TaggerBridge(QObject):
    """Async identify; download stays on the shared hub."""

    identify_ready = Signal(str, object)  # request_id, IdentifyResult
    identify_failed = Signal(str, str)  # request_id, message

    def __init__(
        self,
        *,
        pool: QThreadPool | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._pool = pool if pool is not None else QThreadPool.globalInstance()
        self._inflight: set[str] = set()

    def request_identify(self, request_id: str, image_path: Path) -> bool:
        """Run identify on the worker pool. False when ``request_id`` is busy."""
        if request_id in self._inflight:
            return False
        if not is_tagger_ready():
            self.identify_failed.emit(request_id, MSG_NOT_READY)
            return False
        self._inflight.add(request_id)
        path = Path(image_path)

        def work() -> IdentifyResult:
            return identify_characters(path)

        def done(payload: object) -> None:
            self._inflight.discard(request_id)
            if not shiboken6.isValid(self):
                return
            result = payload if isinstance(payload, IdentifyResult) else IdentifyResult()
            self.identify_ready.emit(request_id, result)

        def failed(message: str) -> None:
            self._inflight.discard(request_id)
            if not shiboken6.isValid(self):
                return
            self.identify_failed.emit(request_id, message)

        run_async(self._pool, work, on_done=done, on_error=failed)
        return True
