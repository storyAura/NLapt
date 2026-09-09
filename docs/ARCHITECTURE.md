# NLapt Core Architecture Contract (v1)

This document is the **binding interface contract** for building the NLapt core (no UI).
Every module owner MUST implement the exact public signatures written here. Internal
helpers are free, but cross-module names/signatures must match this contract so that
independently developed modules integrate without changes.

Source spec: `NLapt.md` (requirements doc, Chinese). Target: Python 3.12, Windows-first,
cross-platform. UI is intentionally out of scope; the core must be UI-agnostic and fully
drivable through the facade (`nlapt/app.py`) so a future PySide6/Qt layer only binds signals.

## Global rules (all modules)

- **Language**: code, docstrings, comments in English. Docstrings concise.
- **Immutability**: public data types are `@dataclass(frozen=True)`. Functions that
  transform data return new values; the only sanctioned mutable containers are the
  explicit stores (`CaptionStore`, `SearchIndex`, `UndoStack`, `OperationLog`,
  `TranslationCache`, `CheckpointStore`, `EventBus`, `DebugManager`), which replace
  immutable records rather than mutating them in place.
- **File size**: keep files ~200–400 lines, hard max 800.
- **No hardcoded magic values**: module-level named constants.
- **Validation at boundaries**: validate user/external input (paths, regex, config JSON,
  LLM responses, zip entries) and raise typed exceptions from `nlapt.core.errors`.
- **Errors**: never silently swallow. Catch-log-reraise or convert to a typed error.
- **Dependencies**: stdlib only for core logic. `httpx` (LLM HTTP clients) and `Pillow`
  (image compression) are optional — import inside functions/guarded, raise
  `LLMConfigError` / `StorageError` with an actionable message when missing.
- **Logging**: every module gets its logger via `nlapt.diagnostics.get_logger(__name__)`.
  Never `print()`. Never log API keys (mask to `sk-***abc` style via `mask_secret`).
- **Text encoding**: all txt IO through `nlapt.storage.text_io` (UTF-8 no BOM, `\n`).
- **Atomic writes**: all file writes through `nlapt.storage.atomic`.
- **Type hints everywhere**; `from __future__ import annotations` at top of every module.
- **Tests**: pytest under `tests/<package>/test_<module>.py`, mirror package layout.
  Use `tmp_path` fixtures; no network, no sleeps (inject clocks/sleep functions).
- Threads: batch engine uses `concurrent.futures.ThreadPoolExecutor`; shared state
  guarded by locks inside the owning store.

## Package layout & ownership

```
nlapt/
  __init__.py            # __version__, re-export NLaptApp        [foundation]
  app.py                 # NLaptApp facade                        [integrator]
  core/                  # errors, events, models, states, config [foundation]
  diagnostics/           # logging, tracing, debug manager, crash [foundation]
  storage/               # atomic, text_io                        [foundation]
                         # scanner, snapshots, session            [agent B]
  captions/              # store, chips, sentences, tokens        [agent A]
  llm/                   # base, clients, retry, cleaning,
                         # templates, translate, rewrite, vision  [agent D]
  indexing/              # search_index, sorting                  [agent C]
  ops/                   # base, find_replace, prefix_suffix, dedup [agent C]
  batch/                 # engine, progress, checkpoint           [agent E]
  history/               # undo, oplog                            [agent E]
  workflow/              # suggestions, diff, review              [agent E]
tests/                   # mirrors nlapt/ (each agent owns tests for its modules)
docs/ARCHITECTURE.md     # this contract
pyproject.toml           # [foundation]
```

Working directory layout inside a dataset root (runtime artifacts):

- `.backups/` — zip snapshots (`SnapshotManager`)
- `.nlapt/session.json` — crash-recovery session state (`SessionStore`)
- `.nlapt/checkpoints.json` — batch resume checkpoints (`CheckpointStore`)

---

## nlapt.core.errors  [foundation]

```python
class NLaptError(Exception): ...            # base; message required
class ValidationError(NLaptError): ...      # bad user/config input
class StorageError(NLaptError): ...
class EncodingDetectionError(StorageError): ...   # attrs: path
class SnapshotError(StorageError): ...
class SessionError(StorageError): ...
class OperationError(NLaptError): ...
class RegexPatternError(OperationError): ...      # attrs: pattern, detail
class LLMError(NLaptError): ...
class LLMConfigError(LLMError): ...
class LLMRequestError(LLMError): ...              # network/API/HTTP failure
class LLMTimeoutError(LLMRequestError): ...
class LLMOutputError(LLMError): ...               # empty/unusable after cleaning
class BatchCancelledError(NLaptError): ...
```

## nlapt.core.events  [foundation]

```python
@dataclass(frozen=True)
class Event:
    name: str
    payload: Mapping[str, Any]   # stored as immutable MappingProxyType or dict copy

class EventBus:
    def subscribe(self, name: str | None, handler: Callable[[Event], None]) -> Callable[[], None]:
        """name=None subscribes to all events. Returns an unsubscribe callable."""
    def publish(self, name: str, **payload: Any) -> None:
        """Handler exceptions are logged and swallowed (bus never breaks the caller)."""
```

Well-known event names (constants in `events.py`): `EVT_CAPTION_CHANGED`,
`EVT_STATE_CHANGED`, `EVT_PENDING_SET`, `EVT_PENDING_CLEARED`, `EVT_FILE_SAVED`,
`EVT_SAVE_FAILED`, `EVT_BATCH_STARTED`, `EVT_BATCH_PROGRESS`, `EVT_BATCH_FINISHED`,
`EVT_SNAPSHOT_CREATED`, `EVT_OPLOG_APPENDED`.

## nlapt.core.states  [foundation]

```python
class CaptionState(str, Enum):
    UNLABELED = "unlabeled"
    DRAFT = "draft"
    CONFIRMED = "confirmed"

@dataclass(frozen=True)
class FileStatus:
    state: CaptionState
    has_pending: bool = False   # pending AI suggestion overlays state (spec 4.1)

def state_after_edit(current: CaptionState, new_text: str, *, revert_confirmed: bool = True) -> CaptionState:
    """Empty/whitespace text -> UNLABELED. Non-empty: CONFIRMED stays CONFIRMED when
    revert_confirmed is False, else -> DRAFT. UNLABELED/DRAFT + text -> DRAFT."""

def initial_state(text: str) -> CaptionState:
    """UNLABELED if empty/whitespace else DRAFT (used at scan time)."""
```

## nlapt.core.models  [foundation]

```python
@dataclass(frozen=True)
class ImageFile:
    key: str            # POSIX-style path of the image relative to dataset root; unique ID
    image_path: Path    # absolute
    txt_path: Path      # absolute; may not exist yet (unlabeled)
    txt_exists: bool
    mtime: float        # image file modified time at scan

@dataclass(frozen=True)
class DatasetScanResult:
    root: Path
    images: tuple[ImageFile, ...]
    orphan_txts: tuple[Path, ...]   # txt without image; ignored by default (spec 2.1)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
```

## nlapt.core.config  [foundation]

```python
@dataclass(frozen=True)
class LLMProfile:
    name: str
    api_type: str                 # "openai" | "anthropic" | "ollama" (registry-extensible)
    base_url: str
    api_key: str = ""             # stored locally only; must be masked in any log/export
    text_model: str = ""
    vision_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = ""

@dataclass(frozen=True)
class RequestControl:
    concurrency: int = 4
    min_interval: float = 0.0     # seconds between request starts
    timeout: float = 60.0
    max_retries: int = 2          # exponential backoff

@dataclass(frozen=True)
class AppConfig:
    profiles: tuple[LLMProfile, ...] = ()
    active_profile: str = ""
    request: RequestControl = RequestControl()
    image_max_edge: int = 1024
    revert_confirmed_on_edit: bool = True     # spec 4.1 supplementary rule
    snapshot_retention: int = 20
    prompt_orphan_cleanup: bool = False
    custom_templates: Mapping[str, str] = field(default_factory=dict)  # name -> template text
    trigger_presets: tuple[str, ...] = ()

def load_config(path: Path) -> AppConfig      # missing file -> defaults; invalid JSON/fields -> ValidationError
def save_config(path: Path, config: AppConfig) -> None    # atomic, UTF-8 JSON
def get_active_profile(config: AppConfig) -> LLMProfile | None
def mask_secret(value: str) -> str            # "" -> "", short -> "***", else "sk-***xyz9" style
def masked_config_dict(config: AppConfig) -> dict   # config as dict with api_key masked (for logs/bundles)
```

## nlapt.diagnostics  [foundation]  — built-in debug management for QA

```python
# logging_setup.py
LOGGER_ROOT = "nlapt"
def configure_logging(log_dir: Path | None = None, level: str = "INFO", json_lines: bool = False) -> logging.Logger
    # console handler + RotatingFileHandler (nlapt.log, 5MB x 3) when log_dir given.
    # Format includes timestamp, level, logger, funcName, lineno. Idempotent (no dup handlers).
def get_logger(name: str) -> logging.Logger      # child of "nlapt"

# tracer.py
def trace(logger: logging.Logger | None = None):   # decorator
    """DEBUG-log call name, truncated args repr, duration ms, and exceptions (re-raised)."""
def truncate_repr(value: Any, limit: int = 200) -> str

# debug_manager.py
class DebugManager:
    """Process-wide debug/QA hub. Get via get_debug_manager()."""
    def set_level(self, subsystem: str, level: str) -> None   # e.g. ("nlapt.llm", "DEBUG") at runtime
    def incr(self, counter: str, by: int = 1) -> None         # thread-safe metrics
    def metrics(self) -> Mapping[str, int]
    def record_error(self, source: str, error: BaseException) -> None   # keeps last N=50 error records
    def recent_errors(self) -> tuple[str, ...]
    def tap_events(self, bus: EventBus) -> Callable[[], None] # DEBUG-log every event; returns untap
    def export_bundle(self, target_zip: Path, *, log_dir: Path | None = None,
                      config: AppConfig | None = None, extra_files: Sequence[Path] = ()) -> Path:
        """QA bundle: logs + masked config + metrics + recent errors + env info. NEVER raw api keys."""
def get_debug_manager() -> DebugManager

# crash.py
def install_crash_handler(log_dir: Path) -> None   # sys.excepthook + threading.excepthook -> crash-YYYYMMDD-HHMMSS.log
```

## nlapt.storage.atomic / text_io  [foundation]

```python
# atomic.py
def atomic_write_bytes(path: Path, data: bytes) -> None
    # temp file in same directory, flush+fsync, os.replace; cleans temp on failure; StorageError on failure
def atomic_write_text(path: Path, text: str) -> None   # UTF-8 no BOM, newline="\n"

# text_io.py
CANDIDATE_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "big5", "shift_jis", "latin-1")
@dataclass(frozen=True)
class TextReadResult:
    text: str            # newlines normalized to "\n"
    encoding: str        # encoding actually used
    needs_conversion: bool   # True when not plain UTF-8 (caller may prompt & re-save, spec 2.2)
def read_text_detect(path: Path) -> TextReadResult    # EncodingDetectionError if nothing decodes
def write_caption(path: Path, text: str) -> None      # atomic UTF-8 no BOM \n
```

## nlapt.storage.scanner  [agent B]

```python
def scan_dataset(root: Path, *, recursive: bool = True) -> DatasetScanResult
```
Rules (spec 2.1): images by `IMAGE_EXTENSIONS` (extension case-insensitive); pair txt in
the **same directory** with the **same stem** (stem+ext matching case-insensitive);
recursive scan for kohya `10_conceptname` dirs; skip `.backups/`, `.nlapt/` and hidden
dirs; `key` = POSIX relative path; orphan txts collected (txt with no matching image).
Deterministic ordering (natural sort by key). `ValidationError` when root missing/not a dir.

## nlapt.storage.snapshots  [agent B]

```python
@dataclass(frozen=True)
class SnapshotInfo:
    path: Path
    created: datetime
    operation: str
    file_count: int

@dataclass(frozen=True)
class RestoreResult:
    restored_files: tuple[str, ...]      # keys (txt paths relative to root, POSIX)
    pre_restore_snapshot: SnapshotInfo

BACKUP_DIR_NAME = ".backups"

class SnapshotManager:
    def __init__(self, root: Path, *, retention: int = 20) -> None
    def create(self, operation: str) -> SnapshotInfo
        # zips ALL *.txt under root (recursive; excluding .backups/ and .nlapt/);
        # name: "YYYY-MM-DD_HHMM_<sanitized-op>.zip"; on collision append "_2", "_3"...
        # enforces retention (delete oldest beyond limit). Empty dataset -> zip with 0 entries is OK.
    def list_snapshots(self) -> tuple[SnapshotInfo, ...]     # newest first
    def preview_restore(self, snapshot: SnapshotInfo) -> int  # number of files it would restore
    def restore(self, snapshot: SnapshotInfo, *, only: Collection[str] | None = None) -> RestoreResult
        # 1) create pre-restore snapshot ("恢复前快照"); 2) extract txts (all or `only` keys)
        # SECURITY: reject zip entries escaping root (zip-slip) with SnapshotError.
```

## nlapt.storage.session  [agent B]

```python
SESSION_DIR_NAME = ".nlapt"

@dataclass(frozen=True)
class SessionSnapshot:
    drafts: Mapping[str, str]          # key -> unsaved caption text
    pending: Mapping[str, str]         # key -> pending suggestion text
    pending_sources: Mapping[str, str] # key -> source label (e.g. "rewrite:polish")
    states: Mapping[str, str]          # key -> CaptionState value
    saved_at: float                    # unix timestamp

class SessionStore:
    def __init__(self, root: Path) -> None
    def save(self, snapshot: SessionSnapshot) -> None       # atomic JSON
    def load(self) -> SessionSnapshot | None                # None if absent; SessionError if corrupt
    def clear(self) -> None
```

## nlapt.captions.store  [agent A]

```python
@dataclass(frozen=True)
class PendingSuggestion:
    text: str
    source: str          # e.g. "rewrite:polish", "vision:initial"
    created_at: float

@dataclass(frozen=True)
class CaptionRecord:
    key: str
    text: str
    state: CaptionState
    dirty: bool = False                       # unsaved changes
    pending: PendingSuggestion | None = None

class CaptionStore:
    """Central in-memory store: key -> CaptionRecord. Thread-safe (RLock).
    Replaces frozen records; publishes events on the injected EventBus."""
    def __init__(self, bus: EventBus, *, revert_confirmed_on_edit: bool = True) -> None
    def load(self, key: str, text: str) -> CaptionRecord           # initial load, state=initial_state(text), not dirty
    def get(self, key: str) -> CaptionRecord                       # KeyError if unknown
    def keys(self) -> tuple[str, ...]                              # insertion order
    def set_text(self, key: str, text: str) -> CaptionRecord      # applies state_after_edit, dirty=True, EVT_CAPTION_CHANGED
    def confirm(self, key: str) -> CaptionRecord                   # state=CONFIRMED (ValidationError if text empty), EVT_STATE_CHANGED
    def set_pending(self, key: str, suggestion: PendingSuggestion) -> CaptionRecord   # EVT_PENDING_SET
    def clear_pending(self, key: str) -> CaptionRecord             # EVT_PENDING_CLEARED
    def mark_saved(self, key: str) -> CaptionRecord                # dirty=False
    def dirty_keys(self) -> tuple[str, ...]
    def pending_keys(self) -> tuple[str, ...]
    def status(self, key: str) -> FileStatus
    def counts(self) -> Mapping[str, int]   # {"unlabeled": n, "draft": n, "confirmed": n, "pending": n}
    def texts(self) -> Mapping[str, str]    # snapshot copy: key -> current text
```

## nlapt.captions.chips  [agent A]  (spec 6.1)

```python
CHIP_SEPARATORS = (",", "，")
SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？")
def split_chips(text: str) -> tuple[str, ...]        # split on both commas, strip, drop empties
def join_chips(chips: Sequence[str]) -> str          # strip each, drop empties, join with ", "
def is_long_sentence(chip: str) -> bool              # contains any sentence-ending punctuation
def find_duplicates(chips: Sequence[str]) -> Mapping[str, tuple[int, ...]]   # exact text -> indices, only len>=2
def dedup_chips(chips: Sequence[str]) -> tuple[str, ...]                     # keep first occurrence
def edit_chip(chips: Sequence[str], index: int, new_text: str) -> tuple[str, ...]
    # replacing with text containing commas splits into multiple chips in place; empty -> removes
def move_chip(chips: Sequence[str], src: int, dst: int) -> tuple[str, ...]
def count_tag_in_texts(tag: str, texts: Mapping[str, str]) -> int
    # number of files whose chip list contains exact tag (spec 6.1 right-click stat)
```

## nlapt.captions.sentences  [agent A]  (spec 6.2)

```python
def split_sentences(text: str) -> tuple[str, ...]
    # naive split after . ! ? 。 ！ ？ (keep punctuation attached); known v1 limitation:
    # decimals/abbreviations may over-split — documented, not "fixed".
def join_sentences(sentences: Sequence[str]) -> str
    # sentence ending with ASCII punct -> single space before next; CJK punct -> no space
def merge_with_previous(sentences: Sequence[str], index: int) -> tuple[str, ...]  # ValidationError if index<=0
def move_sentence(sentences: Sequence[str], src: int, dst: int) -> tuple[str, ...]
def replace_sentence(sentences: Sequence[str], index: int, text: str) -> tuple[str, ...]
def delete_sentence(sentences: Sequence[str], index: int) -> tuple[str, ...]
```

## nlapt.captions.tokens  [agent A]  (spec 6.5)

```python
CLIP_TOKEN_LIMIT = 77
class TokenEstimator(Protocol):
    def estimate(self, text: str) -> int
class HeuristicClipEstimator:
    """Deterministic CLIP-BPE approximation (documented rules: word/punct/CJK-char based)."""
def get_default_estimator() -> TokenEstimator     # future hook for a real CLIP tokenizer plugin
def is_over_limit(count: int, limit: int = CLIP_TOKEN_LIMIT) -> bool
```

## nlapt.indexing.search_index  [agent C]  (spec 4.2)

```python
class SearchIndex:
    """In-memory full-text index over captions + filenames. Case-insensitive substring
    matching; thread-safe updates."""
    def build(self, entries: Mapping[str, str]) -> None      # key -> caption text (filenames derived from key)
    def update(self, key: str, text: str) -> None
    def remove(self, key: str) -> None
    def query(self, query: str) -> tuple[str, ...]
        # whitespace-separated terms; all must match (AND); term "-foo" means caption must
        # NOT contain foo (negation matches caption content only); positive terms match
        # filename OR caption; empty/blank query -> all keys; returns keys in index order.
```

## nlapt.indexing.sorting  [agent C]

```python
class SortBy(str, Enum): NAME = "name"; MTIME = "mtime"; TOKENS = "tokens"
def natural_key(name: str) -> tuple    # case-insensitive, digit runs compared numerically
def sort_keys(keys: Sequence[str], by: SortBy, *,
              mtimes: Mapping[str, float] | None = None,
              token_counts: Mapping[str, int] | None = None) -> tuple[str, ...]
```

## nlapt.ops.base  [agent C]

```python
@dataclass(frozen=True)
class MatchPreview:
    start: int
    end: int
    matched: str
    replacement: str
    context_before: str    # up to CONTEXT_CHARS=20 chars
    context_after: str

class TextOperation(Protocol):
    """Pure text transform. apply() must be deterministic and side-effect free."""
    name: str
    def preview(self, text: str) -> tuple[MatchPreview, ...]
    def apply(self, text: str) -> str

class OperationRegistry:
    """Extensibility point: register/create operations by name."""
    def register(self, name: str, factory: Callable[..., TextOperation]) -> None  # ValidationError on dup
    def create(self, name: str, **params: Any) -> TextOperation                   # ValidationError if unknown
    def names(self) -> tuple[str, ...]
def get_default_registry() -> OperationRegistry   # pre-registered: find_replace, prefix_suffix, dedup_tags
```

## nlapt.ops.find_replace  [agent C]  (spec 7.1)

```python
@dataclass(frozen=True)
class FindReplaceSpec:
    find: str
    replace: str
    case_sensitive: bool = False
    whole_word: bool = False
    regex: bool = False

def compile_spec(spec: FindReplaceSpec) -> re.Pattern[str]
    # ValidationError if find empty; RegexPatternError on bad regex (with detail);
    # non-regex: re.escape; whole_word wraps \b...\b; regex+whole_word also wraps.
class FindReplaceOperation:   # implements TextOperation; name = "find_replace"
    def __init__(self, spec: FindReplaceSpec) -> None
    def preview(self, text: str) -> tuple[MatchPreview, ...]
    def apply(self, text: str) -> str          # regex replace supports \1 group refs when spec.regex
    def apply_one(self, text: str, match_index: int) -> str   # replace a single hit (逐个替换)
@dataclass(frozen=True)
class ScopeHitStats:
    current_file_hits: int
    total_hits: int
    files_with_hits: int
def scope_stats(spec: FindReplaceSpec, current_key: str | None, texts: Mapping[str, str]) -> ScopeHitStats
```

## nlapt.ops.prefix_suffix  [agent C]  (spec 7.2)

```python
JOINERS = (", ", " ", "")
@dataclass(frozen=True)
class PrefixSuffixSpec:
    prefix: str = ""
    suffix: str = ""
    joiner: str = ", "                 # ValidationError if not in JOINERS
    skip_if_present: bool = True

class PrefixSuffixOperation:   # implements TextOperation; name = "prefix_suffix"
    # empty text + prefix -> text becomes prefix alone (no joiner); same for suffix.
def remove_exact_prefix(text: str, prefix: str, joiner: str) -> str
    # removes leading f"{prefix}{joiner}" exact match; also bare prefix when it is the whole text
def remove_exact_suffix(text: str, suffix: str, joiner: str) -> str
def make_trigger_add_op(trigger: str, joiner: str = ", ") -> TextOperation
def make_trigger_remove_op(trigger: str, joiner: str = ", ") -> TextOperation
```

## nlapt.ops.dedup  [agent C]

```python
class DedupTagsOperation:   # implements TextOperation; name = "dedup_tags"
    # split_chips -> dedup_chips -> join_chips; preview lists removed duplicates
```

## nlapt.llm.base  [agent D]  (spec 8)

```python
@dataclass(frozen=True)
class LLMMessage:
    role: str                          # "user" | "assistant"
    text: str
    images: tuple[bytes, ...] = ()     # JPEG/PNG bytes; only sent by vision-capable calls

@dataclass(frozen=True)
class LLMRequest:
    messages: tuple[LLMMessage, ...]
    model: str
    system: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: float = 60.0

@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str

class LLMClient(ABC):
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse   # raises LLMRequestError/LLMTimeoutError
    def test_connection(self, model: str) -> bool             # minimal request; raises on failure

CLIENT_REGISTRY: registry api_type -> factory(profile: LLMProfile) -> LLMClient
def register_client(api_type: str, factory) -> None
def create_client(profile: LLMProfile) -> LLMClient   # LLMConfigError for unknown type/missing base_url
```

`openai_client.py` / `anthropic_client.py` / `ollama_client.py`: thin httpx-based
implementations (guarded import; unit tests use `httpx.MockTransport`). Payload shapes:
OpenAI `/chat/completions` (images as `image_url` data URLs), Anthropic `/v1/messages`
(images as base64 source blocks), Ollama `/api/chat` (images as base64 list).
`mock.py`: `MockLLMClient(responses: Sequence[str] | Callable, fail_times: int = 0, ...)`
for all downstream tests.

## nlapt.llm.retry  [agent D]

```python
@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base_delay: float = 1.0
    multiplier: float = 2.0
def with_retry(fn: Callable[[], T], policy: RetryPolicy, *,
               sleep: Callable[[float], None] = time.sleep,
               retry_on: tuple[type[BaseException], ...] = (LLMRequestError,)) -> T
class MinIntervalLimiter:
    def __init__(self, min_interval: float, *, clock=time.monotonic, sleep=time.sleep) -> None
    def wait(self) -> None      # thread-safe
```

## nlapt.llm.cleaning  [agent D]  (spec 7.4 output cleaning)

```python
OUTPUT_CONSTRAINT = "Output only the final caption text. No explanations, quotes, or code blocks."
def clean_llm_output(raw: str) -> str
    # strip whitespace; markdown code fences (``` with optional lang); matching surrounding
    # quotes ("" '' “” 「」); leading label prefixes (case-insensitive): "caption:", "标注：",
    # "输出：", "output:", "result:"; collapse trailing whitespace.
    # raises LLMOutputError if result is empty.
```

## nlapt.llm.templates  [agent D]

```python
ALLOWED_VARIABLES = frozenset({"caption", "filename", "trigger"})
def render_template(template: str, *, caption: str = "", filename: str = "", trigger: str = "") -> str
    # {var} interpolation; unknown {var} -> ValidationError; "{{"/"}}" are literal braces.
DEFAULT_TEMPLATES: Mapping[str, str]   # keys: "polish", "rewrite", "expand", "condense", "translate_en_zh", "translate_zh_en"
    # condense template contains "{target_tokens}" handled by rewrite service via str replacement — keep
    # it as an allowed extra variable for that template only (render_template gains optional extra: Mapping[str,str]).
class TemplateStore:
    """Named custom templates persisted via AppConfig.custom_templates."""
    def __init__(self, initial: Mapping[str, str] | None = None) -> None
    def save(self, name: str, template: str) -> None    # ValidationError on empty name/template
    def get(self, name: str) -> str                     # KeyError if unknown
    def delete(self, name: str) -> None
    def names(self) -> tuple[str, ...]
    def as_dict(self) -> dict[str, str]
```

## nlapt.llm.translate  [agent D]  (spec 7.3)

```python
class Direction(str, Enum): EN_TO_ZH = "en->zh"; ZH_TO_EN = "zh->en"
def detect_direction(text: str) -> Direction    # CJK char ratio >= 0.3 -> ZH_TO_EN else EN_TO_ZH; empty -> EN_TO_ZH
@dataclass(frozen=True)
class CachedTranslation:
    source_text: str
    translated: str
    direction: Direction
    created_at: float
class TranslationCache:
    def get(self, key: str) -> CachedTranslation | None
    def put(self, key: str, entry: CachedTranslation) -> None
    def is_stale(self, key: str, current_text: str) -> bool   # True if missing or source_text != current
class Translator:
    def __init__(self, client: LLMClient, profile: LLMProfile, *, cache: TranslationCache | None = None,
                 templates: Mapping[str, str] | None = None, clock: Callable[[], float] = time.time) -> None
    def translate(self, key: str, text: str, direction: Direction | None = None) -> CachedTranslation
        # uses text model; caches; cleans output; LLMOutputError propagates
```

## nlapt.llm.vision  [agent D]  (spec 7.4 vision switch, 8)

```python
def prepare_image(path: Path, *, max_edge: int = 1024, quality: int = 85) -> bytes
    # Pillow (guarded import -> LLMConfigError if missing); downscale so max(w,h) <= max_edge;
    # convert to RGB JPEG bytes; ValidationError on unreadable file.
```

## nlapt.llm.rewrite  [agent D]  (spec 7.4)

```python
class RewriteType(str, Enum): POLISH="polish"; REWRITE="rewrite"; EXPAND="expand"; CONDENSE="condense"; CUSTOM="custom"

@dataclass(frozen=True)
class RewriteSpec:
    type: RewriteType
    custom_instruction: str = ""       # required when type=CUSTOM (ValidationError otherwise)
    target_tokens: int = 60            # used by CONDENSE
    use_vision: bool = False
    template_override: str = ""        # optional explicit template text

@dataclass(frozen=True)
class RewriteResult:
    key: str
    original: str
    result: str          # cleaned output

class RewriteService:
    def __init__(self, *, text_client: LLMClient, vision_client: LLMClient | None,
                 profile: LLMProfile, trigger: str = "") -> None
    def build_prompt(self, spec: RewriteSpec, *, caption: str, filename: str) -> str
        # template by type (or override/custom), render variables, append OUTPUT_CONSTRAINT.
    def run_one(self, spec: RewriteSpec, *, key: str, caption: str, filename: str,
                image: bytes | None = None) -> RewriteResult
        # vision on: requires image + vision_client (LLMConfigError if absent), uses profile.vision_model;
        # vision off: text client + profile.text_model. Output cleaned via clean_llm_output.
    def dry_run(self, spec: RewriteSpec, items: Sequence[tuple[str, str, str]],  # (key, caption, filename)
                *, sample_size: int = 5, rng: random.Random | None = None,
                image_provider: Callable[[str], bytes] | None = None) -> tuple[RewriteResult, ...]
```

## nlapt.batch  [agent E]  (spec 9)

```python
# progress.py
class BatchStatus(str, Enum): RUNNING="running"; PAUSED="paused"; CANCELLED="cancelled"; COMPLETED="completed"
@dataclass(frozen=True)
class BatchItemResult:
    key: str
    ok: bool
    detail: str = ""      # e.g. "3 replacements" or suggestion text summary
    error: str = ""
@dataclass(frozen=True)
class BatchReport:
    operation: str
    status: BatchStatus                  # CANCELLED keeps completed part (spec 9)
    results: tuple[BatchItemResult, ...]
    snapshot: SnapshotInfo | None
    succeeded: int
    failed: int
    failed_keys: tuple[str, ...]
class BatchController:
    def pause(self) -> None
    def resume(self) -> None
    def cancel(self) -> None
    def wait_if_paused(self) -> None     # blocks while paused; returns immediately when cancelled
    @property
    def cancelled(self) -> bool

# checkpoint.py
class CheckpointStore:                    # .nlapt/checkpoints.json, atomic writes
    def __init__(self, root: Path) -> None
    def completed(self, checkpoint_id: str) -> frozenset[str]
    def mark(self, checkpoint_id: str, key: str) -> None
    def clear(self, checkpoint_id: str) -> None
    def has(self, checkpoint_id: str) -> bool
def make_checkpoint_id(operation: str, keys: Sequence[str], params_repr: str) -> str   # stable sha1 hash

# engine.py
ProgressCallback = Callable[[int, int, str], None]      # done, total, current_key
class BatchEngine:
    def __init__(self, *, snapshots: SnapshotManager | None, bus: EventBus,
                 checkpoints: CheckpointStore | None = None) -> None
    def run(self, *, operation: str, keys: Sequence[str],
            worker: Callable[[str], BatchItemResult],
            concurrency: int = 1,
            controller: BatchController | None = None,
            on_progress: ProgressCallback | None = None,
            take_snapshot: bool = True,
            checkpoint_id: str | None = None) -> BatchReport
        # 1) snapshot (if enabled & snapshots given) BEFORE execution; 2) skip keys already in
        # checkpoint; 3) ThreadPoolExecutor(concurrency) — worker exceptions become failed
        # BatchItemResult (never crash the run); 4) honor pause/cancel between item dispatches;
        # 5) mark checkpoint after each success; 6) clear checkpoint when fully completed;
        # 7) publish EVT_BATCH_STARTED/PROGRESS/FINISHED.
```

## nlapt.history  [agent E]  (spec 7.5)

```python
# undo.py
UNDO_LIMIT = 50
GROUP_WINDOW_SECONDS = 0.6
class UndoStack:
    """Snapshot-based per-file undo. push() groups rapid consecutive edits (< GROUP_WINDOW_SECONDS)."""
    def __init__(self, initial: str, *, limit: int = UNDO_LIMIT,
                 clock: Callable[[], float] = time.monotonic) -> None
    def push(self, text: str) -> None            # no-op when text == current
    def undo(self) -> str | None                 # None when nothing to undo
    def redo(self) -> str | None
    @property
    def current(self) -> str
    def can_undo(self) -> bool
    def can_redo(self) -> bool

# oplog.py
@dataclass(frozen=True)
class OperationRecord:
    op_id: int
    timestamp: float
    description: str                 # e.g. 'batch replace "girl" -> "woman" · 9 files 17 hits'
    affected_keys: tuple[str, ...]
    snapshot: SnapshotInfo | None    # enables rollback
    kind: str                        # "batch" | "edit" | "rollback" | "restore" | ...
class OperationLog:
    def __init__(self, bus: EventBus) -> None
    def append(self, *, description: str, affected_keys: Sequence[str],
               snapshot: SnapshotInfo | None, kind: str) -> OperationRecord
    def records(self) -> tuple[OperationRecord, ...]      # newest first
    def rollback(self, op_id: int, snapshots: SnapshotManager) -> RestoreResult
        # OperationError if record unknown/no snapshot; restores ONLY affected keys' txts;
        # appends a new "rollback" record.
```

## nlapt.workflow  [agent E]  (spec 10)

```python
# diff.py
class DiffOp(str, Enum): EQUAL="equal"; INSERT="insert"; DELETE="delete"
@dataclass(frozen=True)
class DiffSegment:
    op: DiffOp
    text: str
def word_diff(old: str, new: str) -> tuple[DiffSegment, ...]
    # word-level (split on whitespace/CJK chars, keep separators); difflib.SequenceMatcher;
    # adjacent same-op segments merged; "".join(equal+delete)==old, "".join(equal+insert)==new.

# suggestions.py  — accept/reject glue over CaptionStore
def accept_suggestion(store: CaptionStore, key: str) -> CaptionRecord
    # pending -> set_text(pending.text) -> clear_pending; OperationError if no pending.
def reject_suggestion(store: CaptionStore, key: str) -> CaptionRecord
def accept_for_edit(store: CaptionStore, key: str) -> CaptionRecord
    # "编辑后接受": pending becomes DRAFT body text, pending cleared, NOT confirmed.

# review.py
def next_unconfirmed(keys: Sequence[str], states: Mapping[str, FileStatus], current: str | None) -> str | None
    # next key after `current` (wrap-around) whose state != CONFIRMED; None if all confirmed.
def next_pending(keys: Sequence[str], states: Mapping[str, FileStatus], current: str | None) -> str | None
```

## nlapt.app — NLaptApp facade  [integrator]

```python
class ScopeType(str, Enum): CURRENT="current"; SELECTED="selected"; FILTERED="filtered"; ALL="all"

class NLaptApp:
    """UI-agnostic application core. A future UI binds to `bus` events and calls these methods."""
    def __init__(self, *, config: AppConfig | None = None, log_dir: Path | None = None) -> None
    # wiring: EventBus, DebugManager tap, CaptionStore, SearchIndex, SnapshotManager,
    # SessionStore, CheckpointStore, BatchEngine, OperationLog, UndoStack per file, TokenEstimator.

    def open_dataset(self, root: Path) -> DatasetScanResult   # scan, load captions (encoding detect),
        # build index, restore session (drafts/pending) if present
    def close(self) -> None                                    # save dirty + session
    # editing
    def caption(self, key: str) -> CaptionRecord
    def edit(self, key: str, text: str) -> CaptionRecord       # store + undo push + index update
    def undo(self, key: str) -> str | None                     # applies to store/index too
    def redo(self, key: str) -> str | None
    def confirm(self, key: str) -> str | None                  # confirm + save + return next unconfirmed key
    def save(self, key: str) -> None                           # write_caption + mark_saved + EVT_FILE_SAVED
    def save_all_dirty(self) -> tuple[str, ...]
    def token_count(self, key: str) -> int
    # search/scope
    def search(self, query: str) -> tuple[str, ...]
    def resolve_scope(self, scope: ScopeType, *, current: str | None = None,
                      selected: Sequence[str] = (), filtered: Sequence[str] = ()) -> tuple[str, ...]
    # batch text ops
    def preview_operation(self, op: TextOperation, keys: Sequence[str]) -> Mapping[str, tuple[MatchPreview, ...]]
    def apply_operation(self, op: TextOperation, keys: Sequence[str], *,
                        description: str, controller: BatchController | None = None,
                        on_progress: ProgressCallback | None = None) -> BatchReport
        # snapshot -> apply per file (store.set_text) -> save -> oplog append
    # suggestions / review
    def set_suggestion(self, key: str, text: str, source: str) -> CaptionRecord
    def accept_suggestion(self, key: str) -> str | None        # accept + save + next_pending
    def reject_suggestion(self, key: str) -> str | None
    def suggestion_diff(self, key: str) -> tuple[DiffSegment, ...]
    # llm batches
    def run_rewrite_batch(self, spec: RewriteSpec, keys: Sequence[str], *,
                          service: RewriteService, controller: BatchController | None = None,
                          on_progress: ProgressCallback | None = None) -> BatchReport
        # results land as pending suggestions (never overwrite body), checkpointed for resume
    # history / snapshots
    def rollback_operation(self, op_id: int) -> RestoreResult  # restore + reload affected captions
    def save_session(self) -> None
```

Notes for the integrator: after any restore/rollback, affected captions must be re-read
from disk into the store and the index refreshed. `confirm()` and `accept_suggestion()`
save to disk (auto-save triggers, spec 2.3).

---

## pyproject.toml  [foundation]

- `[project] name="nlapt", version="0.1.0", requires-python=">=3.11"`, no hard deps;
  `[project.optional-dependencies] llm=["httpx"], images=["Pillow"], dev=["pytest","pytest-cov"]`.
- setuptools build backend; packages found under `nlapt*`.
- `[tool.pytest.ini_options] testpaths=["tests"]`.

## Test commands (Windows)

- Own modules: `python -m pytest tests/<pkg> -q`
- Full suite: `python -m pytest -q`
- Coverage: `python -m pytest --cov=nlapt --cov-report=term-missing -q` (target ≥ 80%)

---

## v1.1 addendum — additive changes from the review/fix cycle

All confirmed review findings (`docs/REVIEW_FINDINGS.json`, decisions in
`docs/FIXPLAN.md`) were fixed with **additive or internal** changes only:

- `core.config.mask_secret`: partial reveal only for secrets of length ≥ 12; shorter
  values return `"***"`.
- `core.events`: new constants `EVT_TXT_CONFLICT`, `EVT_ENCODING_ISSUES`.
- `captions.store.CaptionStore.set_text_forced_state(key, text, state)`: set text with an
  explicit target state (used by `accept_for_edit` to force DRAFT regardless of
  `revert_confirmed_on_edit`).
- `storage.snapshots.SnapshotManager.restore()`: reads and validates all target-zip bytes
  before creating the pre-restore snapshot; retention pruning never deletes the zip being
  restored.
- `storage.text_io.read_text_detect()`: a UTF-8 BOM with a non-UTF-8 body is stripped
  before candidate detection (body encoding reported, `needs_conversion=True`).
- `batch.checkpoint.CheckpointStore`: self-heals corrupt `checkpoints.json` (warning +
  rename to `checkpoints.json.corrupt` + empty store) instead of raising.
- `batch.engine`: pause blocks only new dispatches; finished in-flight futures keep being
  drained (progress/events/checkpoint marks fire while paused).
- `ops.prefix_suffix`: `skip_if_present` matches on a boundary (`text == prefix` or
  `text.startswith(prefix + joiner)`; suffix mirrored). Raw semantics kept for `joiner=""`.
- `llm.cleaning`: surrounding quotes are stripped only when they wrap the whole string.
- `llm.rewrite.RewriteService` / `llm.translate.Translator`: new keyword-only params
  `request: RequestControl | None`, `retry_sleep`, `limiter` — wire spec-8 timeout /
  retries (exponential backoff) / min-interval into every LLM call.
- `NLaptApp` additions:
  - `store` / `index` / `snapshots` read-only properties (contract alignment).
  - `make_rewrite_service(trigger=...)` / `make_translator(cache=...)`: build services
    from the active profile with `config.request` wired (raise `LLMConfigError` when no
    active profile).
  - `txt_conflicts()`: duplicate-stem images sharing one txt — first key (natural order)
    is the canonical editable record, the rest are excluded and reported
    (`EVT_TXT_CONFLICT`).
  - `keys_needing_conversion()` / `convert_to_utf8(keys)`: spec 2.2 detect-and-convert
    flow (`EVT_ENCODING_ISSUES` published on open).
  - `run_rewrite_batch` persists each pending suggestion to the session **before** its
    checkpoint mark (crash-safe resume), and logs the batch with `kind="ai_batch"`.
  - `rollback_operation` on an `ai_batch` record clears the produced pending suggestions
    (returns `SuggestionRollbackResult`); snapshot-backed records keep the restore path.

---

## v1.6 addendum — local inference module (additive)

New stdlib-only package `nlapt/local` (no UI, no hard deps) plus additive
error types in `nlapt.core.errors`: `LocalInferenceError(NLaptError)`,
`DownloadError`, `DownloadCancelledError`, `LocalServerError`.

- `catalog.py` — hand-curated GGUF catalog, snapshot-dated
  (`CATALOG_SNAPSHOT_DATE`): `ModelSeries` (大系列) → `ModelFamily` (小系列)
  → `QuantFile` (量化档). File sizes AND SHA256 digests (LFS oids) are exact
  values from the HuggingFace API; download counts are display-only
  snapshots. Repo ids are pattern-validated (`REPO_ID_PATTERN`) and family
  ids are directory-safe (`SAFE_ID_PATTERN`, checked in `family_dir` —
  defense in depth against a future non-hardcoded catalog source). Series:
  **Gemma 4**
  (unsloth conversions — top downloads AND ship the vision mmproj),
  **Gemma 4 Heretic** (highest-download uncensored derivatives; the 26B MoE
  conversion has no mmproj → `vision=False`), **ToriiGate 0.5**
  (Qwen3.5-4B based, MIT), **JoyCaption Beta One** (Llama-3.1-8B LLaVA).
  Helpers: `find_family` / `find_quant` / `recommended_quant` /
  `download_url` (segment-quoted) / `repo_page_url`; local layout is
  namespaced per family (`family_dir` / `quant_path` / `mmproj_path`) so
  identical basenames across repos (e.g. `mmproj-F16.gguf`) cannot collide.
  `kv_bytes_per_token` is a documented coarse heuristic per family.
- `hardware.py` — `detect_hardware()` NEVER raises: RAM via
  `GlobalMemoryStatusEx` (Windows) / sysconf + `/proc/meminfo` (POSIX),
  NVIDIA VRAM via `nvidia-smi` CSV (missing binary → no GPU), CPU cores.
  On Windows `nvidia-smi` is resolved ONLY from the driver's fixed install
  paths (System32 / NVSMI) — never PATH or the CWD (binary-planting
  hardening); POSIX uses `shutil.which`. All probes injectable.
  `HardwareInfo.best_gpu()`, `format_bytes()`.
- `advisor.py` — documented heuristic, not a llama.cpp simulation:
  `estimate_memory` = weights + mmproj + `kv_bytes_per_token`·ctx +
  (`OVERHEAD_BASE_BYTES` + 5% weights); `assess` compares against
  `VRAM_USABLE_SHARE`·free-VRAM and `RAM_USABLE_SHARE`·total-RAM →
  `RunVerdict` `GPU_FULL / GPU_PARTIAL / CPU_ONLY / NOT_RUNNABLE / UNKNOWN`
  (+ shortfall bytes for the UI) and a five-level `RunGrade`
  (`PERFECT/SMOOTH/OK/BARELY/NO/UNKNOWN` — `HEADROOM_FACTOR` 1.3 splits
  the comfortable vs tight cases) rendered as plain Chinese in the GUI.
- `settings.py` — frozen `LocalSettings` (models_dir = primary/download
  dir, `extra_dirs` = additional reuse search dirs, server_path, port,
  context_length, gpu_layers `-1`=auto, threads `0`=auto, **parallel**, last
  family/quant selection); `load_local_settings` clamps every numeric field
  into its range, sanitizes `extra_dirs`, and falls back to defaults on
  corrupt files; `save_local_settings` is atomic. The GUI persists
  `app_data_dir()/local_llm.json`; the default primary dir lives INSIDE
  the app (`models/`, git-ignored).
- `download.py` — stdlib resumable downloader: streams to `<dest>.part`,
  resumes via `Range` (server ignoring the range → clean restart; HTTP 416 →
  drop part and restart), cancel via `threading.Event` →
  `DownloadCancelledError` (part kept), an in-stream overrun cap (a server
  sending more than promised is aborted immediately, part discarded),
  exact-size verification (short → keep part for resume) and SHA256
  verification against the catalog digest BEFORE the atomic `os.replace`
  (mismatch → discard). Scheme allow-list: https only.
- `server.py` — `ServerSpec` + `build_server_args` (validated argument
  vector, never a shell string) and `LocalServerManager` (injectable
  popen / health / sleep / clock): spawns llama-server, polls `/health`
  until ready (early process exit and ready-timeout raise
  `LocalServerError`, the process is stopped), `stop()` terminate → kill,
  `base_url(port)` = `http://127.0.0.1:{port}/v1` so the existing
  OpenAI-compatible client stack drives local models unchanged.

Tests mirror the package under `tests/local/` (fake openers, fake
processes, injected clocks — no network, no real waits).

---

## v1.7 addendum — 推标 caption batches + pinned llama.cpp runtime (additive)

- `nlapt/local/runtime.py` (new, stdlib + `nlapt.local.download`): pinned
  llama.cpp release snapshot (`LLAMA_CPP_TAG` = b10088, byte sizes + SHA256
  from the GitHub release API `digest` field, captured on
  `RUNTIME_SNAPSHOT_DATE`; update table and date together). Windows/Linux
  x64 pin the **Vulkan** builds (one archive drives NVIDIA/AMD/Intel and
  falls back to llama.cpp's CPU backend), win-arm64/macos-arm64 pin the CPU
  builds. `runtime_platform_key()` / `current_asset()` gate per-platform
  support (unsupported → manual `server_path` flow only);
  `runtime_dir(base)` is versioned by tag; `find_server_exe(dir)` searches
  recursively (flat zip vs `build/bin` tar layouts), shallowest match wins;
  `ensure_runtime(base_dir, progress=, cancel=, opener=)` fast-paths an
  extracted exe, else downloads via `download_file` (resumable +
  digest-verified), extracts (`zipfile` / `tarfile` with `filter="data"`),
  deletes the archive, chmods the exe on POSIX. Raises `LocalServerError`
  with actionable Chinese messages.
- `LocalServerManager` tracks the launching `ServerSpec`
  (`current_spec`, None when stopped) and gains
  `ensure(spec, ready_timeout=...)`: running with an **equal** spec →
  return the base URL immediately (no model reload); anything else →
  `start(spec)` (which still stops a previous server first). This is the
  keep-alive primitive behind batch inference: load once, serve every
  item, unload only after the batch.
- `NLaptApp.run_caption_batch(keys, caption_fn, *, description, engine="",
  concurrency=1, controller=None, on_progress=None) -> BatchReport`:
  destructive vision batch (推标) that WRITES captions. Follows the batch
  safety model: pre-execution snapshot, per-file save, one `kind="batch"`
  oplog record (whole-batch rollback), resume checkpoint id from
  `("caption:"+engine, keys, engine)`. `caption_fn(key, image_path)` runs
  on the engine's thread pool with the requested concurrency; store /
  undo / index / disk mutations are serialized under the app lock. An
  empty/whitespace result → failed item (`EMPTY_CAPTION_ERROR`), caption
  untouched; unchanged text → ok with `NO_CHANGE_DETAIL`.
- `nlapt/local/gguf.py` (new, stdlib): `read_block_count(path)` — minimal
  GGUF v2/v3 metadata scan for the `<arch>.block_count` key (skips
  fixed/string/array values with defensive caps); never raises, `None` on
  anything unexpected.
- `advisor.auto_gpu_layers(family, quant, hardware, *, context_length,
  block_count)` — the `-ngl` value that FITS the scaled free VRAM: full
  estimate fits → `GPU_LAYERS_ALL`; no GPU → 0; otherwise weights+kv are
  spread over `block_count` layers, mmproj+overhead reserved first, and
  the remaining budget buys whole layers (clamped). Unknown block_count
  while not fitting → 0 (a slow CPU run beats a crashed server). Fixes
  the field OOM: `-ngl 999` on an 8 GB GPU with an 8.3 GB estimate made
  llama-server exit(1) during load. The early-exit `LocalServerError`
  message now also suggests lowering GPU 层数 / 上下文.
- `server.build_server_args` (behavior, field-debugged on real hardware):
  emits `-c` = `total_context(spec)` = `context_length × parallel`
  (llama-server SPLITS `-c` across its slots — the raw setting silently
  gave each request only 1/parallel of the configured 上下文长度) and
  appends `--reasoning off` (thinking models — Gemma 4 — burned the whole
  token budget in their reasoning channel and returned an EMPTY `content`
  for caption requests; verified against a live llama-server b10088).
  `ServerSpec.context_length` is therefore PER-SLOT; memory estimates for
  launches must use the total (`auto_gpu_layers` callers pass ctx×parallel).

## v1.8 addendum — Florence-2 PromptGen ONNX engine (additive)

- Catalog families now carry an inference engine: `ModelFamily.engine`
  (`ENGINE_LLAMA` "llama" — default, single GGUF + optional mmproj served
  by llama-server — or `ENGINE_FLORENCE` "florence") and
  `ModelFamily.extra_files: tuple[QuantFile, ...]` (additional required
  files beside the quant; empty for llama families). Extra files download,
  verify (exact bytes + SHA256) and resolve exactly like quants
  (basename inside the per-family dir; `quant_path` works for them).
  New series `florence2-promptgen` with family `florence2-promptgen-v2`
  (repo `laub/Florence-2-large-PromptGen-v2.0-onnx` — the only FUNCTIONAL
  community ONNX export; despite the repo name the weights are the
  base-size 0.23B architecture, verified from the graph dims. The
  sibling `...-base-...-onnx` repo is a broken text-only export whose
  encoder cannot take image features).
- `nlapt/local/florence.py` (new): in-process Florence-2 pipeline over
  onnxruntime (optional dep, lazy import; `numpy`/`Pillow` likewise).
  Public surface: task tokens (`TASK_GENERATE_TAGS` … `TASK_MIXED_CAPTION_PLUS`),
  `FLORENCE_TASK_LABELS` (token → Chinese label, display order),
  `FLORENCE_TASK_TOKENS`, `DEFAULT_FLORENCE_TASK` (= `<GENERATE_TAGS>`),
  `validate_task`, `prompt_for_task` (standard Florence tasks map to their
  fixed English questions verbatim from the official processor; PromptGen
  additions pass through literally — how the model was trained),
  `FlorenceTokenizer` (byte-level BPE over `tokenizer.json`; ASCII
  pre-tokenizer — prompts are catalog-pinned ASCII), and `FlorenceEngine`:
  `FlorenceEngine(files: Mapping[basename, Path], *, session_factory=None)`,
  `load()` (lazy, locked, idempotent), `unload()`, `is_loaded()`,
  `caption(image_path, task) -> str`. Pipeline: CLIP-style 768×768
  preprocess → vision_encoder → embed_tokens(prompt ids `<s>…</s>`) →
  concat → encoder → greedy merged-decoder loop with KV cache, forced BOS
  and 3-gram repetition blocking (`forced_bos_token_id=0`,
  `no_repeat_ngram_size=3` from the model config; reference uses
  num_beams=3 — greedy is the documented simplification). KV layout
  (layers/heads/head-dim) is read from the decoder session's declared
  input shapes, so the engine serves any Florence-2 export size; on cache
  steps the export returns zero-element cross-attention dummies
  (e.g. shape (0,1,1,1)) which are ignored — the first real tensors are
  kept. Sessions are thread-safe for concurrent `caption` calls. Typed
  errors: `LocalInferenceError` (missing deps/files, actionable Chinese),
  `ValidationError` (unknown task / incomplete file mapping). All required
  file basenames: `REQUIRED_FILES` (4 ONNX parts + tokenizer.json).
- `LocalSettings.florence_task: str = DEFAULT_FLORENCE_TASK` — the
  persisted 指令; unknown stored values fall back to the default on load.
- `advisor.estimate_memory`: weights now include `extra_files` sizes; for
  `ENGINE_FLORENCE` the caller's context_length is replaced by the fixed
  `FLORENCE_CONTEXT_TOKENS` (= 1664: 577 image tokens + ≤1024 generated —
  the llama-server 上下文长度 setting does not apply to this engine).
- Optional dependency group `local = ["onnxruntime", "numpy"]`
  (pyproject); the PyInstaller spec adds both to hiddenimports.

## v1.9 addendum — interception guards, web-translate timeout, prompt presets (additive)

- `nlapt.llm.cleaning`: new `ensure_not_refusal(text) -> str` plus the
  `REFUSAL_PREFIXES` / `REFUSAL_MARKERS` / `MSG_REFUSAL` constants. Raises
  `LLMOutputError` (Chinese, includes an 80-char preview) when the text
  reads like a safety refusal ("I'm sorry…", "抱歉…", "content policy" …).
  Matching is lowercase; prefixes anchor at the start, markers match
  anywhere. Applied by VISION captioners only — translations may
  legitimately start with such words, so `clean_llm_output` itself is
  unchanged.
- `nlapt.llm.openai_client`: `_extract_text` now checks
  `choices[0].finish_reason == "content_filter"` BEFORE parsing the
  message content and raises `LLMRequestError(MSG_CONTENT_FILTERED)`
  (Chinese, retryable — filter trips are often probabilistic). A filtered
  choice with a null/replaced message no longer surfaces as a generic
  parse error.
- `nlapt.llm.web_translate`: new `WEB_TRANSLATE_TIMEOUT_SECONDS = 15.0`
  is the default `timeout` of all three providers (was the LLM default
  60 s). Healthy endpoints answer in seconds; a short per-attempt timeout
  plus the caller's retry beats one long hang. The constructor parameter
  is unchanged for callers that need a different budget.
- `nlapt.local.presets` (new module): hardcoded snapshot (2026-08-01) of
  the official instructions the caption specialists were trained on.
  `PromptPreset(preset_id, label, system, user_prompt)` (frozen),
  `PRESET_CUSTOM = "custom"` sentinel, `TORIIGATE_PRESETS` (10 formats,
  verbatim from Minthy/ToriiGate-0.5 `scripts/prompts.py`, assembled like
  the official `make_user_query`: `# Captioning format:` + body + the
  recognize-characters directive), `JOYCAPTION_PRESETS` (12 caption
  types, verbatim index-0 templates of the official Space's
  `CAPTION_TYPE_MAP`, plus its recommended system prompt),
  `PRESETS_BY_FAMILY` keyed by catalog `family_id`, `presets_for(id)`,
  and `resolve_preset(family_id, preset_id) -> PromptPreset | None`:
  `PRESET_CUSTOM` or a family without presets → None (free-form prompts);
  empty/unknown id → the family's FIRST preset (default). Gemma families
  intentionally have no entry; Florence keeps its own 指令模式.
- `LocalSettings.prompt_preset: str = ""` — persisted preset id
  (round-tripped as-is; semantics live in `resolve_preset`).

## v1.10 addendum — empty-completion guard, connection-test budget (additive)

- `nlapt.llm.openai_client`: `_extract_text` raises
  `LLMRequestError(MSG_EMPTY_COMPLETION)` (Chinese, includes the choice's
  `finish_reason`) when `choices[0].message.content` is an empty/whitespace
  string. HTTP-200-with-empty-content is what proxies under high
  concurrency and thinking models with an exhausted output budget return;
  as an `LLMRequestError` it is retryable under the spec-8 backoff instead
  of surfacing as an empty caption / `LLMOutputError` (which never
  retried). `nlapt.llm.ollama_client` gets the same guard (English
  message). Anthropic already raised on empty content blocks.
- `nlapt.llm.base`: `CONNECTION_TEST_MAX_TOKENS` raised 8 → 1024. Thinking
  models spend completion tokens on hidden reasoning before the visible
  reply; with an 8-token cap the test reply always came back empty, so
  测试连接 always failed on such models. The prompt is unchanged.

## v1.11 addendum — Florence PEFT LoRA merging (additive)

- `nlapt/local/lora.py` (new): loads PEFT LoRA adapters and merges them
  into the Florence ONNX weights at session-creation time — no `onnx`
  package, no torch, nothing written back to disk. Public surface:
  `ADAPTER_CONFIG_NAME`, `LoraModule(down, up, scale)` /
  `LoraAdapter(path, modules)` / `MergeResult(initializers, matched)`
  (frozen), `load_adapter(path) -> LoraAdapter` (parses the
  `.safetensors` file — stdlib header walk, F32/F16/BF16 widened to
  float32 — plus its REQUIRED sibling `adapter_config.json`; PEFT scale
  `alpha / r` per module rank, rsLoRA `alpha / sqrt(r)`, DoRA rejected),
  and `merge_into_onnx(path, adapter) -> MergeResult` (minimal protobuf
  walk over the frozen ONNX field numbers collects every MatMul node and
  initializer across nested subgraphs — the merged decoder hides both
  KV-cache branches inside an `If`, sharing one weight set; node scope
  paths, with repeated ModuleList scopes like `blocks.0/blocks.0.0`
  collapsed, suffix-match the PEFT module names; merged weight is
  `M + scale * down.T @ up.T`, dtype preserved). Typed failures
  (`LocalInferenceError`, actionable Chinese): missing file / missing
  config / corrupt file / DoRA / dimension mismatch (`MSG_LORA_MISMATCH`
  names both shapes — a large-architecture LoRA on the base export) /
  `MSG_LORA_NO_MATCH` when nothing matches at all.
- `FlorenceEngine(files, *, session_factory=None, lora_path=None)`:
  with a `lora_path`, `load()` merges every file's overrides and calls
  the session factory as `factory(path, initializers)`; without one the
  call stays unary, so existing factories keep working. `SessionFactory`
  is now a Protocol with the optional `initializers` parameter. The
  default factory feeds overrides through
  `SessionOptions.add_initializer` and ties the `OrtValue` wrappers to
  the session object (`nlapt_lora_keepalive`) — ORT keeps raw pointers
  into them for the session's lifetime. A LoRA matching nothing anywhere
  raises; partial matches log a warning; any load failure resets the
  half-built session set.
- `LocalSettings.florence_lora: str = ""` (selected `.safetensors` path)
  and `LocalSettings.florence_loras: tuple[str, ...] = ()` (registered
  files for the dropdown); junk values degrade to defaults on load.

## v1.12 addendum — official Florence-2 families + curated LoRA catalog (additive)

- Catalog: the Florence series (id `florence2-promptgen` unchanged, display
  name now "Florence-2") gains the four official Microsoft exports by
  onnx-community, all `ENGINE_FLORENCE`, fp32 file sets pinned from the HF
  API (sizes + LFS sha256; the identical BART `tokenizer.json` is shared
  via `_florence_tokenizer()`): `florence2-base-ft`, `florence2-large-ft`,
  `florence2-base`, `florence2-large` (large ones: 0.77B,
  `kv_bytes_per_token=98_304`). `ModelFamily.florence_tasks:
  tuple[str, ...] = ()` — the 指令 tokens the family was trained on
  (empty for llama families; enforced by tests). PromptGen keeps its 7
  tokens; officials get the three standard caption tasks; large officials
  additionally list `TASK_BAI_JSON` (= `<BAI_JSON>`, new in
  `nlapt.local.florence`, passed through literally). The FIRST entry is
  the fallback the captioner clamps to when the persisted task is not in
  the set.
- Curated LoRAs: `LoraEntry(lora_id, name, base_url, page_url, files,
  compatible_family_ids, task, notes)` (frozen) + `ALL_LORAS` with
  `bai-json-large` (silverlong's BAI_JSON PromptGen LoRA, two files
  pinned with exact bytes + sha256, served from ModelScope — hence a full
  `base_url` instead of a HF repo id). Helpers: `all_loras()`,
  `find_lora(id)` (ValidationError on unknown), `loras_for_family(id)`,
  `lora_dir(models_dir, entry)` (= `models_dir/loras/<lora_id>`, SAFE_ID
  checked), `lora_file_path(...)`, `lora_adapter_path(...)` (the
  .safetensors member), `lora_download_url(entry, file)`
  (base_url + encoded filename). `LORA_DIR_NAME = "loras"`.
- `nlapt.local.lora.merge_into_onnx` gained keyword `module_prefix: str
  = ""`: candidate adapter modules are pre-filtered to that PEFT-tree
  prefix. Needed because onnx-community's encoder export roots node
  names at the submodule (bare `/layers.0/...`), where a suffix alone
  matches both encoder and decoder layers; the engine passes
  `florence.LORA_MODULE_PREFIXES[file]` per file (decoder/embed →
  `language_model.`, encoder → `language_model.model.encoder.`, vision →
  `vision_tower.`). Verified against the real exports: the BAI_JSON
  adapter maps 241/241 modules on the large families.

## v1.13 addendum — Florence CUDA/cuDNN DLL preload (additive)

- `nlapt/local/florence.py::_preload_ort_dlls(ort)` (idempotent): before the
  default session factory creates an `InferenceSession`, call
  `onnxruntime.preload_dlls(cuda=True, cudnn=True, directory="")` when the
  attribute exists. Prefer `nvidia-*` pip site-packages
  (`pip install onnxruntime-gpu[cuda,cudnn]`) over PATH/PyTorch layouts so a
  missing `cudnn64_9.dll` no longer surfaces as a mid-run
  `NOT_IMPLEMENTED` Conv failure. Failures are swallowed — CPU EP remains
  usable without GPU runtime DLLs.

## v1.14 addendum — dataset zip export (additive)

- `nlapt/storage/export.py::export_dataset_zip(root, dest, files) -> int`:
  stream each scanned `ImageFile` (image + existing same-stem txt) into a
  zip via a temp file in `dest.parent`, then `os.replace`. Datasets can be
  large — do not buffer the archive in memory the way snapshot zips do.
  `.backups` / `.nlapt` are excluded because they are never in the scanned
  file list. Returns the number of archive members written. Empty `files`
  raises `ValidationError`; missing parent directory raises `StorageError`.
- `NLaptApp.export_dataset(dest) -> int`: thin wrapper that requires an
  open dataset and forwards `self._files` to `export_dataset_zip`.
- GUI redo is **not** `NLaptApp.redo`: the editor undo model is the
  `FileHistory` cursor (`step_older` / `step_newer`).

## v1.15 addendum — dataset state lives under the per-user data dir

Snapshots, crash-recovery session, and batch checkpoints no longer write
into the dataset root. `NLaptApp.open_dataset` places them at
`app_state_dir()/datasets/<sha1(normcase(resolve(root)))>/`:

- `backups/` — `SnapshotManager(..., backup_dir=...)`
- `session.json` / `checkpoints.json` — `SessionStore` / `CheckpointStore`
  `(..., state_dir=...)`

`SnapshotManager` / `SessionStore` / `CheckpointStore` keep their default
in-dataset paths (`.backups/`, `.nlapt/`) when the new kwargs are omitted,
so existing unit tests stay hermetic. On open, `migrate_legacy_dataset_state`
best-effort moves leftover `<root>/.backups` and `<root>/.nlapt` into the
new location and removes the old directories. Scanner/export still prune
those names.

`nlapt.storage.paths.app_state_dir` is the single resolver
(`NLAPT_DATA_DIR` / `%APPDATA%/NLapt` / `~/.config/nlapt`).
`nlapt_gui.resources.app_data_dir` delegates to it.

## v1.16 addendum — console logging is opt-in

`configure_logging` gained `*, console: bool = False`. The rotating file
handler is unchanged. A `StreamHandler` is attached only when
`console=True` (GUI 调试模式). Default is quiet so launchers are not
flooded with worker stacks.

## v1.17 addendum — custom OpenAI-compatible translate provider

`nlapt.llm.web_translate` adds `PROVIDER_CUSTOM` / `CustomOpenAIProvider`:
POST `{base_url}/chat/completions` (or a full completions URL) with a
caption-translation prompt. Base URL and model are required; API key is
optional. GUI credentials for this provider persist under
`user_documents_app_dir()/translate_api.json`
(`Documents/NLapt`, overridable via `NLAPT_DOCUMENTS_DIR`).

## v1.18 addendum — DeepLX free translate provider

`PROVIDER_DEEPLX` / `DeepLXProvider` posts JSON to `{base}/translate`
(`source_lang=auto`, `target_lang` ZH/EN/JA). A full URL already ending in
`/translate` is kept. Optional token uses `Bearer`. Response `code` other
than 200, or a missing `data` string, raises `LLMRequestError`.

## v1.19 addendum — DeepLX throttle + Hy-MT2 local MT

`http_post_json` accepts optional `display_url` used in logs and exception
text (DeepLX redacts secret path/query segments). `DeepLXProvider` shares a
process-wide `MinIntervalLimiter` (1 s) and retries HTTP/body `429` up to
3 times (1 s / 2 s / 4 s). DeepLX URL + token persist in
`Documents/NLapt/translate_api.json` (one-time migrate from AppData).

`nlapt.local.mt_catalog` snapshots three official Hy-MT2 GGUF tiers
(`fast` / `balanced` / `quality`) with exact LFS sizes and SHA256.
`nlapt_gui.mt_bridge.LocalMTProvider` serves them on `settings.port + 1`
via a second `LocalServerManager` + idle stopper, using the official
Hy-MT2 user prompt and `temperature=0.7` / `top_p=0.6`. Downloads reuse
`local_bridge.launch_download_jobs` into `models_dir/mt/<tier>/`.

## v1.20 addendum — DeepLX 418 busy

Community DeepLX hosts often answer HTTP 418 (`I'm a teapot`) as a
rate-limit. `DeepLXProvider` retries 418 / 429 / 503 with the same
backoff as 429, then raises a Chinese busy message (no teapot / JSON
body) so the 分层推标 cards and 译文对照 stay readable.

## v1.21 addendum — translate fallback chain

`nlapt.llm.fallback.run_fallback_chain` tries `(name, thunk)` pairs in
order. `LLMConfigError` skips an unusable provider; other `LLMError`
values continue to the next name. The last error is re-raised. Empty
input raises `LLMRequestError("所有翻译备选均失败")`.

## v1.22 addendum — Hy-MT2 chat inference

`LocalMTProvider` uses the single-user-message chat protocol and inference
parameters documented by [Tencent](https://huggingface.co/tencent/Hy-MT2-1.8B).
The llama.cpp-specific request and response contract is maintained in
`docs/UI_ARCHITECTURE.md` v1.30, alongside the GUI-owned MT server lifecycle.

## v1.23 addendum — public profile parsers + Known Folder Documents

`nlapt.core.config` exposes the profile parsers used by the GUI's unified
`api.json` (see `docs/UI_ARCHITECTURE.md` v1.38). `load_config` /
`save_config` still accept `profiles` / `active_profile` so migration and
core tests can read leftover AppData files.

```python
def profile_from_dict(data: Any, index: int = 0) -> LLMProfile
def profiles_from_list(raw: Any) -> tuple[LLMProfile, ...]
```

`user_documents_dir()` resolution order: `NLAPT_DOCUMENTS_DIR` (tests and
portable installs), then Windows `SHGetKnownFolderPath(FOLDERID_Documents)`
so OneDrive / redirected folders resolve, then `Path.home()/Documents`.
`windows_known_documents_dir()` returns `None` off Windows or when the
shell call fails.

## v1.24 addendum — image tools (`nlapt.images`)

Pixel-level helpers for flattening transparent backgrounds and grouping
near-duplicate images. Pillow is optional: `load_pil()` raises
`ImageProcessingError` with an install hint when it is missing.

```python
def load_pil() -> ModuleType
def has_transparency(path: Path) -> bool
def flatten_alpha(path: Path, color: tuple[int, int, int]) -> bool
def pick_color(spec: FlattenSpec, key: str, rng: object | None = None) -> tuple[int, int, int]
def dhash(path: Path, size: int = 8) -> int
def hamming(left: int, right: int) -> int
def sha256_file(path: Path) -> str
def fingerprint(key: str, path: Path, *, size: int = 8) -> ImageFingerprint
def group_similar(fingerprints, max_distance: int) -> tuple[DuplicateGroup, ...]
def suggest_keep(group: DuplicateGroup) -> ImageFingerprint

class FlattenSpec:  # mode: "fixed"|"random"; color; palette; seed
class ImageBackupManager:
    def create(self, operation: str, paths: tuple[Path, ...]) -> ImageBackupInfo
    def quarantine(self, paths: tuple[Path, ...], *, operation: str = "duplicates") -> ImageBackupInfo
    def list_backups(self) -> tuple[ImageBackupInfo, ...]
    def restore(self, info: ImageBackupInfo) -> tuple[str, ...]
```

Flatten writes the original container back (PNG unchanged, WEBP `lossless=True`)
via `atomic_write_bytes`. Image copies live under
`<dataset>/.backups/images/<stamp>_<op>/`; quarantined near-duplicates (image +
sibling txt) live under `<dataset>/.backups/duplicates/<stamp>/`. Both kinds
keep the 5 newest folders. Dataset scan already skips `.` directories.

## v1.25 addendum — model pool: several APIs, per-model switches, two targets

An `LLMProfile` is one API (endpoint + key) that knows a catalog of models;
the switched-on models of every profile form one pool. Two roles pick from
the pool — the 文本 target (翻译 / 改写) and the 视觉 target (推标 / CHA) —
and may sit on different APIs.

```python
# nlapt/core/config.py
@dataclass(frozen=True)
class LLMProfile:
    ...                                   # name / api_type / base_url / api_key / temperature / max_tokens / system_prompt
    text_model: str = ""                  # id a RESOLVED profile is bound to (legacy default on stored ones)
    vision_model: str = ""
    models: tuple[str, ...] = ()          # catalog (fetched via list_models or typed)
    enabled_models: tuple[str, ...] = ()  # subset switched on for the pool

@dataclass(frozen=True)
class ModelRef:
    profile: str = ""
    model: str = ""
    def is_set(self) -> bool

ROLE_TEXT = "text"; ROLE_VISION = "vision"

@dataclass(frozen=True)
class AppConfig:
    profiles: tuple[LLMProfile, ...] = ()
    active_profile: str = ""              # legacy single selection; the pool UI writes ""
    text_target: ModelRef = ModelRef()
    vision_target: ModelRef = ModelRef()
    ...

def model_ref_from_dict(data: Any, context: str) -> ModelRef      # None -> unset; non-dict -> ValidationError
def find_profile(config, name) -> LLMProfile | None
def enabled_model_refs(config) -> tuple[ModelRef, ...]           # profile order, then enabled_models order
def resolve_model_ref(config, ref) -> LLMProfile | None          # None when unset / profile gone / model switched off
def resolve_text_profile(config) -> LLMProfile | None            # target's profile with text_model = ref.model
def resolve_vision_profile(config) -> LLMProfile | None          # ... with vision_model = ref.model
def upgrade_legacy_targets(config) -> AppConfig                  # pure, idempotent
```

Rules:

1. A stale target (profile deleted or model switched off) resolves to
   `None` — callers treat it as unconfigured; nothing falls back to another
   model silently.
2. When a target is **unset**, the resolvers fall back to the legacy
   `active_profile` + its `text_model` / `vision_model`, so a pre-pool
   `AppConfig` (tests, `reload_config`) keeps working unchanged.
3. `upgrade_legacy_targets`: a profile with neither `models` nor
   `enabled_models` enables its `text_model` / `vision_model` (and lists them
   in `models`); unset targets are pointed at the `active_profile`'s models.
   A profile that has a catalog but nothing enabled stays off (deliberate).
   The GUI applies this on every `api.json` load; core `load_config` does
   not (round-trip fidelity for tests / migration).
4. `NLaptApp.make_translator` uses `resolve_text_profile`;
   `make_rewrite_service` builds `text_client` from the text profile and
   `vision_client` from the vision profile (possibly another API), passing
   `replace(text_profile, vision_model=vision_profile.vision_model)` as the
   service's profile so `_select_client_and_model` is unchanged.
   `LLMConfigError` when no text model resolves.
5. `masked_config_dict` still masks every `api_key`; `text_target` /
   `vision_target` serialise as `{"profile", "model"}` objects.

## v1.26 addendum — binding an arbitrary pool ref

```python
# nlapt/core/config.py
def bind_model_ref(config: AppConfig, ref: ModelRef, role: str) -> LLMProfile | None
```

The pool profile `ref` points at, with `text_model` (ROLE_TEXT) or
`vision_model` (ROLE_VISION) replaced by `ref.model`; `None` for an unset or
stale ref (same rule as `resolve_model_ref`), `ValidationError` for an
unknown role. `resolve_text_profile` / `resolve_vision_profile` are now thin
wrappers over it (plus the legacy `active_profile` fallback). GUI callers
that need a model other than the two targets — CHA标注 per-scheme picks and
多对比推标 — go through `AppController.profile_for_ref(ref)` which binds
ROLE_VISION.

