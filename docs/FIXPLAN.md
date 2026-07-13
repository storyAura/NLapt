# NLapt Review Fix Plan (v1)

Input: `docs/REVIEW_FINDINGS.json` — 21 adversarially confirmed findings (indices `[n]`
below refer to array positions in that file). Deduplicated into 13 issues, split into 3
groups with **disjoint file ownership** so fixes can be applied in parallel.

Binding decisions below override the findings' own `suggested_fix` where they differ.
The public contract in `docs/ARCHITECTURE.md` still applies; all changes here are either
internal or **additive** (new optional params, new methods, new constants) — never break
an existing signature. Every fix gets a regression test.

Common rules for fixers: run ONLY your own test paths while iterating (other fixers are
editing the repo concurrently — a full-suite run may see transient failures that are not
yours). The orchestrator runs the full suite after all groups land.

---

## Group 1 — storage & core

Files owned: `nlapt/core/config.py`, `nlapt/storage/snapshots.py`, `nlapt/storage/text_io.py`,
new `tests/regression/test_fixes_storage_core.py`, plus updates to
`tests/core/test_core_config.py`, `tests/storage/test_storage_snapshots.py`,
`tests/storage/test_storage_text_io.py` where behavior legitimately changed.

### 1.1 mask_secret boundary leak — findings [0] (low)
Reveal partial characters ONLY when `len(value) >= 12`; otherwise return `"***"`.
Keep `"" -> ""`. Update the boundary tests (lengths 8–11 must be fully masked; at least
half of any secret must remain hidden).

### 1.2 restore() can destroy the snapshot being restored — findings [1], [13] (high)
In `SnapshotManager.restore()`: read and zip-slip-validate ALL required entry bytes from
the target zip into memory BEFORE calling `create(PRE_RESTORE_OPERATION)`. As defense in
depth, add an optional `exclude: Path | None` param to the internal retention-pruning
helper and exclude the restore-target zip during the pre-restore `create()` call.
Regression test: `retention=1` (or capacity-filled) manager, restore the oldest snapshot —
must succeed and the target zip must still exist afterwards.

### 1.3 BOM + non-UTF-8 body falls through to latin-1 mojibake — finding [6] (medium)
In `read_text_detect`: when the raw bytes start with the UTF-8 BOM but `utf-8-sig` decode
fails, strip the 3 BOM bytes and run the remaining candidate encodings over the body only;
result flags `needs_conversion=True` and reports the body encoding. Regression test:
BOM + GBK-encoded Chinese body decodes as gb18030, not latin-1.

---

## Group 2 — llm, ops, batch, workflow, captions-store

Files owned: `nlapt/llm/rewrite.py`, `nlapt/llm/translate.py`, `nlapt/llm/cleaning.py`,
`nlapt/ops/prefix_suffix.py`, `nlapt/batch/engine.py`, `nlapt/batch/checkpoint.py`,
`nlapt/workflow/suggestions.py`, `nlapt/captions/store.py`,
new `tests/regression/test_fixes_llm_ops_batch.py`, plus updates to the existing tests of
those modules where behavior legitimately changed.

### 2.1 RequestControl is dead config — findings [7], [10], [14] (high)
Wire `nlapt.core.config.RequestControl` through the real request paths (ADDITIVE param):
- `RewriteService.__init__` and `Translator.__init__` gain
  `request: RequestControl | None = None` (keyword-only, default None = old behavior).
- When set: build every `LLMRequest` with `timeout=request.timeout`; wrap each
  `client.complete(...)` in `with_retry(RetryPolicy(max_retries=request.max_retries), ...)`;
  create ONE shared `MinIntervalLimiter(request.min_interval)` per service instance and
  call `limiter.wait()` before each request (this makes it effective across batch workers
  because the app shares one service instance).
- Keep sleep/clock injectable for tests (`retry_sleep`/`limiter` optional params or
  equivalent) — tests must not really sleep.
Regression tests: configured timeout reaches the client (assert on MockLLMClient captured
request); `MockLLMClient(fail_times=1)` succeeds after one retry with backoff delays
asserted via injected sleep; limiter consulted once per request.

### 2.2 Quote stripping corrupts multi-quoted captions — finding [3] (medium)
`_strip_matching_quotes`: strip the outer pair only when it wraps the WHOLE string —
i.e. the closing quote character must not occur anywhere in the inner text (scan before
stripping; keep iterating for nested distinct pairs as before). Tests:
`"blue eyes" and "red hair"` unchanged; `"whole caption"` stripped; `“中文引号”` stripped;
`"a" or "b"` unchanged.

### 2.3 accept_for_edit can land AI text as CONFIRMED — finding [5] (medium)
Add an ADDITIVE method to `CaptionStore`:
`set_text_forced_state(key, text, state: CaptionState) -> CaptionRecord` (dirty=True,
publishes EVT_CAPTION_CHANGED and EVT_STATE_CHANGED when state changed; validates
non-empty text for non-UNLABELED states). `accept_for_edit` uses it to force
`CaptionState.DRAFT` regardless of `revert_confirmed_on_edit`. `accept_suggestion` keeps
the normal `set_text` transition (contract-compliant). Regression test: store with
`revert_confirmed_on_edit=False`, CONFIRMED record + pending → `accept_for_edit` yields
DRAFT.

### 2.4 skip_if_present matches substrings — findings [9], [15] (medium)
Skip rule becomes: skip iff `text == prefix` or `text.startswith(prefix + joiner)`
(suffix: `text == suffix` or `text.endswith(joiner + suffix)`). When `joiner == ""`, keep
the raw startswith/endswith semantics (explicitly chosen by the user). Tests: trigger
`cat` IS added to `caterpillar, ...`; `minahamu` NOT re-added to `minahamu, ...`; same
pattern for suffix.

### 2.5 Paused engine stalls collection of finished futures — finding [16] (low)
Restructure the engine loop so pause blocks only NEW dispatches: while paused, keep
draining completed in-flight futures (progress callbacks, events, checkpoint marks fire).
Use bounded waits (e.g. `concurrent.futures.wait(..., timeout=...)`) — no busy spin.
Regression test: gate-controlled workers; pause; release gates; assert progress observed
while still paused; then resume and complete.

### 2.6 Corrupt checkpoints.json bricks open_dataset — finding [2] (medium)
Self-heal inside `CheckpointStore`: on corrupt/unreadable JSON, log a warning, rename the
bad file aside to `checkpoints.json.corrupt` (best-effort), and start with an empty store
instead of raising. Regression test mirrors the corrupt-session test.

---

## Group 3 — app facade

Files owned: `nlapt/app.py`, `nlapt/core/events.py` (ADDITIVE constants only),
new `tests/regression/test_fixes_app.py`, plus updates to `tests/app/*` where behavior
legitimately changed. NOTE: Group 2 concurrently adds
`CaptionStore.set_text_forced_state` and RequestControl wiring — do not depend on or
touch those files; your fixes below need none of them.

### 3.1 Duplicate-stem images alias one txt — findings [4], [11] (high)
Kohya pairing in the scanner stays as-is (two images with the same stem legitimately
reference the same txt). The FACADE must prevent aliased editable records:
in `open_dataset`, group scanned images by resolved `txt_path`; the FIRST key in natural
order becomes the canonical editable record; the remaining keys are EXCLUDED from the
caption store/index, collected into a `txt_conflicts()` mapping
(`canonical_key -> tuple(excluded_keys)`), logged as a warning, and announced via a new
event constant `EVT_TXT_CONFLICT` in `nlapt/core/events.py` (additive). `_txt_to_key`
therefore stays 1:1 with canonical keys, which also fixes the stale-rollback-reload alias
bug. Regression tests: `cat.jpg`+`cat.png`+`cat.txt` → one editable record, conflict
reported, batch apply + rollback reload behave consistently.

### 3.2 Suggestion durability lags checkpoint durability — findings [8], [12], [17], [20] (high)
In `run_rewrite_batch`'s worker: after `set_suggestion(...)` succeeds, persist the session
(`save_session()`) BEFORE returning success (the engine marks the checkpoint only after
the worker returns ok — verify and keep that ordering). Guard session writes with a lock
(concurrent workers) — atomic write already exists. Also save the session when the batch
finishes with ANY status (completed/cancelled). Regression test: run a rewrite batch with
MockLLMClient, do NOT call close(), build a second NLaptApp on the same root → every
completed key has its pending suggestion restored; a resumed batch (same checkpoint_id)
skips exactly those keys.

### 3.3 needs_conversion discarded — finding [18] (medium)
During `open_dataset`, collect `TextReadResult.needs_conversion` files into an internal
mapping; expose `keys_needing_conversion() -> Mapping[str, str]` (key -> detected
encoding) and `convert_to_utf8(keys: Sequence[str]) -> tuple[str, ...]` which re-saves
each file via `write_caption` (UTF-8 no BOM) and clears it from the mapping. Publish a new
additive event constant `EVT_ENCODING_ISSUES` from `open_dataset` when the mapping is
non-empty. Regression test: GBK txt on disk → reported; convert_to_utf8 → file is UTF-8,
mapping cleared.

### 3.4 AI batches cannot be rolled back — finding [19] (medium)
`run_rewrite_batch` appends its oplog record with `kind="ai_batch"` (snapshot stays None —
suggestions never touch txt files). `rollback_operation` branches: for records with
`kind == "ai_batch"`, clear the pending suggestion on every affected key that still has
one, append a `kind="rollback"` record via `oplog.append`, save the session, and return a
`RestoreResult`-equivalent summary (define the return additively if needed — do NOT call
`oplog.rollback` for this kind; snapshot-backed kinds keep the existing path). Regression
test: rewrite batch → rollback_operation(op_id) → all pendings gone, rollback logged.

---

## Out of scope (explicitly rejected or deferred)

- 4 findings were refuted during adversarial verification — do not "fix" them.
- Anything requiring a breaking contract change is deferred; note it in the report instead.
