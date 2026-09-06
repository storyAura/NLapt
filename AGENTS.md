# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```powershell
pip install -e .[gui,images,llm,local,dev]  # PySide6 + Pillow + httpx + onnxruntime/numpy + pytest

python -m pytest -q                          # full suite (~30s, must stay green)
python -m pytest tests/gui -q                # GUI only (offscreen, no display needed)
python -m pytest tests/local/test_local_catalog.py::TestLookups::test_find_family -q
python -m pytest --cov=nlapt --cov=nlapt_gui -q      # coverage (project target ≥80%)
python -m ruff check nlapt nlapt_gui tests   # lint (default rules)

python -m nlapt_gui                          # run the app
Build-NLapt.bat                              # PyInstaller onedir build → dist/NLapt/NLapt.exe
```

## Binding contracts

`docs/ARCHITECTURE.md` (core) and `docs/UI_ARCHITECTURE.md` (GUI) are **binding interface
contracts**, maintained as versioned addendum sections (v1.1, v1.2, …; the doc tail has
the latest). Follow the public
signatures written there; when you change behavior, append a new versioned section rather
than rewriting history. `NLapt.md` is the original Chinese requirements spec.

## Architecture

Two strictly layered packages:

- **`nlapt/`** — UI-agnostic core, **stdlib-only** (`httpx`/`Pillow` are optional: import
  inside functions, raise a typed error with an actionable message when missing).
  `nlapt/app.py::NLaptApp` is the facade wiring everything: storage (atomic writes,
  dataset scanner, zip snapshots, crash-safe session), captions (in-memory `CaptionStore`
  publishing events), ops (text operations via `OperationRegistry`), llm (client registry
  `api_type → factory`: openai/anthropic/ollama + translate/rewrite/vision services),
  batch (thread-pool engine: snapshot → execute → oplog, pause/cancel/checkpoint),
  history (per-file undo + operation log rollback), local (model catalog / hardware
  advisor / downloader / llama-server manager / Florence-2 ONNX engine / PEFT LoRA
  merge).
- **`nlapt_gui/`** — PySide6 UI. **Widgets never touch the filesystem or core packages
  directly**: everything goes through `nlapt_gui/controller.py::AppController` (the only
  importer of `nlapt.app`) plus the async bridges (`translate_bridge`, `vision_bridge`,
  `local_bridge`). Sanctioned exceptions: `widgets/settings_dialog.py` and
  `widgets/local_tab.py` may read/write the core `config.json` directly, and
  `widgets/thumbnails.py` persists decoded thumbnails under `app_data_dir()/thumbs`
  (keyed by source mtime+size — its `clear()` drops the memory LRU only, on purpose).

Data-safety model the UI relies on: every batch operation snapshots all txt files to
`.backups/` **before** executing and is recorded in the operation log for whole-batch
rollback; txt files are written only on explicit save; unsaved drafts survive crashes via
the session file in `.nlapt/`.

Local inference (`nlapt/local`) has two engines, selected by `ModelFamily.engine`:

- **`ENGINE_LLAMA`** (GGUF): reuses the existing OpenAI-compatible client stack by
  launching llama.cpp's `llama-server` (`http://127.0.0.1:{port}/v1`); a pinned official
  llama.cpp release is auto-downloaded on first use (`nlapt/local/runtime.py`). The
  server manager is a process-wide singleton (`local_bridge.get_server_manager()`,
  stopped via `atexit`). A server started BY inference is stopped only after 30s idle
  (`local_bridge.get_idle_stopper()` — every local run brackets itself with
  `note_request`/`note_finished`, a new run cancels the pending stop); a server the
  USER started is never auto-stopped. `ServerSpec.context_length` is PER SLOT — the
  emitted `-c` is ctx × parallel, and `--reasoning off` is always passed (thinking
  models return empty captions otherwise).
- **`ENGINE_FLORENCE`** (Florence-2, `nlapt/local/florence.py`): llama.cpp cannot
  serve this architecture, so it runs in-process via onnxruntime (lazy optional
  import, extra `local`). No server, no runtime download; driven by task instructions
  (指令模式, persisted as `LocalSettings.florence_task`), not free-form prompts —
  `ModelFamily.florence_tasks` lists what each family was trained on (official
  Microsoft exports know only the three caption tasks; PromptGen its seven); the UI
  filters the dropdown AND the captioner clamps to the family's first task. Multi-file
  model: the quant is the decoder, siblings live in `ModelFamily.extra_files`. Engine
  singleton: `local_bridge.get_florence_engine(files, lora_path)` — keyed by files +
  the LoRA's path/mtime/size.

Florence PEFT LoRA support (`nlapt/local/lora.py`): safetensors + adapter_config.json
parsed with stdlib+numpy, matched to ONNX MatMul weights by a protobuf walk (NO `onnx`
package, no torch), merged as `M + scale·downᵀ@upᵀ` and injected at session creation
via `SessionOptions.add_initializer`. Two traps: the OrtValue wrappers MUST outlive the
session (ORT keeps raw pointers — dropping them segfaults; they're tied to the session
as `nlapt_lora_keepalive`), and matching needs `florence.LORA_MODULE_PREFIXES[file]`
passed as `module_prefix` (onnx-community's encoder export has bare `/layers.N/...`
node names — a suffix alone cannot tell encoder from decoder). Curated downloadable
LoRAs live in the catalog as `LoraEntry`/`ALL_LORAS` (full `base_url`, e.g.
ModelScope) and download to `models_dir/loras/<lora_id>/`; user-picked safetensors
are a plain path in `LocalSettings.florence_lora`.

The model catalog (families AND curated LoRAs) and the llama.cpp runtime table are
hardcoded snapshots (exact byte sizes + SHA256 from the HuggingFace / GitHub /
ModelScope APIs) — update the data AND `CATALOG_SNAPSHOT_DATE` /
`RUNTIME_SNAPSHOT_DATE` together.

Per-user state lives in `%APPDATA%/NLapt` (override with env `NLAPT_DATA_DIR`; GUI tests
isolate it automatically): `config.json` (non-interface AppConfig), `ui_settings.json`,
`translate.json` (provider / fallback / MT tier), `vision_prompts.json`, `local_llm.json`,
plus the `thumbs/` thumbnail cache and `runtime/` (auto-provisioned llama.cpp). All
interface credentials (LLM profiles, translate channels, CHA own-API) live in
`Documents/NLapt/api.json` (`NLAPT_DOCUMENTS_DIR` isolates the Documents root in tests).

## Hard rules (enforced by tests or bitter experience)

- **Immutability**: public data types are `@dataclass(frozen=True)`; transforms return new
  values. Mutable state only in the explicit stores (CaptionStore, UndoStack, …).
- **Typed errors** from `nlapt.core.errors` at every boundary; never silently swallow.
  Loggers via `nlapt.diagnostics.get_logger(__name__)`, never `print()`; mask secrets.
- **Files ≤800 lines** (200–400 typical). Type hints + `from __future__ import annotations`.
- **Language**: code/docstrings/comments in English; user-facing strings are Chinese,
  defined as module-top constants (never inline literals).
- **No hardcoded colors in widgets** — everything from `nlapt_gui/theme/tokens.py` through
  the QSS or token lookups; a test greps widget modules for hex literals. In QSS, quote
  any `url()` containing `;` — unquoted it silently truncates the rest of the stylesheet.
- **Never `QGraphicsOpacityEffect`** on live widgets — effect buffers hard-crash Qt during
  native window resizes (a test greps for it). Animate `windowOpacity` or paint-level
  alpha instead; gate all animations behind `nlapt_gui.anim` (disabled in tests).
- **GUI threading**: blocking work goes through `nlapt_gui/workers.py::run_async` on the
  QThreadPool; results come back via queued signals. Core `EventBus` handlers may fire on
  worker threads — bridge to the UI by emitting Qt signals only. Guard async replies with
  `shiboken6.isValid` when the receiver may be torn down.
- **Version bumps**: `nlapt/__init__.py::__version__` + `pyproject.toml` + the pinned
  assertions in `tests/core/test_core_models.py` and
  `tests/diagnostics/test_diagnostics_debug_manager.py`.

## Testing conventions

Tests mirror the package layout (`tests/<pkg>/test_<module>.py`; GUI files are
`tests/gui/test_gui_<module>.py`, globally unique names). GUI tests run offscreen —
`tests/gui/conftest.py` sets `QT_QPA_PLATFORM` before any Qt import, isolates
`NLAPT_DATA_DIR` per test, disables animations, and provides `demo_dataset`/`controller`
fixtures. No network and no real sleeps anywhere: inject clocks/sleeps/transports/openers
(every module in `nlapt/local` and the LLM stack takes injectable callables). Fake LLMs:
`nlapt.llm.base.register_client` + `MockLLMClient`, one unique `api_type` per test module.
`httpx.MockTransport` drives the HTTP clients.

Local-inference hermeticity: tests that reach `start_download`/`start_server` must set a
manual `server_path` or monkeypatch `local_bridge.ensure_runtime` (else the runtime
auto-provision hits the real network), and set explicit `gpu_layers` or patch
`local_bridge.detect_hardware` (自动 probes real hardware). Florence tests inject
`session_factory` fakes — onnxruntime is never required by the suite. Process-wide
singletons (server manager, download hub, idle stopper, Florence engine) outlive any
test — patch them (e.g. `vision_bridge.get_idle_stopper`) instead of letting real 30s
timers arm; `tests/gui/test_gui_vision_bridge.py` has an autouse isolation fixture to
copy.

The default models dir is the REPO's own `models/` in a source checkout (NLAPT_DATA_DIR
isolation does NOT cover it) — any test that writes fake model/LoRA files or asserts
download state must first `bridge.update_settings(models_dir=str(tmp_path))`, or it
reads (and can CLOBBER) real downloaded weights on the dev machine.

Windows console is GBK: printing Chinese file content in the terminal shows mojibake
even when the file is valid UTF-8 — verify content with Python assertions
(`assert '某词' in text`), never by eyeballing shell output.

## Packaging

`packaging/nlapt.spec` anchors paths on `SPECPATH` and needs explicit
`collect_submodules("nlapt"/"nlapt_gui")` hiddenimports plus the lazily imported deps
(`httpx`, `PIL.Image`, `onnxruntime`, `numpy`) — the codebase uses lazy/string imports
(PEP 562 `__getattr__`, self-registering LLM clients, in-function imports) invisible to
static analysis. All runtime paths go through `nlapt_gui/resources.py`. When smoke-testing the
windowed exe: a crash keeps the process alive showing a dialog — check the window title
and a fresh startup line in `%APPDATA%/NLapt/logs/nlapt.log`, not just process existence.

## Git

Conventional commits (`feat:`/`fix:`/…), no attribution trailers. Repo is public at
github.com/storyAura/NLapt (branch `main`). `models/`, `dist/`, `build/`, `NLapt-head/`,
`anli.md` (the user's personal notes) and all per-user config stay untracked.
