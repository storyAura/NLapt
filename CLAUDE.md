# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
pip install -e .[gui,images,llm,dev]        # PySide6 + Pillow + httpx + pytest

python -m pytest -q                          # full suite (~25s, must stay green)
python -m pytest tests/gui -q                # GUI only (offscreen, no display needed)
python -m pytest tests/local/test_local_catalog.py::TestLookups::test_find_family -q
python -m pytest --cov=nlapt --cov=nlapt_gui -q      # coverage (project target ≥80%, currently ~94%)
python -m ruff check nlapt nlapt_gui tests   # lint (default rules)

python -m nlapt_gui                          # run the app
Build-NLapt.bat                              # PyInstaller onedir build → dist/NLapt/NLapt.exe
```

## Binding contracts

`docs/ARCHITECTURE.md` (core) and `docs/UI_ARCHITECTURE.md` (GUI) are **binding interface
contracts**, maintained as versioned addendum sections (v1.1 … v1.6.1). Follow the public
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
  history (per-file undo + operation log rollback), local (GGUF catalog / hardware
  advisor / downloader / llama-server manager).
- **`nlapt_gui/`** — PySide6 UI. **Widgets never touch the filesystem or core packages
  directly**: everything goes through `nlapt_gui/controller.py::AppController` (the only
  importer of `nlapt.app`) plus the async bridges (`translate_bridge`, `vision_bridge`,
  `local_bridge`). Sanctioned exception: `widgets/settings_dialog.py` and
  `widgets/local_tab.py` may read/write the core `config.json` directly.

Data-safety model the UI relies on: every batch operation snapshots all txt files to
`.backups/` **before** executing and is recorded in the operation log for whole-batch
rollback; txt files are written only on explicit save; unsaved drafts survive crashes via
the session file in `.nlapt/`.

Local inference (`nlapt/local`) reuses the existing OpenAI-compatible client stack by
launching llama.cpp's `llama-server` (`http://127.0.0.1:{port}/v1`). The model catalog is
a hardcoded snapshot (exact byte sizes + SHA256 from the HuggingFace API) — update the
data AND `CATALOG_SNAPSHOT_DATE` together. The server manager is a process-wide singleton
(`local_bridge.get_server_manager()`, stopped via `atexit`).

Per-user state lives in `%APPDATA%/NLapt` (override with env `NLAPT_DATA_DIR`; GUI tests
isolate it automatically): `config.json` (LLM profiles — the only file with API keys),
`ui_settings.json`, `translate.json`, `vision_prompts.json`, `local_llm.json`.

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

## Packaging

`packaging/nlapt.spec` anchors paths on `SPECPATH` and needs explicit
`collect_submodules("nlapt"/"nlapt_gui")` hiddenimports — the codebase uses lazy/string
imports (PEP 562 `__getattr__`, self-registering LLM clients) invisible to static
analysis. All runtime paths go through `nlapt_gui/resources.py`. When smoke-testing the
windowed exe: a crash keeps the process alive showing a dialog — check the window title
and a fresh startup line in `%APPDATA%/NLapt/logs/nlapt.log`, not just process existence.

## Git

Conventional commits (`feat:`/`fix:`/…), no attribution trailers. Repo is public at
github.com/storyAura/NLapt (branch `main`). `models/`, `dist/`, `build/`, `NLapt-head/`
and all per-user config stay untracked.
