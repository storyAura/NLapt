"""Suite-wide isolation: never write per-user state into the real %APPDATA%."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_nlapt_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point ``NLAPT_DATA_DIR`` at a per-test directory.

    GUI tests also set this (to ``tmp_path/appdata``); the later fixture wins.
    Core / app / regression tests would otherwise land snapshots and sessions
    in the developer's real ``%APPDATA%/NLapt``.
    """
    monkeypatch.setenv("NLAPT_DATA_DIR", str(tmp_path / "nlapt-data"))
    monkeypatch.setenv("NLAPT_DOCUMENTS_DIR", str(tmp_path / "documents"))


@pytest.fixture(autouse=True)
def _quiet_deeplx_limiter() -> None:
    """DeepLX's process-wide limiter must not sleep between tests."""
    from nlapt.llm.web_translate import reset_deeplx_limiter

    reset_deeplx_limiter(min_interval=0.0)
