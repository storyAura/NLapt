"""Shared fixtures for facade tests: a tiny on-disk dataset.

Images are 1-byte stub files — the scanner never decodes them; only txt
content matters for the flows under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

STUB_IMAGE_BYTES = b"\x89"

CAPTION_A = "a girl, smiling"
CAPTION_B = "a girl with hat"


def make_dataset(root: Path) -> Path:
    """Create a/b (labeled) + c (unlabeled) image/txt pairs under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.png").write_bytes(STUB_IMAGE_BYTES)
    (root / "a.txt").write_text(CAPTION_A, encoding="utf-8")
    (root / "b.png").write_bytes(STUB_IMAGE_BYTES)
    (root / "b.txt").write_text(CAPTION_B, encoding="utf-8")
    (root / "c.png").write_bytes(STUB_IMAGE_BYTES)
    return root


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    return make_dataset(tmp_path / "dataset")
