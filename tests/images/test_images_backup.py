"""Tests for nlapt.images.backup (copy backups + quarantine + restore)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image

from nlapt.images.backup import KIND_BACKUP, KIND_QUARANTINE, ImageBackupManager


def _png(path: Path, color: tuple[int, int, int] = (12, 34, 56)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (3, 3), color).save(path, format="PNG")
    return path


def test_create_copies_and_restore_overwrites(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    image = _png(root / "sub" / "a.png", (1, 2, 3))
    manager = ImageBackupManager(root)
    info = manager.create("flatten_alpha", (image,))
    assert info.kind == KIND_BACKUP
    assert info.file_count == 1
    assert (info.path / "sub" / "a.png").is_file()
    Image.new("RGB", (3, 3), (9, 9, 9)).save(image, format="PNG")
    restored = manager.restore(info)
    assert restored == ("sub/a.png",)
    with Image.open(image) as opened:
        assert opened.getpixel((0, 0)) == (1, 2, 3)


def test_quarantine_moves_image_and_txt(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    image = _png(root / "a.png")
    txt = root / "a.txt"
    txt.write_text("hello", encoding="utf-8")
    manager = ImageBackupManager(root)
    info = manager.quarantine((image,))
    assert info.kind == KIND_QUARANTINE
    assert not image.exists()
    assert not txt.exists()
    assert (info.path / "a.png").is_file()
    assert (info.path / "a.txt").read_text(encoding="utf-8") == "hello"
    manager.restore(info)
    assert image.is_file()
    assert txt.read_text(encoding="utf-8") == "hello"


def test_retention_keeps_five(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    image = _png(root / "a.png")
    manager = ImageBackupManager(root, retention=2)
    first = manager.create("one", (image,))
    second = manager.create("two", (image,))
    third = manager.create("three", (image,))
    listed = [info.path for info in manager.list_backups() if info.kind == KIND_BACKUP]
    assert first.path not in listed
    assert second.path in listed
    assert third.path in listed
