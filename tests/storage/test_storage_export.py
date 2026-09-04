"""Tests for nlapt.storage.export.export_dataset_zip."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from nlapt.core.errors import StorageError, ValidationError
from nlapt.core.models import ImageFile
from nlapt.storage.export import export_dataset_zip
from nlapt.storage.scanner import scan_dataset


def _image(root: Path, rel: str, *, caption: str | None = "caption") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG")
    if caption is not None:
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "set"
    _image(root, "a.png", caption="one")
    _image(root, "sub/b.png", caption="two")
    _image(root, "orphan.png", caption=None)
    (root / ".nlapt").mkdir()
    (root / ".nlapt" / "session.json").write_text("{}", encoding="utf-8")
    (root / ".backups").mkdir()
    (root / ".backups" / "snap.zip").write_bytes(b"nope")
    return root


class TestExportDatasetZip:
    def test_packs_images_and_existing_txts(self, dataset: Path, tmp_path: Path) -> None:
        result = scan_dataset(dataset)
        dest = tmp_path / "out.zip"
        count = export_dataset_zip(dataset, dest, result.images)
        assert dest.is_file()
        with zipfile.ZipFile(dest) as archive:
            names = set(archive.namelist())
        assert names == {"a.png", "a.txt", "sub/b.png", "sub/b.txt", "orphan.png"}
        assert count == 5
        assert ".nlapt" not in {Path(name).parts[0] for name in names}
        assert ".backups" not in {Path(name).parts[0] for name in names}
        with zipfile.ZipFile(dest) as archive:
            assert archive.read("a.txt") == b"one"

    def test_empty_file_list_raises(self, dataset: Path, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            export_dataset_zip(dataset, tmp_path / "empty.zip", ())

    def test_missing_parent_raises(self, dataset: Path, tmp_path: Path) -> None:
        result = scan_dataset(dataset)
        dest = tmp_path / "missing" / "out.zip"
        with pytest.raises(StorageError):
            export_dataset_zip(dataset, dest, result.images)

    def test_temp_cleaned_when_write_fails(
        self, dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nlapt.storage.atomic import TEMP_PREFIX

        dest = tmp_path / "out.zip"

        def boom(_tmp: Path, _files: object) -> int:
            raise RuntimeError("zip fail")

        monkeypatch.setattr("nlapt.storage.export._write_members", boom)
        with pytest.raises(StorageError, match="zip fail"):
            export_dataset_zip(dataset, dest, scan_dataset(dataset).images)
        assert not dest.exists()
        leftovers = [path for path in tmp_path.iterdir() if path.name.startswith(TEMP_PREFIX)]
        assert leftovers == []

    def test_skips_missing_image_bytes(self, tmp_path: Path) -> None:
        dest = tmp_path / "ghost.zip"
        ghost = ImageFile(
            key="gone.png",
            image_path=tmp_path / "gone.png",
            txt_path=tmp_path / "gone.txt",
            txt_exists=False,
            mtime=0.0,
        )
        count = export_dataset_zip(tmp_path, dest, (ghost,))
        assert count == 0
        with zipfile.ZipFile(dest) as archive:
            assert archive.namelist() == []
