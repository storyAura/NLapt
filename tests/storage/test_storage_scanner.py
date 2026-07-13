"""Tests for nlapt.storage.scanner (spec 2.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.core.errors import ValidationError
from nlapt.storage.scanner import natural_sort_key, scan_dataset

IMAGE_BYTES = b"\x89PNG-fake"


def make_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(IMAGE_BYTES)
    return path


def make_txt(path: Path, text: str = "a caption") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestRootValidation:
    def test_missing_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            scan_dataset(tmp_path / "nope")

    def test_file_root_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "file.txt"
        target.write_text("x", encoding="utf-8")
        with pytest.raises(ValidationError):
            scan_dataset(target)

    def test_empty_root_ok(self, tmp_path: Path) -> None:
        result = scan_dataset(tmp_path)
        assert result.images == ()
        assert result.orphan_txts == ()
        assert result.root == tmp_path

    def test_relative_root_yields_absolute_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_image(tmp_path / "img.png")
        monkeypatch.chdir(tmp_path.parent)
        result = scan_dataset(Path(tmp_path.name))
        assert result.root.is_absolute()
        assert result.images[0].image_path.is_absolute()


class TestPairing:
    def test_paired_image_and_txt(self, tmp_path: Path) -> None:
        image = make_image(tmp_path / "img.jpg")
        txt = make_txt(tmp_path / "img.txt")
        result = scan_dataset(tmp_path)
        assert len(result.images) == 1
        item = result.images[0]
        assert item.key == "img.jpg"
        assert item.image_path == image
        assert item.txt_path == txt
        assert item.txt_exists is True
        assert isinstance(item.mtime, float)
        assert result.orphan_txts == ()

    def test_extension_matching_is_case_insensitive(self, tmp_path: Path) -> None:
        make_image(tmp_path / "IMG.JPG")
        txt = make_txt(tmp_path / "img.TXT")
        result = scan_dataset(tmp_path)
        assert len(result.images) == 1
        item = result.images[0]
        assert item.key == "IMG.JPG"
        assert item.txt_exists is True
        assert item.txt_path == txt
        assert result.orphan_txts == ()

    def test_all_supported_image_extensions(self, tmp_path: Path) -> None:
        names = ["a.jpg", "b.jpeg", "c.png", "d.webp", "e.bmp"]
        for name in names:
            make_image(tmp_path / name)
        make_image(tmp_path / "skip.gif")  # unsupported
        (tmp_path / "notes.md").write_text("x", encoding="utf-8")  # unrelated
        result = scan_dataset(tmp_path)
        assert [item.key for item in result.images] == sorted(names)

    def test_unlabeled_image_has_default_txt_path(self, tmp_path: Path) -> None:
        make_image(tmp_path / "photo.png")
        result = scan_dataset(tmp_path)
        item = result.images[0]
        assert item.txt_exists is False
        assert item.txt_path == tmp_path / "photo.txt"

    def test_pairing_requires_same_directory(self, tmp_path: Path) -> None:
        make_image(tmp_path / "sub" / "img.png")
        make_txt(tmp_path / "img.txt")  # same stem, wrong directory
        result = scan_dataset(tmp_path)
        assert result.images[0].txt_exists is False
        assert result.orphan_txts == (tmp_path / "img.txt",)


class TestOrphans:
    def test_orphan_txt_collected(self, tmp_path: Path) -> None:
        make_image(tmp_path / "img.png")
        make_txt(tmp_path / "img.txt")
        orphan = make_txt(tmp_path / "lonely.txt")
        result = scan_dataset(tmp_path)
        assert result.orphan_txts == (orphan,)

    def test_orphan_txt_extension_case_insensitive(self, tmp_path: Path) -> None:
        orphan = make_txt(tmp_path / "lonely.TXT")
        result = scan_dataset(tmp_path)
        assert result.orphan_txts == (orphan,)


class TestRecursion:
    def test_kohya_layout_keys_use_posix_relative_paths(self, tmp_path: Path) -> None:
        make_image(tmp_path / "10_concept" / "img.png")
        make_txt(tmp_path / "10_concept" / "img.txt")
        result = scan_dataset(tmp_path)
        item = result.images[0]
        assert item.key == "10_concept/img.png"
        assert item.txt_exists is True
        assert item.image_path.is_absolute()
        assert item.txt_path.is_absolute()

    def test_non_recursive_skips_subdirectories(self, tmp_path: Path) -> None:
        make_image(tmp_path / "top.png")
        make_image(tmp_path / "sub" / "nested.png")
        make_txt(tmp_path / "sub" / "nested.txt")
        result = scan_dataset(tmp_path, recursive=False)
        assert [item.key for item in result.images] == ["top.png"]
        assert result.orphan_txts == ()

    def test_skips_backups_nlapt_and_dot_directories(self, tmp_path: Path) -> None:
        make_image(tmp_path / "keep.png")
        make_image(tmp_path / ".backups" / "old.png")
        make_txt(tmp_path / ".backups" / "old.txt")
        make_image(tmp_path / ".nlapt" / "cache.png")
        make_txt(tmp_path / ".hidden" / "secret.txt")
        make_image(tmp_path / "sub" / ".nested-hidden" / "deep.png")
        result = scan_dataset(tmp_path)
        assert [item.key for item in result.images] == ["keep.png"]
        assert result.orphan_txts == ()


class TestOrdering:
    def test_images_natural_sorted_by_key(self, tmp_path: Path) -> None:
        for name in ("img10.png", "img2.png", "img1.png", "IMG3.png"):
            make_image(tmp_path / name)
        result = scan_dataset(tmp_path)
        assert [item.key for item in result.images] == [
            "img1.png",
            "img2.png",
            "IMG3.png",
            "img10.png",
        ]

    def test_directories_natural_sorted(self, tmp_path: Path) -> None:
        make_image(tmp_path / "10_b" / "img.png")
        make_image(tmp_path / "2_a" / "img.png")
        result = scan_dataset(tmp_path)
        assert [item.key for item in result.images] == [
            "2_a/img.png",
            "10_b/img.png",
        ]

    def test_scan_is_deterministic(self, tmp_path: Path) -> None:
        for name in ("b.png", "a.png", "c.png"):
            make_image(tmp_path / name)
        make_txt(tmp_path / "z-orphan.txt")
        make_txt(tmp_path / "a-orphan.txt")
        first = scan_dataset(tmp_path)
        second = scan_dataset(tmp_path)
        assert first == second
        assert [p.name for p in first.orphan_txts] == ["a-orphan.txt", "z-orphan.txt"]


class TestNaturalSortKey:
    def test_numeric_runs_compare_numerically(self) -> None:
        names = ["file10", "file2", "file1"]
        assert sorted(names, key=natural_sort_key) == ["file1", "file2", "file10"]

    def test_case_insensitive(self) -> None:
        assert natural_sort_key("ABC") == natural_sort_key("abc")
