"""Regression tests for review fixes, Group 1 (storage & core).

Covers FIXPLAN items:
- 1.1 mask_secret boundary leak (findings [0])
- 1.2 restore() destroying the snapshot being restored (findings [1], [13])
- 1.3 UTF-8 BOM + non-UTF-8 body mojibake (finding [6])
"""

from __future__ import annotations

import codecs
import json
import zipfile
from datetime import datetime
from pathlib import Path

import pytest

from nlapt.core.config import (
    AppConfig,
    LLMProfile,
    MIN_MASKABLE_LENGTH,
    mask_secret,
    masked_config_dict,
)
from nlapt.storage import snapshots as snapshots_module
from nlapt.storage.snapshots import BACKUP_DIR_NAME, SnapshotManager
from nlapt.storage.text_io import read_text_detect, write_caption

# ---------------------------------------------------------------------------
# 1.1 mask_secret boundary leak
# ---------------------------------------------------------------------------

MIN_PARTIAL_REVEAL_LENGTH = 12


def test_min_maskable_length_is_twelve() -> None:
    assert MIN_MASKABLE_LENGTH == MIN_PARTIAL_REVEAL_LENGTH


@pytest.mark.parametrize("length", [8, 9, 10, 11])
def test_mask_secret_fully_masks_lengths_eight_to_eleven(length: int) -> None:
    secret = "k" * length
    assert mask_secret(secret) == "***"


@pytest.mark.parametrize("length", range(1, MIN_PARTIAL_REVEAL_LENGTH))
def test_mask_secret_hides_at_least_half_below_reveal_threshold(length: int) -> None:
    secret = "abcdefghijk"[:length]
    masked = mask_secret(secret)
    # Below the reveal threshold nothing at all is revealed, so trivially at
    # least half of the secret remains hidden.
    assert masked == "***"
    revealed = sum(1 for ch in secret if ch in masked.replace("***", ""))
    assert revealed == 0


def test_mask_secret_reveals_only_from_twelve_chars() -> None:
    assert mask_secret("abcdefghijkl") == "abc***ijkl"
    assert mask_secret("sk-secret123456789") == "sk-***6789"


def test_masked_config_dict_does_not_leak_short_api_key() -> None:
    short_key = "abcd1234"  # 8 chars: previously leaked 7 of 8 characters
    config = AppConfig(
        profiles=(
            LLMProfile(name="p", api_type="openai", base_url="u", api_key=short_key),
        ),
        active_profile="p",
    )
    dumped = json.dumps(masked_config_dict(config))
    assert short_key not in dumped
    assert "abc" not in dumped.replace('"api_key": "***"', "")
    data = masked_config_dict(config)
    assert data["profiles"][0]["api_key"] == "***"


# ---------------------------------------------------------------------------
# 1.2 restore() must never destroy the snapshot being restored
# ---------------------------------------------------------------------------


@pytest.fixture
def frozen_now(monkeypatch: pytest.MonkeyPatch):
    """Freeze snapshots._now; returns a setter to advance the clock."""
    holder = {"value": datetime(2026, 7, 11, 10, 0)}
    monkeypatch.setattr(snapshots_module, "_now", lambda: holder["value"])

    def set_time(value: datetime) -> None:
        holder["value"] = value

    return set_time


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    (tmp_path / "img1.txt").write_bytes(b"caption one\n")
    (tmp_path / "img2.txt").write_bytes(b"caption two\n")
    return tmp_path


def test_restore_oldest_snapshot_at_retention_one_succeeds(
    dataset: Path, frozen_now
) -> None:
    manager = SnapshotManager(dataset, retention=1)
    frozen_now(datetime(2026, 7, 11, 10, 0))
    info = manager.create("op")
    (dataset / "img1.txt").write_bytes(b"MODIFIED")

    frozen_now(datetime(2026, 7, 11, 10, 5))
    result = manager.restore(info)

    # The restore succeeded and the file content is back.
    assert (dataset / "img1.txt").read_bytes() == b"caption one\n"
    assert "img1.txt" in result.restored_files
    # The restored-from zip still exists (retention did not destroy it).
    assert info.path.exists()
    # The pre-restore snapshot also survives so the restore can be undone.
    assert result.pre_restore_snapshot.path.exists()
    with zipfile.ZipFile(result.pre_restore_snapshot.path) as archive:
        assert archive.read("img1.txt") == b"MODIFIED"


def test_restore_oldest_snapshot_at_retention_capacity_succeeds(
    dataset: Path, frozen_now
) -> None:
    retention = 3
    manager = SnapshotManager(dataset, retention=retention)
    infos = []
    for minute in range(retention):
        frozen_now(datetime(2026, 7, 11, 10, minute))
        infos.append(manager.create(f"op{minute}"))
    oldest = infos[0]
    (dataset / "img1.txt").write_bytes(b"MODIFIED")

    frozen_now(datetime(2026, 7, 11, 11, 0))
    result = manager.restore(oldest)

    assert (dataset / "img1.txt").read_bytes() == b"caption one\n"
    assert oldest.path.exists()
    assert result.pre_restore_snapshot.path.exists()


def test_restore_target_bytes_survive_even_if_zip_pruned(
    dataset: Path, frozen_now, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Belt-and-braces: bytes are read before create(), so a restore completes
    correctly even when retention pruning is not suppressed for the target."""
    manager = SnapshotManager(dataset, retention=1)
    frozen_now(datetime(2026, 7, 11, 10, 0))
    info = manager.create("op")
    (dataset / "img1.txt").write_bytes(b"MODIFIED")

    original_enforce = SnapshotManager._enforce_retention

    def prune_without_exclusion(self: SnapshotManager, **_kwargs) -> None:
        original_enforce(self)  # simulate pruning that ignores the exclusion

    monkeypatch.setattr(SnapshotManager, "_enforce_retention", prune_without_exclusion)
    frozen_now(datetime(2026, 7, 11, 10, 5))
    result = manager.restore(info)

    assert (dataset / "img1.txt").read_bytes() == b"caption one\n"
    assert result.restored_files == ("img1.txt", "img2.txt")


def test_retention_exclusion_does_not_delete_newer_snapshots_instead(
    dataset: Path, frozen_now
) -> None:
    """Skipping the protected zip must not sacrifice a newer snapshot in its place."""
    manager = SnapshotManager(dataset, retention=1)
    frozen_now(datetime(2026, 7, 11, 10, 0))
    oldest = manager.create("op0")
    frozen_now(datetime(2026, 7, 11, 10, 5))
    result = manager.restore(oldest)

    names = sorted(p.name for p in (dataset / BACKUP_DIR_NAME).glob("*.zip"))
    assert oldest.path.name in names
    assert result.pre_restore_snapshot.path.name in names


# ---------------------------------------------------------------------------
# 1.3 UTF-8 BOM + non-UTF-8 body
# ---------------------------------------------------------------------------

CHINESE_TEXT = "一只猫，坐在窗台上"


def test_bom_plus_gbk_body_detected_as_gb18030(tmp_path: Path) -> None:
    path = tmp_path / "cap.txt"
    path.write_bytes(codecs.BOM_UTF8 + CHINESE_TEXT.encode("gb18030"))
    result = read_text_detect(path)
    assert result.text == CHINESE_TEXT
    assert result.encoding == "gb18030"
    assert result.needs_conversion is True


def test_bom_plus_gbk_body_convert_and_resave_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cap.txt"
    path.write_bytes(codecs.BOM_UTF8 + CHINESE_TEXT.encode("gb18030"))
    detected = read_text_detect(path)
    write_caption(path, detected.text)  # convert to UTF-8 as the app would
    reread = read_text_detect(path)
    assert reread.text == CHINESE_TEXT
    assert reread.encoding == "utf-8"
    assert reread.needs_conversion is False


def test_bom_plus_valid_utf8_body_still_uses_utf8_sig(tmp_path: Path) -> None:
    path = tmp_path / "cap.txt"
    path.write_bytes(codecs.BOM_UTF8 + "1girl, smile".encode("utf-8"))
    result = read_text_detect(path)
    assert result.text == "1girl, smile"
    assert result.encoding == "utf-8-sig"
    assert result.needs_conversion is True


def test_bom_plus_latin1_body_reports_latin1_not_mojibake_bom(tmp_path: Path) -> None:
    # Body is invalid UTF-8 / gb18030 / big5 / shift_jis -> latin-1 fallback,
    # but the BOM bytes must not appear in the decoded text.
    path = tmp_path / "cap.txt"
    path.write_bytes(codecs.BOM_UTF8 + b"caf\xe9")
    result = read_text_detect(path)
    assert result.text == "café"
    assert result.encoding == "latin-1"
    assert result.needs_conversion is True
