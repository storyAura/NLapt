"""Tests for nlapt.batch.checkpoint: stable ids and the persistent store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlapt.batch.checkpoint import (
    CHECKPOINT_FILE_NAME,
    CORRUPT_FILE_SUFFIX,
    CheckpointStore,
    make_checkpoint_id,
)
from nlapt.core.errors import ValidationError
from nlapt.storage.session import SESSION_DIR_NAME

KEYS = ("b.txt", "a.txt", "c.txt")
PARAMS = "find='girl' replace='woman'"


def _checkpoint_path(root: Path) -> Path:
    return root / SESSION_DIR_NAME / CHECKPOINT_FILE_NAME


# -- make_checkpoint_id --------------------------------------------------------


def test_id_is_stable_across_calls() -> None:
    first = make_checkpoint_id("replace", KEYS, PARAMS)
    second = make_checkpoint_id("replace", KEYS, PARAMS)
    assert first == second
    assert len(first) == 40  # sha1 hex digest
    int(first, 16)  # must be valid hex


def test_id_independent_of_key_order() -> None:
    shuffled = ("c.txt", "b.txt", "a.txt")
    assert make_checkpoint_id("replace", KEYS, PARAMS) == make_checkpoint_id(
        "replace", shuffled, PARAMS
    )


def test_id_differs_for_different_params() -> None:
    assert make_checkpoint_id("replace", KEYS, PARAMS) != make_checkpoint_id(
        "replace", KEYS, "find='cat' replace='dog'"
    )


def test_id_differs_for_different_operation_and_keys() -> None:
    base = make_checkpoint_id("replace", KEYS, PARAMS)
    assert base != make_checkpoint_id("rewrite", KEYS, PARAMS)
    assert base != make_checkpoint_id("replace", KEYS + ("d.txt",), PARAMS)


@pytest.mark.parametrize(
    ("operation", "keys", "params"),
    [
        ("", KEYS, PARAMS),
        ("   ", KEYS, PARAMS),
        ("replace", "a.txt", PARAMS),  # bare string is not a key sequence
        ("replace", (1, 2), PARAMS),
        ("replace", KEYS, 42),
    ],
)
def test_id_validates_arguments(operation, keys, params) -> None:
    with pytest.raises(ValidationError):
        make_checkpoint_id(operation, keys, params)


# -- CheckpointStore -----------------------------------------------------------


def test_unknown_id_has_empty_completed_set(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    assert store.completed("deadbeef") == frozenset()
    assert store.has("deadbeef") is False


def test_state_dir_writes_outside_dataset(tmp_path: Path) -> None:
    dest = tmp_path / "state"
    store = CheckpointStore(tmp_path, state_dir=dest)
    store.mark("cid", "a.txt")
    assert (dest / CHECKPOINT_FILE_NAME).is_file()
    assert not _checkpoint_path(tmp_path).exists()
    reloaded = CheckpointStore(tmp_path, state_dir=dest)
    assert reloaded.completed("cid") == frozenset({"a.txt"})


def test_mark_persists_and_reloads(tmp_path: Path) -> None:
    cid = make_checkpoint_id("replace", KEYS, PARAMS)
    store = CheckpointStore(tmp_path)
    store.mark(cid, "a.txt")
    store.mark(cid, "b.txt")
    assert store.completed(cid) == frozenset({"a.txt", "b.txt"})
    assert store.has(cid) is True
    assert _checkpoint_path(tmp_path).is_file()
    # A fresh store instance (new process after a crash) sees the same state.
    reloaded = CheckpointStore(tmp_path)
    assert reloaded.completed(cid) == frozenset({"a.txt", "b.txt"})


def test_mark_is_idempotent(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.mark("cid", "a.txt")
    store.mark("cid", "a.txt")
    assert store.completed("cid") == frozenset({"a.txt"})
    payload = json.loads(_checkpoint_path(tmp_path).read_text(encoding="utf-8"))
    assert payload == {"cid": ["a.txt"]}


def test_persisted_json_is_sorted_object(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.mark("cid", "b.txt")
    store.mark("cid", "a.txt")
    payload = json.loads(_checkpoint_path(tmp_path).read_text(encoding="utf-8"))
    assert payload == {"cid": ["a.txt", "b.txt"]}


def test_clear_removes_checkpoint_and_is_idempotent(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.mark("cid", "a.txt")
    store.clear("cid")
    assert store.has("cid") is False
    assert store.completed("cid") == frozenset()
    store.clear("cid")  # idempotent, no error
    payload = json.loads(_checkpoint_path(tmp_path).read_text(encoding="utf-8"))
    assert payload == {}


def test_independent_checkpoint_ids(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    store.mark("one", "a.txt")
    store.mark("two", "b.txt")
    store.clear("one")
    assert store.has("one") is False
    assert store.completed("two") == frozenset({"b.txt"})


def test_root_must_be_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        CheckpointStore(tmp_path / "missing")


@pytest.mark.parametrize("content", ["{not json", "[]", '{"cid": "a.txt"}', '{"cid": [1]}'])
def test_corrupt_file_self_heals_to_empty_store(tmp_path: Path, content: str) -> None:
    path = _checkpoint_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")

    store = CheckpointStore(tmp_path)  # must not raise (self-heal)

    assert store.has("cid") is False
    assert store.completed("cid") == frozenset()
    # Bad file moved aside for inspection; original path free for fresh writes.
    corrupt = path.with_name(path.name + CORRUPT_FILE_SUFFIX)
    assert corrupt.read_text(encoding="utf-8") == content
    assert not path.exists()
    store.mark("cid", "a.txt")  # store is fully usable afterwards
    assert json.loads(path.read_text(encoding="utf-8")) == {"cid": ["a.txt"]}


def test_store_validates_ids_and_keys(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path)
    with pytest.raises(ValidationError):
        store.completed("")
    with pytest.raises(ValidationError):
        store.has("")
    with pytest.raises(ValidationError):
        store.mark("", "a.txt")
    with pytest.raises(ValidationError):
        store.mark("cid", "")
    with pytest.raises(ValidationError):
        store.clear("")
