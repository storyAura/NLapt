"""Tests for nlapt_gui.tagger_bridge: ready check, gated download, identify."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlapt.core.errors import NLaptError
from nlapt.local.catalog import (
    CHARACTER_CSV_FILENAME,
    TAGGER_FAMILY,
    TAGGER_FAMILY_ID,
    family_dir,
    find_family,
    quant_path,
)
from nlapt.local.character_index import CharacterIndex
from nlapt.local.tagger import (
    FILE_MODEL,
    FILE_VOCAB,
    FILE_WEIGHTS,
    REQUIRED_FILES,
    TagResult,
    TagScore,
)
from nlapt.local.settings import load_local_settings, save_local_settings

from nlapt_gui.local_bridge import local_settings_path
from nlapt_gui.tagger_bridge import (
    CHARACTER_MIN_PROB,
    MSG_NEED_HF_TOKEN,
    MSG_NOT_READY,
    REL_CHILD,
    REL_DETECTED,
    REL_PARENT,
    REL_SIBLING,
    CharacterCandidate,
    IdentifyResult,
    TaggerBridge,
    get_character_index,
    get_tagger_engine,
    identify_characters,
    is_tagger_ready,
    release_tagger_engine,
    start_tagger_download,
    tagger_files,
)

CSV = """character_tag,other_names,copyright,parent_tag,post_count
kageyama_shien,,hololive,,1000
kageyama_shien_(1st_costume),,hololive,kageyama_shien,100
kageyama_shien_(2nd_costume),,hololive,kageyama_shien,200
hatsune_miku,,vocaloid,,5000
"""


@pytest.fixture(autouse=True)
def _isolate_models_and_singletons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = local_settings_path()
    current = load_local_settings(path)
    save_local_settings(path, current.with_changes(models_dir=str(tmp_path / "models")))
    monkeypatch.setattr("nlapt_gui.download_hub._ACTIVE_TASK", None)
    yield
    release_tagger_engine()
    import nlapt_gui.tagger_bridge as bridge

    bridge._CHAR_INDEX = None
    bridge._CHAR_INDEX_KEY = ("", 0, 0)


def _plant_tagger(models_dir: Path, *, csv_text: str = CSV) -> dict[str, Path]:
    family = find_family(TAGGER_FAMILY_ID)
    dest = family_dir(models_dir, family)
    dest.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for spec in (family.quants[0], *family.extra_files):
        path = quant_path(models_dir, family, spec)
        path.write_bytes(b"x")
        files[path.name] = path
    csv_path = dest / CHARACTER_CSV_FILENAME
    csv_path.write_text(csv_text, encoding="utf-8")
    files[CHARACTER_CSV_FILENAME] = csv_path
    return files


class FakeEngine:
    def __init__(self, result: TagResult) -> None:
        self.result = result
        self.paths: list[Path] = []

    def tag(self, image_path: Path, threshold: float = CHARACTER_MIN_PROB) -> TagResult:
        self.paths.append(Path(image_path))
        assert threshold == CHARACTER_MIN_PROB
        return self.result


class TestReady:
    def test_missing_files_are_not_ready(self, tmp_path: Path) -> None:
        assert tagger_files() is None
        assert is_tagger_ready() is False

    def test_empty_file_is_not_ready(self, tmp_path: Path) -> None:
        files = _plant_tagger(tmp_path / "models")
        files[FILE_WEIGHTS].write_bytes(b"")
        assert is_tagger_ready() is False

    def test_planted_files_are_ready(self, tmp_path: Path) -> None:
        planted = _plant_tagger(tmp_path / "models")
        ready = tagger_files()
        assert ready is not None
        assert set(REQUIRED_FILES) <= set(ready)
        assert ready[CHARACTER_CSV_FILENAME] == planted[CHARACTER_CSV_FILENAME]
        assert is_tagger_ready() is True


class TestDownload:
    def test_empty_token_refuses(self) -> None:
        assert start_tagger_download("") is False
        assert start_tagger_download("   ") is False
        assert MSG_NEED_HF_TOKEN

    def test_jobs_and_bearer_header(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_launch(family_id, quant, jobs, runtime, pool=None, headers=None):  # noqa: ANN001
            captured.update(
                family_id=family_id,
                quant=quant,
                jobs=jobs,
                headers=headers,
            )
            return True

        monkeypatch.setattr("nlapt_gui.tagger_bridge.launch_download_jobs", fake_launch)
        assert start_tagger_download("hf_abc") is True
        assert captured["family_id"] == TAGGER_FAMILY_ID
        assert captured["headers"] == {"Authorization": "Bearer hf_abc"}
        jobs = captured["jobs"]
        assert isinstance(jobs, list) and len(jobs) == 4
        assert all(expected == 0 and sha == "" for _url, _dest, expected, sha in jobs)
        dests = {Path(dest).name for _url, dest, _e, _s in jobs}
        assert dests == {
            FILE_MODEL,
            FILE_WEIGHTS,
            FILE_VOCAB,
            CHARACTER_CSV_FILENAME,
        }
        urls = [url for url, _dest, _e, _s in jobs]
        assert any("cella110n/cl_tagger_v2" in url for url in urls)
        assert any("Danbooru-Dataset-csv" in url for url in urls)
        assert TAGGER_FAMILY.gated is True

    def test_all_present_skips_launch(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _plant_tagger(tmp_path / "models")
        launched: list[int] = []
        monkeypatch.setattr(
            "nlapt_gui.tagger_bridge.launch_download_jobs",
            lambda *_a, **_k: launched.append(1) or True,
        )
        assert start_tagger_download("hf_abc") is True
        assert launched == []


class TestIdentify:
    def test_relations_and_order(self, tmp_path: Path) -> None:
        files = _plant_tagger(tmp_path / "models")
        index = CharacterIndex.load(files[CHARACTER_CSV_FILENAME])
        engine = FakeEngine(
            TagResult(
                characters=(
                    TagScore("kageyama shien (1st costume)", "Character", 0.91),
                    TagScore("unknown extra", "Character", 0.40),
                ),
                copyrights=(TagScore("hololive", "Copyright", 0.80),),
            )
        )
        result = identify_characters(tmp_path / "ref.png", engine=engine, index=index)
        by_tag = {item.tag: item for item in result.candidates}
        assert by_tag["kageyama_shien_(1st_costume)"].relation == REL_DETECTED
        assert by_tag["kageyama_shien"].relation == REL_PARENT
        assert by_tag["kageyama_shien_(2nd_costume)"].relation == REL_SIBLING
        assert by_tag["unknown_extra"].relation == REL_DETECTED
        assert by_tag["unknown_extra"].series_display == "Hololive"
        assert by_tag["kageyama_shien"].display_name == "Kageyama Shien"
        assert by_tag["kageyama_shien_(1st_costume)"].prob == pytest.approx(0.91)
        assert by_tag["kageyama_shien"].prob == 0.0

    def test_detected_parent_marks_costumes_as_children(self, tmp_path: Path) -> None:
        files = _plant_tagger(tmp_path / "models")
        index = CharacterIndex.load(files[CHARACTER_CSV_FILENAME])
        engine = FakeEngine(
            TagResult(characters=(TagScore("kageyama shien", "Character", 0.7),))
        )
        result = identify_characters(tmp_path / "ref.png", engine=engine, index=index)
        by_tag = {item.tag: item.relation for item in result.candidates}
        assert by_tag["kageyama_shien"] == REL_DETECTED
        assert by_tag["kageyama_shien_(1st_costume)"] == REL_CHILD

    def test_not_ready_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NLaptError, match="下载 CL Tagger"):
            identify_characters(tmp_path / "ref.png")
        assert MSG_NOT_READY

    def test_engine_singleton_replaced_after_release(self, tmp_path: Path) -> None:
        files = _plant_tagger(tmp_path / "models")
        first = get_tagger_engine(files, session_factory=lambda _p: object())
        assert get_tagger_engine(files) is first
        release_tagger_engine()
        second = get_tagger_engine(files, session_factory=lambda _p: object())
        assert second is not first
        assert get_character_index(files[CHARACTER_CSV_FILENAME]).lookup("hatsune miku")


class TestBridgeAsync:
    def test_not_ready_emits_failed(self, qtbot) -> None:
        bridge = TaggerBridge()
        failed: list[tuple[str, str]] = []
        bridge.identify_failed.connect(lambda rid, msg: failed.append((rid, msg)))
        assert bridge.request_identify("identify-1", Path("ref.png")) is False
        assert failed == [("identify-1", MSG_NOT_READY)]

    def test_ready_emits_identify_result(
        self, qtbot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _plant_tagger(tmp_path / "models")
        payload = IdentifyResult(
            candidates=(
                CharacterCandidate(
                    tag="hatsune_miku",
                    display_name="Hatsune Miku",
                    series_tag="vocaloid",
                    series_display="Vocaloid",
                    prob=0.88,
                    relation=REL_DETECTED,
                ),
            )
        )
        monkeypatch.setattr(
            "nlapt_gui.tagger_bridge.identify_characters",
            lambda _path, **_k: payload,
        )
        bridge = TaggerBridge()
        seen: list[IdentifyResult] = []
        bridge.identify_ready.connect(lambda _rid, result: seen.append(result))
        assert bridge.request_identify("identify-7", tmp_path / "ref.png") is True
        qtbot.waitUntil(lambda: bool(seen), timeout=2000)
        assert seen[0] is payload
