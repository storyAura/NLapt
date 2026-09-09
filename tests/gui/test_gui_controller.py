"""Tests for nlapt_gui.controller (AppController behaviors)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtGui import QGuiApplication

from nlapt.app import NLaptApp
from nlapt.core.config import AppConfig, LLMProfile
from nlapt.core.errors import ValidationError
from nlapt.llm.base import register_client
from nlapt.llm.mock import MockLLMClient

from nlapt_gui.controller import FOLDER_ROOT_LABEL, AppController
from nlapt_gui.layered_prompts import count_words, estimate_tokens
from nlapt_gui.settings import UISettings

from nlapt.storage.paths import dataset_state_dir
from nlapt.storage.session import SESSION_FILE_NAME

from tests.gui.conftest import DEMO_CAPTIONS

K1 = "0001.png"
K2 = "0002.png"
K3 = "10_concept/0003.png"
K4 = "10_concept/0004.png"
ALL_KEYS = (K1, K2, K3, K4)

MOCK_API_TYPE = "mock-gui-controller"
register_client(MOCK_API_TYPE, lambda profile: MockLLMClient(responses=["translated"]))


class TestOpenDataset:
    def test_keys_in_natural_order(self, controller: AppController) -> None:
        assert controller.keys() == ALL_KEYS

    def test_current_is_first_key(self, controller: AppController) -> None:
        assert controller.current_key == K1

    def test_history_seeded_with_initial_label(self, controller: AppController) -> None:
        for key in ALL_KEYS:
            entries = controller.history.entries(key)
            assert len(entries) == 1
            assert entries[0].label == "载入原始标注"
        assert controller.history.entries(K1)[0].caption == DEMO_CAPTIONS[K1]

    def test_missing_txt_loads_empty(self, controller: AppController) -> None:
        assert controller.record(K4).text == ""

    def test_dataset_label_and_root(self, controller: AppController, demo_dataset: Path) -> None:
        name, path = controller.dataset_label()
        assert name == demo_dataset.name
        assert path == str(demo_dataset)
        assert controller.root == demo_dataset

    def test_refresh_keeps_root(self, controller: AppController, qtbot, demo_dataset: Path) -> None:
        with qtbot.waitSignal(controller.dataset_opened, timeout=2000):
            controller.refresh()
        assert controller.root == demo_dataset
        assert controller.keys() == ALL_KEYS

    def test_open_failed_emits_signal(self, qtbot, tmp_path: Path) -> None:
        ctrl = AppController(NLaptApp(), settings=UISettings())
        with qtbot.waitSignal(ctrl.dataset_open_failed, timeout=2000):
            ctrl.open_dataset(tmp_path / "does-not-exist")

    def test_encoding_issue_toast(self, qtbot, tmp_path: Path) -> None:
        from PIL import Image

        root = tmp_path / "gbk_set"
        root.mkdir()
        Image.new("RGB", (4, 3)).save(root / "a.png", format="PNG")
        (root / "a.txt").write_bytes("少女".encode("gb18030"))
        ctrl = AppController(NLaptApp(), settings=UISettings())
        collected: list[tuple[str, str]] = []
        ctrl.toast_requested.connect(lambda t, k: collected.append((t, k)))
        with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
            ctrl.open_dataset(root)
        qtbot.waitUntil(lambda: any("UTF-8" in t for t, _ in collected), timeout=2000)
        assert any(k == "warn" for _, k in collected)


class TestFoldersAndFilter:
    def test_folders_natural_order(self, controller: AppController) -> None:
        assert controller.folders() == (FOLDER_ROOT_LABEL, "10_concept")

    def test_folder_of(self, controller: AppController) -> None:
        assert controller.folder_of(K1) == FOLDER_ROOT_LABEL
        assert controller.folder_of(K3) == "10_concept"

    def test_unlabeled_keys_filters_empty_captions(self, controller: AppController) -> None:
        assert controller.unlabeled_keys(ALL_KEYS) == (K4,)
        assert controller.unlabeled_keys((K1, K2)) == ()

    def test_filter_by_name(self, controller: AppController, qtbot) -> None:
        with qtbot.waitSignal(controller.filter_changed, timeout=1000):
            controller.set_filter("0003")
        assert controller.filtered_keys() == (K3,)

    def test_filter_by_caption_case_insensitive(self, controller: AppController) -> None:
        controller.set_filter("YUKATA")
        assert controller.filtered_keys() == (K3,)
        controller.set_filter("少女")
        assert controller.filtered_keys() == (K1,)

    def test_empty_filter_returns_all(self, controller: AppController) -> None:
        controller.set_filter("")
        assert controller.filtered_keys() == ALL_KEYS


class TestSelection:
    def test_toggle_sets_anchor_and_selects(self, controller: AppController) -> None:
        controller.toggle_selected(K2)
        assert controller.is_selected(K2)
        assert controller.selected_keys() == (K2,)

    def test_current_jumps_to_first_selected_when_outside_selection(
        self, controller: AppController
    ) -> None:
        assert controller.current_key == K1
        controller.toggle_selected(K3)
        controller.toggle_selected(K4)
        # >=2 selected and K1 not in selection -> jump to first selected
        assert controller.current_key == K3

    def test_current_stays_when_inside_selection(self, controller: AppController) -> None:
        controller.toggle_selected(K1)
        controller.toggle_selected(K2)
        assert controller.current_key == K1

    def test_select_range_over_filtered_keys(self, controller: AppController) -> None:
        controller.set_anchor(K1)
        controller.select_range_to(K3)
        assert controller.selected_keys() == (K1, K2, K3)

    def test_select_range_uses_filtered_list(self, controller: AppController) -> None:
        controller.set_filter("1girl")  # K1..K3 (K4 empty caption)
        controller.set_anchor(K1)
        controller.select_range_to(K3)
        assert controller.selected_keys() == (K1, K2, K3)
        controller.clear_selection()
        controller.set_filter("hair")  # K1, K2
        controller.set_anchor(K1)
        controller.select_range_to(K2)
        assert controller.selected_keys() == (K1, K2)

    def test_deselect_and_clear(self, controller: AppController) -> None:
        controller.select_all()
        assert controller.selected_keys() == ALL_KEYS
        controller.deselect(K2)
        assert controller.selected_keys() == (K1, K3, K4)
        controller.clear_selection()
        assert controller.selected_keys() == ()

    def test_multi_mode_threshold(self, controller: AppController) -> None:
        assert not controller.multi_mode()
        controller.toggle_selected(K1)
        assert not controller.multi_mode()
        controller.toggle_selected(K2)
        assert controller.multi_mode()

    def test_editor_keys_single_and_multi(self, controller: AppController) -> None:
        assert controller.editor_keys() == (K1,)
        controller.select_all()
        assert controller.editor_keys() == ALL_KEYS[:4]

    def test_selection_changed_emitted(self, controller: AppController, qtbot) -> None:
        with qtbot.waitSignal(controller.selection_changed, timeout=1000):
            controller.toggle_selected(K1)


class TestNavigation:
    def test_nav_wraps_forward_and_backward(self, controller: AppController) -> None:
        controller.nav(-1)
        assert controller.current_key == K4  # wrap from first to last
        controller.nav(1)
        assert controller.current_key == K1

    def test_nav_within_filter(self, controller: AppController) -> None:
        controller.set_filter("hair")  # K1, K2
        controller.nav(1)
        assert controller.current_key == K2
        controller.nav(1)
        assert controller.current_key == K1

    def test_nav_does_nothing_when_filter_has_no_matches(
        self, controller: AppController
    ) -> None:
        controller.set_filter("zzz-no-match")
        controller.nav(1)
        assert controller.current_key == K1
        controller.nav(-1)
        assert controller.current_key == K1
        assert not controller.can_navigate()

    def test_nav_locked_to_selection_in_multi_mode(self, controller: AppController) -> None:
        controller.toggle_selected(K2)
        controller.toggle_selected(K3)
        assert controller.current_key == K2
        controller.nav(1)
        assert controller.current_key == K3
        controller.nav(1)
        assert controller.current_key == K2  # wraps inside selection

    def test_pos_label_single(self, controller: AppController) -> None:
        assert controller.pos_label() == "1 / 4"
        controller.set_current(K3)
        assert controller.pos_label() == "3 / 4"

    def test_pos_label_multi(self, controller: AppController) -> None:
        controller.toggle_selected(K2)
        controller.toggle_selected(K4)
        assert controller.current_key == K2
        assert controller.pos_label() == "1 / 2"
        controller.set_current(K4)
        assert controller.pos_label() == "2 / 2"

    def test_pos_label_current_outside_filter(self, controller: AppController) -> None:
        controller.set_current(K2)
        controller.set_filter("0003")  # current not in filtered list
        assert controller.pos_label() == "未匹配 / 1"


class TestEditing:
    def test_set_caption_pushes_history_and_emits(
        self, controller: AppController, qtbot
    ) -> None:
        with qtbot.waitSignal(controller.caption_changed, timeout=1000) as blocker:
            controller.set_caption(K1, "new caption", "自由编辑")
        assert blocker.args == [K1]
        assert controller.record(K1).text == "new caption"
        assert controller.record(K1).dirty
        entries = controller.history.entries(K1)
        assert entries[0].label == "自由编辑"
        assert entries[0].caption == "new caption"

    def test_set_caption_noop_when_unchanged(self, controller: AppController, qtbot) -> None:
        with qtbot.assertNotEmitted(controller.caption_changed, wait=50):
            controller.set_caption(K1, DEMO_CAPTIONS[K1], "自由编辑")
        assert len(controller.history.entries(K1)) == 1

    def test_set_caption_no_history(self, controller: AppController, qtbot) -> None:
        with qtbot.waitSignal(controller.caption_changed, timeout=1000):
            controller.set_caption_no_history(K1, "silent edit")
        assert controller.record(K1).text == "silent edit"
        assert len(controller.history.entries(K1)) == 1  # only the seed

    def test_commit_segments_joins_chips(self, controller: AppController) -> None:
        controller.commit_segments(K2, ("a", " b ", "", "c"), "拖拽排序")
        assert controller.record(K2).text == "a, b, c"
        assert controller.history.entries(K2)[0].label == "拖拽排序"

    def test_undo_current(self, controller: AppController, toasts) -> None:
        controller.set_caption(K1, "edited", "编辑")
        controller.undo_current()
        assert controller.record(K1).text == DEMO_CAPTIONS[K1]
        assert ("已撤销", "info") in toasts

    def test_undo_nothing_to_undo(self, controller: AppController, toasts) -> None:
        controller.set_current(K2)
        controller.undo_current()
        assert ("没有可撤销的操作", "info") in toasts
        assert controller.record(K2).text == DEMO_CAPTIONS[K2]

    def test_redo_current(self, controller: AppController, toasts) -> None:
        controller.set_caption(K1, "edited", "编辑")
        controller.undo_current()
        controller.redo_current()
        assert controller.record(K1).text == "edited"
        assert ("已重做", "info") in toasts
        assert not controller.can_redo()
        assert controller.can_undo()

    def test_redo_nothing_to_redo(self, controller: AppController, toasts) -> None:
        controller.redo_current()
        assert ("没有可重做的操作", "info") in toasts

    def test_segments_and_char_seg_info(self, controller: AppController) -> None:
        assert controller.segments(K3) == ("1girl", "yukata", "fireworks")
        text = DEMO_CAPTIONS[K3]
        assert controller.char_seg_info(K3) == (
            f"{len(text)} 字符 · 3 段"
            f" · 约 {estimate_tokens(text)} tokens · {count_words(text)} 词"
        )


class TestSaving:
    def test_save_current_writes_txt_and_toasts(
        self, controller: AppController, toasts, demo_dataset: Path, qtbot
    ) -> None:
        controller.set_caption(K1, "saved text", "编辑")
        with qtbot.waitSignal(controller.files_saved, timeout=1000) as blocker:
            controller.save_current()
        assert blocker.args == [(K1,)]
        assert (demo_dataset / "0001.txt").read_text(encoding="utf-8") == "saved text"
        assert not controller.record(K1).dirty
        assert ("已保存 0001.txt", "ok") in toasts

    def test_save_current_without_changes(self, controller: AppController, toasts) -> None:
        controller.save_current()
        assert ("没有未保存的更改", "info") in toasts

    def test_save_all(
        self, controller: AppController, toasts, demo_dataset: Path, qtbot
    ) -> None:
        controller.set_caption(K1, "one", "编辑")
        controller.set_caption(K3, "three", "编辑")
        with qtbot.waitSignal(controller.files_saved, timeout=1000):
            controller.save_all()
        assert ("已保存 2 个文件", "ok") in toasts
        assert controller.dirty_count() == 0
        assert (demo_dataset / "10_concept" / "0003.txt").read_text(encoding="utf-8") == "three"

    def test_save_all_without_changes(self, controller: AppController, toasts) -> None:
        controller.save_all()
        assert ("没有未保存的更改", "info") in toasts

    def test_copy_caption(self, controller: AppController, toasts) -> None:
        controller.copy_caption()
        assert QGuiApplication.clipboard().text() == DEMO_CAPTIONS[K1]
        assert ("已复制标注文本", "ok") in toasts

    def test_export_dataset(
        self, controller: AppController, toasts, demo_dataset: Path, qtbot, tmp_path
    ) -> None:
        dest = tmp_path / "pack.zip"
        with qtbot.waitSignal(controller.toast_requested, timeout=2000):
            controller.export_dataset(dest)
        assert dest.is_file()
        assert any(text.startswith("已导出") for text, _kind in toasts)

    def test_export_refused_while_batch_in_flight(
        self, controller: AppController, toasts, tmp_path: Path
    ) -> None:
        controller._batch_in_flight = True
        try:
            controller.export_dataset(tmp_path / "unused.zip")
        finally:
            controller._batch_in_flight = False
        assert any("正在处理" in text for text, _kind in toasts)
        assert not (tmp_path / "unused.zip").exists()

    def test_export_without_dataset(self, qtbot, qapp) -> None:
        ctrl = AppController(NLaptApp(), settings=UISettings())
        seen: list[tuple[str, str]] = []
        ctrl.toast_requested.connect(lambda text, kind: seen.append((text, kind)))
        ctrl.export_dataset(Path("unused.zip"))
        assert ("请先打开数据集", "warn") in seen


class TestModesAndStats:
    def test_mode_change_emits(self, controller: AppController, qtbot) -> None:
        with qtbot.waitSignal(controller.mode_changed, timeout=1000) as blocker:
            controller.set_mode("text")
        assert blocker.args == ["text"]
        assert controller.mode == "text"

    def test_invalid_mode_raises(self, controller: AppController) -> None:
        with pytest.raises(ValidationError):
            controller.set_mode("nope")

    def test_view_mode_change_emits_and_persists(self, controller: AppController, qtbot) -> None:
        with qtbot.waitSignal(controller.view_mode_changed, timeout=1000):
            controller.set_view_mode("big")
        assert controller.view_mode == "big"
        assert controller.settings.view_mode == "big"

    def test_invalid_view_mode_raises(self, controller: AppController) -> None:
        with pytest.raises(ValidationError):
            controller.set_view_mode("huge")

    def test_stat_line(self, controller: AppController) -> None:
        controller.toggle_selected(K1)
        controller.set_caption(K2, "changed", "编辑")
        assert controller.stat_line() == "4 张图片 · 已选 1 · 未保存 1"

    def test_image_meta(self, controller: AppController) -> None:
        meta = controller.image_meta(K1)
        assert meta.startswith("4 × 3 · PNG · ")
        # cached second call
        assert controller.image_meta(K1) is meta

    def test_image_format_and_mtime_label(self, controller: AppController) -> None:
        assert controller.image_format(K1) == "PNG"
        assert controller.image_modified_label(K1).startswith("修改于 ")


class TestScopesAndBatches:
    def test_scope_keys(self, controller: AppController) -> None:
        assert controller.scope_keys("current") == (K1,)
        assert controller.scope_keys("all") == ALL_KEYS
        controller.toggle_selected(K2)
        controller.toggle_selected(K3)
        assert controller.scope_keys("selected") == (K2, K3)
        with pytest.raises(ValidationError):
            controller.scope_keys("everything")

    def test_folder_scope_includes_nested_keys(self, controller: AppController) -> None:
        controller.set_current(K3)
        assert controller.scope_keys("folder") == (K3, K4)
        controller.set_current(K1)
        assert controller.scope_keys("folder") == ALL_KEYS
        assert controller.folder_tree_keys("10_concept") == (K3, K4)
        assert controller.folder_tree_keys(FOLDER_ROOT_LABEL) == ALL_KEYS

    def test_count_matches(self, controller: AppController) -> None:
        assert controller.count_matches("1girl", False, "all") == (3, 3)
        assert controller.count_matches("hair", False, "all") == (2, 2)
        assert controller.count_matches("1GIRL", True, "all") == (0, 0)
        assert controller.count_matches("1GIRL", False, "all") == (3, 3)
        assert controller.count_matches("", False, "all") == (0, 0)

    def test_count_matches_inside_sentences(self, controller: AppController) -> None:
        # 匹配的是全文子串,不受逗号分段限制:句子内部的词组同样命中。
        assert controller.count_matches("樱花树", False, "all") == (1, 1)
        assert controller.count_matches("school uni", False, "all") == (1, 1)

    def test_count_matches_whole_word(self, controller: AppController) -> None:
        # 子串模式:"girl" 命中每个 "1girl";整词模式下词内子串不再误伤。
        assert controller.count_matches("girl", False, "all") == (3, 3)
        assert controller.count_matches("girl", False, "all", whole_word=True) == (0, 0)
        assert controller.count_matches("hair", False, "all", whole_word=True) == (2, 2)

    def test_replace_all_whole_word(
        self, controller: AppController, toasts, qtbot
    ) -> None:
        controller.replace_all("girl", "woman", False, "all", whole_word=True)
        assert ("没有找到匹配内容", "warn") in toasts
        with qtbot.waitSignal(controller.batch_finished, timeout=2000):
            controller.replace_all("quality", "value", False, "all", whole_word=True)
        assert controller.record(K2).text.endswith("best value")

    def test_replace_all_full_flow(
        self, controller: AppController, toasts, qtbot, demo_dataset: Path
    ) -> None:
        with qtbot.waitSignal(controller.batch_finished, timeout=2000) as blocker:
            controller.replace_all("1girl", "1woman", False, "all")
        description, report = blocker.args
        assert "查找替换" in description
        assert report.snapshot is not None  # zip snapshot taken before the batch
        assert report.succeeded == 4
        assert controller.record(K1).text.startswith("1woman")
        assert controller.record(K3).text.startswith("1woman")
        # saved to disk by the batch engine
        assert "1woman" in (demo_dataset / "0002.txt").read_text(encoding="utf-8")
        assert ("已在 3 个文件中替换 3 处", "ok") in toasts
        assert controller.history.entries(K1)[0].label == "查找替换 ×1"

    def test_replace_all_empty_find_warns(self, controller: AppController, toasts) -> None:
        controller.replace_all("", "x", False, "all")
        assert ("请输入查找内容", "warn") in toasts

    def test_replace_all_selected_scope_without_selection(
        self, controller: AppController, toasts
    ) -> None:
        controller.replace_all("1girl", "x", False, "selected")
        assert ("尚未选择任何图片", "warn") in toasts

    def test_replace_all_no_match_warns(self, controller: AppController, toasts) -> None:
        controller.replace_all("zzz-not-there", "x", False, "all")
        assert ("没有找到匹配内容", "warn") in toasts

    def test_apply_prefix_suffix_as_tag(
        self, controller: AppController, toasts, qtbot
    ) -> None:
        with qtbot.waitSignal(controller.batch_finished, timeout=2000):
            controller.apply_prefix_suffix("aoba", "prefix", True, "all")
        assert controller.record(K1).text == "aoba, " + DEMO_CAPTIONS[K1]
        assert controller.record(K4).text == "aoba"  # empty caption -> prefix alone
        assert ("已为 4 个文件添加前缀", "ok") in toasts
        assert controller.history.entries(K1)[0].label == "添加前缀「aoba」"

    def test_apply_prefix_skip_if_present(
        self, controller: AppController, toasts, qtbot
    ) -> None:
        with qtbot.waitSignal(controller.batch_finished, timeout=2000):
            controller.apply_prefix_suffix("aoba", "prefix", True, "all")
        with qtbot.waitSignal(controller.batch_finished, timeout=2000):
            controller.apply_prefix_suffix("aoba", "prefix", True, "all")
        # second run changes nothing (trigger word not duplicated)
        assert controller.record(K1).text.count("aoba") == 1
        assert ("已为 0 个文件添加前缀", "ok") in toasts

    def test_apply_suffix(self, controller: AppController, toasts, qtbot) -> None:
        with qtbot.waitSignal(controller.batch_finished, timeout=2000):
            controller.apply_prefix_suffix("hq", "suffix", True, "current")
        assert controller.record(K1).text.endswith(", hq")
        assert ("已为 1 个文件添加后缀", "ok") in toasts
        assert controller.history.entries(K1)[0].label == "添加后缀「hq」"

    def test_apply_prefix_empty_text_warns(self, controller: AppController, toasts) -> None:
        controller.apply_prefix_suffix("   ", "prefix", True, "all")
        assert ("请输入前缀/后缀内容", "warn") in toasts

    def test_apply_prefix_invalid_position_raises(self, controller: AppController) -> None:
        with pytest.raises(ValidationError):
            controller.apply_prefix_suffix("x", "middle", True, "all")

    def test_busy_signal_wraps_batch(self, controller: AppController, qtbot) -> None:
        seen: list[bool] = []
        controller.busy_changed.connect(seen.append)
        with qtbot.waitSignal(controller.batch_finished, timeout=2000):
            controller.replace_all("1girl", "x", False, "all")
        assert seen[0] is True
        assert seen[-1] is False


class TestServices:
    def test_translator_none_when_unconfigured(self, controller: AppController) -> None:
        assert controller.make_translator_or_none() is None
        # cached (still None, no exception storm)
        assert controller.make_translator_or_none() is None

    def test_translator_built_from_active_profile(self, qtbot, demo_dataset: Path) -> None:
        profile = LLMProfile(
            name="default",
            api_type=MOCK_API_TYPE,
            base_url="http://localhost:9",
            text_model="mock-model",
        )
        config = AppConfig(profiles=(profile,), active_profile="default")
        ctrl = AppController(NLaptApp(config=config), settings=UISettings())
        with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
            ctrl.open_dataset(demo_dataset)
        translator = ctrl.make_translator_or_none()
        assert translator is not None
        assert ctrl.make_translator_or_none() is translator  # cached
        result = translator.translate(K1, "少女")
        assert result.translated == "translated"

    def test_invalidate_translator(self, controller: AppController) -> None:
        assert controller.make_translator_or_none() is None
        controller.invalidate_translator()
        assert controller.make_translator_or_none() is None

    def test_vision_captioner_uses_override_profile(
        self, qtbot, demo_dataset: Path
    ) -> None:
        recorded: list[MockLLMClient] = []
        api_type = "mock-gui-controller-vision"

        def factory(profile: LLMProfile) -> MockLLMClient:
            client = MockLLMClient(["ok-caption"])
            recorded.append(client)
            return client

        register_client(api_type, factory)
        active = LLMProfile(
            name="default",
            api_type=api_type,
            base_url="http://active",
            text_model="t",
            vision_model="active-vis",
        )
        override = LLMProfile(
            name="cha",
            api_type=api_type,
            base_url="http://override",
            text_model="t",
            vision_model="card-vis",
        )
        config = AppConfig(profiles=(active,), active_profile="default")
        ctrl = AppController(NLaptApp(config=config), settings=UISettings())
        with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
            ctrl.open_dataset(demo_dataset)
        assert ctrl.vision_profile() is not None
        assert ctrl.vision_profile().vision_model == "active-vis"
        assert ctrl.text_profile().text_model == "t"
        captioner = ctrl.make_vision_captioner_or_none(profile=override)
        assert captioner is not None
        captioner(ctrl.image_path(K1), "sys", "user")
        assert recorded
        assert recorded[0].requests[0].model == "card-vis"

    def test_model_pool_and_set_model_target(self, qtbot, demo_dataset: Path) -> None:
        from nlapt.core.config import ModelRef
        from nlapt.core.errors import ValidationError

        from nlapt_gui.api_config import load_api_config
        from nlapt_gui.model_targets import ROLE_TEXT, ROLE_VISION

        alpha = LLMProfile(
            name="alpha",
            api_type="mock-gui-controller-pool",
            base_url="http://alpha",
            models=("a-text", "a-vl"),
            enabled_models=("a-text", "a-vl"),
        )
        beta = LLMProfile(
            name="beta",
            api_type="mock-gui-controller-pool",
            base_url="http://beta",
            models=("b-vl",),
            enabled_models=("b-vl",),
        )
        config = AppConfig(
            profiles=(alpha, beta),
            text_target=ModelRef("alpha", "a-text"),
            vision_target=ModelRef("alpha", "a-vl"),
        )
        ctrl = AppController(NLaptApp(config=config), settings=UISettings())
        with qtbot.waitSignal(ctrl.dataset_opened, timeout=2000):
            ctrl.open_dataset(demo_dataset)
        assert [c.ref for c in ctrl.model_pool()] == [
            ModelRef("alpha", "a-text"),
            ModelRef("alpha", "a-vl"),
            ModelRef("beta", "b-vl"),
        ]
        bound = ctrl.profile_for_ref(ModelRef("beta", "b-vl"))
        assert bound is not None and bound.base_url == "http://beta"
        assert bound.vision_model == "b-vl"
        assert ctrl.profile_for_ref(ModelRef("beta", "nope")) is None
        ctrl.set_model_target(ROLE_VISION, ModelRef("beta", "b-vl"))
        assert ctrl.vision_profile().base_url == "http://beta"
        assert ctrl.vision_profile().vision_model == "b-vl"
        assert ctrl.text_profile().text_model == "a-text"
        saved = load_api_config()
        assert saved.vision_target == ModelRef("beta", "b-vl")
        assert saved.text_target == ModelRef("alpha", "a-text")
        with pytest.raises(ValidationError):
            ctrl.set_model_target(ROLE_TEXT, ModelRef("beta", "nope"))
        assert ctrl.text_profile().text_model == "a-text"


class TestClose:
    def test_close_persists_session_and_settings(
        self, controller: AppController, demo_dataset: Path
    ) -> None:
        controller.set_caption(K1, "unsaved draft", "编辑")
        controller.set_view_mode("list")
        controller.close()
        # crash-safe session written; txt NOT written (explicit-save model)
        assert (dataset_state_dir(demo_dataset) / SESSION_FILE_NAME).exists()
        assert not (demo_dataset / ".nlapt").exists()
        assert (demo_dataset / "0001.txt").read_text(encoding="utf-8") == DEMO_CAPTIONS[K1]
        from nlapt_gui.settings import load_ui_settings

        assert load_ui_settings().view_mode == "list"


class TestCaptionBatch:
    """run_caption_batch: 推标 writes, history, progress, guards."""

    def test_batch_writes_saves_and_records_history(
        self, qtbot, controller, demo_dataset
    ) -> None:
        progress: list[tuple[int, int]] = []
        controller.batch_progress.connect(
            lambda _d, done, total: progress.append((done, total))
        )
        started: list[tuple[str, int]] = []
        controller.batch_started.connect(lambda d, t: started.append((d, t)))
        keys = ("0001.png", "0002.png")
        with qtbot.waitSignal(controller.batch_finished, timeout=4000) as blocker:
            assert controller.run_caption_batch(
                keys,
                lambda key, path: f"cap {key}",
                description="推标(LLM) · 2 张",
                history_label="推标(LLM)",
                engine="llm",
                concurrency=2,
            )
        report = blocker.args[1]
        assert report.succeeded == 2 and report.failed == 0
        for key in keys:
            record = controller.record(key)
            assert record.text == f"cap {key}"
            assert not record.dirty  # batch saves to disk
            assert controller.history.entries(key)[0].label == "推标(LLM)"
        assert (demo_dataset / "0001.txt").read_text(encoding="utf-8") == "cap 0001.png"
        assert progress and progress[-1] == (2, 2)
        assert started == [("推标(LLM) · 2 张", 2)]

    def test_failed_item_keeps_original_caption(self, qtbot, controller) -> None:
        original = controller.record("0002.png").text

        def caption(key, path):  # noqa: ANN001
            if key == "0002.png":
                raise RuntimeError("boom")
            return "ok caption"

        with qtbot.waitSignal(controller.batch_finished, timeout=4000) as blocker:
            assert controller.run_caption_batch(
                ("0001.png", "0002.png"),
                caption,
                description="推标(本地模型) · 2 张",
                history_label="推标(本地)",
            )
        report = blocker.args[1]
        assert report.failed == 1 and report.failed_keys == ("0002.png",)
        assert controller.record("0002.png").text == original

    def test_empty_keys_refused_with_toast(self, controller, toasts) -> None:
        assert not controller.run_caption_batch(
            (), lambda k, p: "x", description="d", history_label="h"
        )
        assert toasts  # 尚未选择任何图片

    def test_busy_guard_refuses_second_batch(self, controller, toasts) -> None:
        controller._batch_in_flight = True
        try:
            assert not controller.run_caption_batch(
                ("0001.png",), lambda k, p: "x", description="d", history_label="h"
            )
        finally:
            controller._batch_in_flight = False
        assert any("正在处理" in text for text, _ in toasts)

    def test_on_finished_hook_runs(self, qtbot, controller) -> None:
        seen: list[object] = []
        with qtbot.waitSignal(controller.batch_finished, timeout=4000):
            controller.run_caption_batch(
                ("0001.png",),
                lambda key, path: "hooked",
                description="d",
                history_label="h",
                on_finished=seen.append,
            )
        qtbot.waitUntil(lambda: bool(seen), timeout=2000)
        assert seen[0] is not None

    def test_batch_running_and_cancel_noop_when_idle(self, controller) -> None:
        assert not controller.batch_running()
        controller.cancel_batch()  # no controller -> silently does nothing
