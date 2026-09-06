"""Tests for the 本地推理 tab (nlapt_gui/widgets/local_tab.py)."""

from __future__ import annotations

from typing import Iterator

import pytest
from PySide6.QtWidgets import QTreeWidgetItem

from nlapt.app import NLaptApp
from nlapt.core.config import load_config
from nlapt.core.errors import LocalServerError
from nlapt.local.catalog import all_series, families_for
from nlapt.local.hardware import GIB, GpuInfo, HardwareInfo
from nlapt.local.settings import LocalSettings, load_local_settings, save_local_settings

from nlapt_gui.api_config import load_app_config
from nlapt_gui.controller import AppController
from nlapt_gui.local_bridge import LocalBridge
from nlapt_gui.resources import app_data_dir
from nlapt_gui.settings import UISettings
from nlapt_gui.widgets.local_tab import (
    HW_UNKNOWN,
    ROLE_FAMILY,
    ROLE_QUANT,
    LocalTab,
)
from nlapt_gui.widgets.settings_dialog import config_path

BIG_RIG = HardwareInfo(
    cpu_cores=16,
    ram_total_bytes=64 * GIB,
    ram_available_bytes=40 * GIB,
    gpus=(GpuInfo(name="RTX 5090", vram_total_bytes=32 * GIB, vram_free_bytes=30 * GIB),),
)
NO_GPU_RIG = HardwareInfo(
    cpu_cores=8, ram_total_bytes=64 * GIB, ram_available_bytes=40 * GIB
)
TINY_RIG = HardwareInfo(cpu_cores=2, ram_total_bytes=2 * GIB, ram_available_bytes=1 * GIB)
UNDETECTED = HardwareInfo(cpu_cores=0, ram_total_bytes=0, ram_available_bytes=0)


class FakeManager:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.running = False
        self.stop_calls = 0

    def is_running(self) -> bool:
        return self.running

    @property
    def current_base_url(self) -> str:
        return "http://127.0.0.1:18434/v1" if self.running else ""

    def start(self, spec) -> str:  # noqa: ANN001
        if self.fail:
            raise LocalServerError("boom")
        self.running = True
        return f"http://127.0.0.1:{spec.port}/v1"

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False


@pytest.fixture()
def tab_controller(qtbot) -> Iterator[AppController]:
    yield AppController(NLaptApp(), settings=UISettings())


@pytest.fixture()
def tab_toasts(tab_controller) -> list[tuple[str, str]]:
    collected: list[tuple[str, str]] = []
    tab_controller.toast_requested.connect(
        lambda text, kind: collected.append((text, kind))
    )
    return collected


def make_tab(qtbot, controller: AppController, **kwargs) -> LocalTab:
    tab = LocalTab(controller, **kwargs)
    qtbot.addWidget(tab)
    return tab


def quant_item(tab: LocalTab, family_id: str, quant_label: str) -> QTreeWidgetItem:
    for item in tab._quant_items():
        if (
            item.data(0, ROLE_FAMILY) == family_id
            and item.data(0, ROLE_QUANT) == quant_label
        ):
            return item
    raise AssertionError(f"quant row {family_id}/{quant_label} not found")


class TestTreeStructure:
    def test_top_level_rows_match_series(self, qtbot, tab_controller) -> None:
        # One top-level row per series; single-family series show the FAMILY
        # name directly (不分层), which always starts with the series name.
        tab = make_tab(qtbot, tab_controller)
        assert tab.tree.topLevelItemCount() == len(all_series())
        for index, series in enumerate(all_series()):
            text = tab.tree.topLevelItem(index).text(0)
            families = families_for(series.series_id)
            if len(families) == 1:
                assert text.startswith(families[0].name)
            else:
                assert text == series.name

    def test_every_catalog_quant_has_exactly_one_row(
        self, qtbot, tab_controller
    ) -> None:
        from nlapt.local.catalog import all_families

        tab = make_tab(qtbot, tab_controller)
        pairs = sorted(
            (item.data(0, ROLE_FAMILY), item.data(0, ROLE_QUANT))
            for item in tab._quant_items()
        )
        expected = sorted(
            (family.family_id, quant.label)
            for family in all_families()
            for quant in family.quants
        )
        assert pairs == expected

    def test_multi_family_series_keep_the_nested_shape(
        self, qtbot, tab_controller
    ) -> None:
        tab = make_tab(qtbot, tab_controller)
        item = quant_item(tab, "gemma4-12b", "Q4_K_M")
        family_row = item.parent()
        series_row = family_row.parent()
        assert series_row is not None and series_row.text(0) == "Gemma 4"

    def test_single_family_series_flattened_to_top_level(
        self, qtbot, tab_controller
    ) -> None:
        # ToriiGate has one family (several quants): no series row.
        tab = make_tab(qtbot, tab_controller)
        family_row = quant_item(tab, "toriigate-0.5", "Q4_K_M").parent()
        assert family_row.parent() is None
        assert family_row.text(0).startswith("ToriiGate 0.5")

    def test_single_quant_family_is_one_selectable_row(
        self, qtbot, tab_controller
    ) -> None:
        # A single-quant family is its own selectable row carrying size,
        # downloads and grade itself; the Florence series (5 families since
        # the official models joined) keeps its series node above them.
        from PySide6.QtCore import Qt

        tab = make_tab(qtbot, tab_controller)
        item = quant_item(tab, "florence2-promptgen-v2", "ONNX")
        assert item.parent() is not None  # the Florence-2 series row
        assert item.parent().text(0) == "Florence-2"
        assert item.childCount() == 0
        assert item.flags() & Qt.ItemFlag.ItemIsSelectable
        assert item.text(2)  # 热度 shown on the merged row
        for family_id in (
            "florence2-base-ft",
            "florence2-large-ft",
            "florence2-base",
            "florence2-large",
        ):
            assert quant_item(tab, family_id, "ONNX") is not None

    def test_requested_models_have_rows(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        assert quant_item(tab, "gemma4-12b", "Q4_K_M") is not None
        assert quant_item(tab, "gemma4-12b-heretic", "Q4_K_M") is not None
        assert quant_item(tab, "toriigate-0.5", "Q4_K_M") is not None
        assert quant_item(tab, "joycaption-beta-one", "Q4_K") is not None

    def test_quant_rows_show_sizes(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        item = quant_item(tab, "toriigate-0.5", "Q4_K_M")
        assert "GB" in item.text(1)


class TestHardwareAndVerdicts:
    def test_show_triggers_async_detection(
        self, qtbot, tab_controller, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "nlapt_gui.local_bridge.detect_hardware", lambda: BIG_RIG
        )
        tab = make_tab(qtbot, tab_controller)
        assert tab.bridge.hardware is None
        tab.show()
        qtbot.waitUntil(lambda: tab.bridge.hardware is not None, timeout=2000)
        assert "16" in tab.hw_label.text()
        assert "RTX 5090" in tab.hw_label.text()

    def test_perfect_grade_on_big_rig(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab._on_hardware_ready(BIG_RIG)
        item = quant_item(tab, "toriigate-0.5", "Q4_K_M")
        assert item.text(3) == "轻松运行"

    def test_cpu_grade_without_gpu(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab._on_hardware_ready(NO_GPU_RIG)
        assert "未检测到 NVIDIA" in tab.hw_label.text()
        item = quant_item(tab, "toriigate-0.5", "Q4_K_M")
        assert item.text(3) == "可以运行"

    def test_no_grade_on_tiny_rig(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab._on_hardware_ready(TINY_RIG)
        item = quant_item(tab, "gemma4-31b", "Q8_0")
        assert item.text(3) == "跑不动"
        assert "还差" in item.toolTip(3)

    def test_detection_failure_shows_unknown(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab._on_hardware_ready(UNDETECTED)
        assert tab.hw_label.text() == HW_UNKNOWN
        item = quant_item(tab, "toriigate-0.5", "Q4_K_M")
        assert item.text(3) == "未检测"

    def test_context_change_recomputes_verdicts(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab._on_hardware_ready(BIG_RIG)
        tab.context_spin.setValue(32_768)
        item = quant_item(tab, "toriigate-0.5", "Q4_K_M")
        assert item.text(3) != "未检测"

    def test_grade_cell_is_bold(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        item = quant_item(tab, "toriigate-0.5", "Q4_K_M")
        assert item.font(3).bold()
        family_item = item.parent()
        assert family_item.font(0).bold()


class TestSelection:
    def test_no_selection_initially(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        assert tab.current_selection() is None
        assert not tab.apply_button.isEnabled()
        assert not tab.page_button.isEnabled()

    def test_selecting_quant_updates_detail(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        selection = tab.current_selection()
        assert selection is not None
        assert selection[0].family_id == "toriigate-0.5"
        assert "预计占用" in tab.detail_label.text()
        assert tab.download_button.isEnabled()
        assert tab.page_button.isEnabled()
        assert not tab.apply_button.isEnabled()  # not downloaded yet


class TestPersistAndPrefill:
    def test_persist_writes_settings_file(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, "joycaption-beta-one", "Q4_K"))
        tab.context_spin.setValue(2048)
        tab.parallel_spin.setValue(6)
        tab.port_spin.setValue(2000)
        tab.server_path_edit.setText("C:/llama/llama-server.exe")
        tab.extra_dirs_list.addItems(["D:/shared-models", "E:/lmstudio"])
        tab.persist()
        stored = load_local_settings(app_data_dir() / "local_llm.json")
        assert stored.context_length == 2048
        assert stored.parallel == 6
        assert stored.port == 2000
        assert stored.server_path == "C:/llama/llama-server.exe"
        assert stored.extra_dirs == ("D:/shared-models", "E:/lmstudio")
        assert stored.family_id == "joycaption-beta-one"
        assert stored.quant_label == "Q4_K"

    def test_prefill_restores_saved_selection(self, qtbot, tab_controller) -> None:
        save_local_settings(
            app_data_dir() / "local_llm.json",
            LocalSettings(
                port=3000,
                parallel=7,
                extra_dirs=("D:/shared",),
                family_id="gemma4-12b",
                quant_label="Q8_0",
            ),
        )
        tab = make_tab(qtbot, tab_controller)
        assert tab.port_spin.value() == 3000
        assert tab.parallel_spin.value() == 7
        assert tab.extra_dirs_list.count() == 1
        assert tab.extra_dirs_list.item(0).text() == "D:/shared"
        selection = tab.current_selection()
        assert selection is not None
        assert selection[0].family_id == "gemma4-12b"
        assert selection[1].label == "Q8_0"

    def test_remove_extra_dir(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.extra_dirs_list.addItems(["D:/a", "D:/b"])
        tab.extra_dirs_list.setCurrentRow(0)
        tab.extra_remove_button.click()
        assert tab._extra_dirs() == ("D:/b",)

    def test_models_dir_placeholder_is_in_app_default(
        self, qtbot, tab_controller
    ) -> None:
        from nlapt_gui.local_bridge import default_models_dir

        tab = make_tab(qtbot, tab_controller)
        assert tab.models_dir_edit.placeholderText() == str(default_models_dir())


class TestDownloadFlow:
    def test_download_disabled_without_selection(
        self, qtbot, tab_controller
    ) -> None:
        tab = make_tab(qtbot, tab_controller)
        assert not tab.download_button.isEnabled()

    def test_download_click_runs_to_finish(
        self, qtbot, tab_controller, tab_toasts, monkeypatch, tmp_path
    ) -> None:
        def fake_download(url, dest, *, expected_bytes=None, progress=None, cancel=None, **kw):  # noqa: ANN001, ANN003
            if progress is not None:
                progress(int(expected_bytes or 0), int(expected_bytes or 0))
            return dest

        monkeypatch.setattr("nlapt_gui.download_hub.download_file", fake_download)
        # The runtime auto-provision job must stay hermetic in tests.
        monkeypatch.setattr(
            "nlapt_gui.download_hub.ensure_runtime",
            lambda base_dir, **kw: tmp_path / "llama-server.exe",
        )
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        with qtbot.waitSignal(tab.bridge.download_finished, timeout=2000):
            tab.download_button.click()
        qtbot.waitUntil(lambda: not tab.progress.isVisible(), timeout=2000)
        assert any("已就绪" in text for text, _ in tab_toasts)
        assert tab.download_button.text() != "取消下载"

    def test_reopened_tab_reattaches_to_running_download(
        self, qtbot, tab_controller, monkeypatch
    ) -> None:
        """关闭再打开设置页,下载进度条与取消按钮必须接回来."""
        import threading

        from nlapt.core.errors import DownloadCancelledError

        from nlapt_gui.local_bridge import DOWNLOAD_CANCELLED

        started = threading.Event()

        def blocking_download(url, dest, *, expected_bytes=None, progress=None, cancel=None, **kw):  # noqa: ANN001, ANN003
            if progress is not None:
                progress(1, None)
            started.set()
            assert cancel.wait(timeout=5)
            raise DownloadCancelledError("下载已取消")

        monkeypatch.setattr("nlapt_gui.download_hub.download_file", blocking_download)
        starter = LocalBridge(manager=FakeManager())
        starter.update_settings(
            server_path="srv.exe",  # keep the runtime provision job out
            family_id="toriigate-0.5",
            quant_label="Q4_K_M",
        )
        assert starter.start_download("toriigate-0.5", "Q4_K_M")
        assert started.wait(timeout=2)
        # The dialog was closed and reopened: the NEW tab (fresh bridge)
        # re-attaches — live progress bar, button in 取消下载 mode.
        tab = make_tab(qtbot, tab_controller)
        assert tab.progress.isVisibleTo(tab)
        assert tab.download_button.text() == "取消下载"
        assert tab.download_button.isEnabled()
        with qtbot.waitSignal(tab.bridge.download_finished, timeout=2000) as blocker:
            tab.download_button.click()  # cancels the predecessor's task
        assert blocker.args[2] == DOWNLOAD_CANCELLED
        qtbot.waitUntil(lambda: not tab.progress.isVisibleTo(tab), timeout=2000)


class TestPromptPresets:
    """提示词预设 combo: shown for caption specialists, persisted round-trip."""

    def test_preset_row_shown_for_joycaption_with_official_entries(
        self, qtbot, tab_controller
    ) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, "joycaption-beta-one", "Q4_K"))
        assert tab.preset_combo.isVisibleTo(tab)
        data = [tab.preset_combo.itemData(i) for i in range(tab.preset_combo.count())]
        assert "Descriptive" in data
        assert "Danbooru tag list" in data
        assert "custom" in data  # 自定义 opt-out is always the last entry
        # Default = the family's first official preset, not 自定义.
        assert tab.preset_combo.currentData() == "Descriptive"

    def test_preset_row_populates_for_toriigate(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        assert tab.preset_combo.isVisibleTo(tab)
        data = [tab.preset_combo.itemData(i) for i in range(tab.preset_combo.count())]
        assert "long" in data
        assert "json" in data

    def test_preset_row_hidden_for_generic_models(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, "gemma4-12b", "Q4_K_M"))
        assert not tab.preset_combo.isVisibleTo(tab)

    def test_persist_and_prefill_round_trip(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, "joycaption-beta-one", "Q4_K"))
        index = tab.preset_combo.findData("Danbooru tag list")
        assert index >= 0
        tab.preset_combo.setCurrentIndex(index)
        tab.persist()
        stored = load_local_settings(app_data_dir() / "local_llm.json")
        assert stored.prompt_preset == "Danbooru tag list"
        # A reopened dialog restores the persisted choice for that family.
        reopened = make_tab(qtbot, tab_controller)
        assert reopened.preset_combo.currentData() == "Danbooru tag list"


class TestApplyProfile:
    def test_apply_writes_local_profile(
        self, qtbot, tab_controller, tab_toasts, monkeypatch
    ) -> None:
        tab = make_tab(qtbot, tab_controller)
        monkeypatch.setattr(tab.bridge, "is_downloaded", lambda family, quant: True)
        tab.tree.setCurrentItem(quant_item(tab, "joycaption-beta-one", "Q4_K"))
        tab.parallel_spin.setValue(4)
        tab.port_spin.setValue(2222)
        assert tab.apply_button.isEnabled()
        tab.apply_button.click()
        stored = load_app_config()
        assert stored.active_profile == "local"
        profile = stored.profiles[0]
        assert profile.name == "local"
        assert profile.api_type == "openai"
        assert profile.base_url == "http://127.0.0.1:2222/v1"
        assert profile.text_model == "joycaption-beta-one"
        assert profile.vision_model == "joycaption-beta-one"
        assert stored.request.concurrency == 4
        assert "127.0.0.1:2222" not in config_path().read_text(encoding="utf-8")
        assert any("已切换到本地模型" in text for text, _ in tab_toasts)

    def test_apply_text_only_family_leaves_vision_empty(
        self, qtbot, tab_controller, monkeypatch
    ) -> None:
        tab = make_tab(qtbot, tab_controller)
        monkeypatch.setattr(tab.bridge, "is_downloaded", lambda family, quant: True)
        tab.tree.setCurrentItem(
            quant_item(tab, "gemma4-26b-a4b-heretic", "i1-Q4_K_M")
        )
        tab.apply_button.click()
        stored = load_app_config()
        assert stored.profiles[0].vision_model == ""
        assert load_config(config_path()).profiles == ()

    def test_apply_with_unreadable_config_refuses(
        self, qtbot, tab_controller, tab_toasts, monkeypatch
    ) -> None:
        config_path().write_text("{broken json", encoding="utf-8")
        tab = make_tab(qtbot, tab_controller)
        monkeypatch.setattr(tab.bridge, "is_downloaded", lambda family, quant: True)
        tab.tree.setCurrentItem(quant_item(tab, "joycaption-beta-one", "Q4_K"))
        tab.apply_button.click()
        assert any("无法读取现有配置" in text for text, _ in tab_toasts)


class TestServerFlow:
    def test_server_requires_server_path_only_without_runtime(
        self, qtbot, tab_controller, tab_toasts, monkeypatch
    ) -> None:
        # Platforms with no pinned runtime still demand a manual pick.
        monkeypatch.setattr(
            "nlapt_gui.widgets.local_tab.runtime_supported", lambda: False
        )
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        tab.server_button.click()
        assert any("llama-server" in text for text, _ in tab_toasts)

    def test_server_with_runtime_skips_manual_path_guard(
        self, qtbot, tab_controller, tab_toasts, monkeypatch
    ) -> None:
        # 就绪即可用: with a pinned runtime the empty path is fine and the
        # next guard (download check) is the one that speaks.
        monkeypatch.setattr(
            "nlapt_gui.widgets.local_tab.runtime_supported", lambda: True
        )
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        tab.server_button.click()
        assert any("尚未下载" in text for text, _ in tab_toasts)
        assert not any("请先选择 llama-server" in text for text, _ in tab_toasts)

    def test_auto_placeholder_when_runtime_supported(
        self, qtbot, tab_controller
    ) -> None:
        from nlapt.local.runtime import current_asset

        from nlapt_gui.widgets.local_tab import SERVER_PATH_AUTO_PLACEHOLDER

        if current_asset() is None:
            pytest.skip("no pinned runtime for this platform")
        tab = make_tab(qtbot, tab_controller)
        assert tab.server_path_edit.placeholderText() == SERVER_PATH_AUTO_PLACEHOLDER

    def test_server_requires_download(
        self, qtbot, tab_controller, tab_toasts
    ) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.server_path_edit.setText("C:/llama/llama-server.exe")
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        tab.server_button.click()
        assert any("尚未下载" in text for text, _ in tab_toasts)

    def test_server_start_and_stop_roundtrip(
        self, qtbot, tab_controller, tab_toasts, monkeypatch
    ) -> None:
        manager = FakeManager()
        bridge = LocalBridge(manager=manager)
        tab = make_tab(qtbot, tab_controller, bridge=bridge)
        monkeypatch.setattr(bridge, "is_downloaded", lambda family, quant: True)
        tab.server_path_edit.setText("C:/llama/llama-server.exe")
        tab.gpu_layers_spin.setValue(9)  # 自动 (-1) would probe real hardware
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        tab.server_button.click()
        qtbot.waitUntil(lambda: "运行中" in tab.server_status.text(), timeout=2000)
        assert tab.server_button.text() == "停止服务"
        tab.server_button.click()
        qtbot.waitUntil(lambda: "未启动" in tab.server_status.text(), timeout=2000)
        assert manager.stop_calls == 1
        assert tab.server_button.text() == "启动本地服务"

    def test_server_failure_reports_error(
        self, qtbot, tab_controller, tab_toasts, monkeypatch
    ) -> None:
        bridge = LocalBridge(manager=FakeManager(fail=True))
        tab = make_tab(qtbot, tab_controller, bridge=bridge)
        monkeypatch.setattr(bridge, "is_downloaded", lambda family, quant: True)
        tab.server_path_edit.setText("C:/llama/llama-server.exe")
        tab.gpu_layers_spin.setValue(9)  # 自动 (-1) would probe real hardware
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        tab.server_button.click()
        qtbot.waitUntil(
            lambda: any("启动失败" in text for text, _ in tab_toasts), timeout=2000
        )
        assert tab.server_button.isEnabled()


class TestParallelAwareEstimates:
    """v1.7: KV memory scales with 上下文长度 × 并发 (per-slot semantics)."""

    def test_parallel_change_recomputes_verdicts(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab._on_hardware_ready(BIG_RIG)
        item = quant_item(tab, "toriigate-0.5", "Q4_K_M")
        before = item.text(3)
        tab.parallel_spin.setValue(16)  # 16x the KV budget
        assert item.text(3) != "未检测"
        assert before  # grade stays populated after the recompute

    def test_estimate_context_multiplies(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.context_spin.setValue(4096)
        tab.parallel_spin.setValue(4)
        assert tab._estimate_context() == 16384


class TestFlorenceTaskUI:
    """The Florence-only 指令模式 row + engine-aware button states."""

    FLOR = "florence2-promptgen-v2"

    def test_task_row_toggles_with_selection(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        assert not tab._form.isRowVisible(tab.florence_task_combo)
        tab.tree.setCurrentItem(quant_item(tab, self.FLOR, "ONNX"))
        assert tab._form.isRowVisible(tab.florence_task_combo)
        assert not tab.server_button.isEnabled()
        assert not tab.apply_button.isEnabled()
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        assert not tab._form.isRowVisible(tab.florence_task_combo)
        assert tab.server_button.isEnabled()

    def test_combo_lists_all_instructions(self, qtbot, tab_controller) -> None:
        from nlapt.local.florence import FLORENCE_TASK_TOKENS

        tab = make_tab(qtbot, tab_controller)
        tokens = [
            tab.florence_task_combo.itemData(i)
            for i in range(tab.florence_task_combo.count())
        ]
        assert tokens == list(FLORENCE_TASK_TOKENS)

    def test_task_persists_and_prefills(self, qtbot, tab_controller) -> None:
        from nlapt.local.florence import TASK_MIXED_CAPTION

        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, self.FLOR, "ONNX"))
        index = tab.florence_task_combo.findData(TASK_MIXED_CAPTION)
        tab.florence_task_combo.setCurrentIndex(index)
        tab.persist()
        stored = load_local_settings(app_data_dir() / "local_llm.json")
        assert stored.florence_task == TASK_MIXED_CAPTION
        fresh = make_tab(qtbot, tab_controller)
        assert fresh.florence_task_combo.currentData() == TASK_MIXED_CAPTION


class TestFlorenceLoraUI:
    """The Florence-only LoRA row: 不使用 sentinel + add / remove / persist."""

    FLOR = "florence2-promptgen-v2"

    def pick_file(self, monkeypatch, path) -> None:  # noqa: ANN001
        monkeypatch.setattr(
            "nlapt_gui.widgets.local_tab.QFileDialog.getOpenFileName",
            staticmethod(lambda *a, **k: (str(path), "")),
        )

    def test_lora_row_toggles_with_selection(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        assert not tab._form.isRowVisible(tab.florence_lora_holder)
        tab.tree.setCurrentItem(quant_item(tab, self.FLOR, "ONNX"))
        assert tab._form.isRowVisible(tab.florence_lora_holder)
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        assert not tab._form.isRowVisible(tab.florence_lora_holder)

    def test_add_select_persist_prefill(
        self, qtbot, tab_controller, tmp_path, monkeypatch
    ) -> None:
        lora = tmp_path / "bai-json.safetensors"
        lora.write_bytes(b"x")
        tab = make_tab(qtbot, tab_controller)
        assert tab.florence_lora_combo.currentData() == ""  # 不使用 sentinel
        self.pick_file(monkeypatch, lora)
        tab._add_lora()
        assert tab.florence_lora_combo.currentData() == str(lora)
        tab._add_lora()  # picking the same file again must not duplicate
        assert tab.florence_lora_combo.count() == 2
        tab.persist()
        stored = load_local_settings(app_data_dir() / "local_llm.json")
        assert stored.florence_lora == str(lora)
        assert stored.florence_loras == (str(lora),)
        fresh = make_tab(qtbot, tab_controller)
        assert fresh.florence_lora_combo.currentData() == str(lora)
        assert fresh.florence_lora_combo.currentText() == "bai-json"

    def test_cancelled_picker_changes_nothing(
        self, qtbot, tab_controller, monkeypatch
    ) -> None:
        tab = make_tab(qtbot, tab_controller)
        self.pick_file(monkeypatch, "")
        tab._add_lora()
        assert tab.florence_lora_combo.count() == 1

    def test_remove_returns_to_none(
        self, qtbot, tab_controller, tmp_path, monkeypatch
    ) -> None:
        lora = tmp_path / "style.safetensors"
        lora.write_bytes(b"x")
        tab = make_tab(qtbot, tab_controller)
        self.pick_file(monkeypatch, lora)
        tab._add_lora()
        tab._remove_lora()
        assert tab.florence_lora_combo.count() == 1
        assert tab.florence_lora_combo.currentData() == ""
        tab._remove_lora()  # the 不使用 sentinel itself is not removable
        assert tab.florence_lora_combo.count() == 1
        tab.persist()
        stored = load_local_settings(app_data_dir() / "local_llm.json")
        assert stored.florence_lora == ""
        assert stored.florence_loras == ()


class TestCuratedLoraUI:
    """内置 LoRA: listed per compatible family, downloadable, task-linked."""

    LARGE = "florence2-large-ft"

    def curated_index(self, tab) -> int:  # noqa: ANN001
        from nlapt_gui.widgets.local_tab import ROLE_LORA_ID

        combo = tab.florence_lora_combo
        for index in range(combo.count()):
            if combo.itemData(index, ROLE_LORA_ID):
                return index
        return -1

    def test_listed_only_for_compatible_families(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, self.LARGE, "ONNX"))
        assert self.curated_index(tab) > 0
        tab.tree.setCurrentItem(quant_item(tab, "florence2-promptgen-v2", "ONNX"))
        assert self.curated_index(tab) == -1

    def test_missing_curated_lora_offers_download(
        self, qtbot, tab_controller, tmp_path, monkeypatch
    ) -> None:
        from nlapt_gui.widgets.local_tab import LORA_MISSING_SUFFIX, ROLE_LORA_ID

        tab = make_tab(qtbot, tab_controller)
        # Hermeticity: never look at (or write near) the repo's real models/.
        tab.bridge.update_settings(models_dir=str(tmp_path))
        tab.tree.setCurrentItem(quant_item(tab, self.LARGE, "ONNX"))
        index = self.curated_index(tab)
        combo = tab.florence_lora_combo
        assert combo.itemData(index, ROLE_LORA_ID) == "bai-json-large"
        assert LORA_MISSING_SUFFIX in combo.itemText(index)
        assert tab.lora_download_button.isHidden()  # 不使用 selected
        started: list[str] = []
        monkeypatch.setattr(
            tab.bridge, "start_lora_download", lambda lora_id: started.append(lora_id) or True
        )
        combo.setCurrentIndex(index)
        assert not tab.lora_download_button.isHidden()
        tab._on_lora_download_clicked()
        assert started == ["bai-json-large"]

    def test_downloaded_curated_lora_needs_no_button(
        self, qtbot, tab_controller, tmp_path
    ) -> None:
        from nlapt.local.catalog import find_lora, lora_file_path

        from nlapt_gui.widgets.local_tab import LORA_MISSING_SUFFIX

        tab = make_tab(qtbot, tab_controller)
        # Hermeticity: fake files go under tmp, never the repo's models/.
        tab.bridge.update_settings(models_dir=str(tmp_path))
        entry = find_lora("bai-json-large")
        for file in entry.files:
            dest = lora_file_path(tab.bridge.models_dir(), entry, file)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"x" * file.size_bytes)
        tab.tree.setCurrentItem(quant_item(tab, self.LARGE, "ONNX"))
        index = self.curated_index(tab)
        combo = tab.florence_lora_combo
        assert LORA_MISSING_SUFFIX not in combo.itemText(index)
        combo.setCurrentIndex(index)
        assert tab.lora_download_button.isHidden()
        assert combo.currentData() == str(tab.bridge.lora_adapter_file(entry))

    def test_selecting_curated_lora_switches_task(
        self, qtbot, tab_controller
    ) -> None:
        from nlapt.local.florence import TASK_BAI_JSON

        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, self.LARGE, "ONNX"))
        assert tab.florence_task_combo.currentData() != TASK_BAI_JSON
        tab.florence_lora_combo.setCurrentIndex(self.curated_index(tab))
        assert tab.florence_task_combo.currentData() == TASK_BAI_JSON

    def test_curated_lora_not_removable(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, self.LARGE, "ONNX"))
        index = self.curated_index(tab)
        combo = tab.florence_lora_combo
        before = combo.count()
        combo.setCurrentIndex(index)
        tab._remove_lora()
        assert combo.count() == before

    def test_task_combo_filtered_per_family(self, qtbot, tab_controller) -> None:
        from nlapt.local.catalog import find_family

        tab = make_tab(qtbot, tab_controller)

        def tokens() -> tuple[str, ...]:
            combo = tab.florence_task_combo
            return tuple(combo.itemData(i) for i in range(combo.count()))

        tab.tree.setCurrentItem(quant_item(tab, self.LARGE, "ONNX"))
        assert tokens() == find_family(self.LARGE).florence_tasks
        tab.tree.setCurrentItem(quant_item(tab, "florence2-promptgen-v2", "ONNX"))
        assert tokens() == find_family("florence2-promptgen-v2").florence_tasks
        assert "<BAI_JSON>" not in tokens()

    def test_size_column_shows_total_of_all_files(
        self, qtbot, tab_controller
    ) -> None:
        from nlapt.local.catalog import find_family
        from nlapt.local.hardware import format_bytes

        tab = make_tab(qtbot, tab_controller)
        family = find_family("florence2-promptgen-v2")
        total = family.quants[0].size_bytes + sum(
            extra.size_bytes for extra in family.extra_files
        )
        assert (
            quant_item(tab, "florence2-promptgen-v2", "ONNX").text(1)
            == format_bytes(total)
        )

    def test_server_click_explains_no_server_needed(
        self, qtbot, tab_controller, tab_toasts
    ) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(
            quant_item(tab, "florence2-promptgen-v2", "ONNX")
        )
        tab._on_server_clicked()  # the button itself is disabled
        assert any("免启动服务" in text for text, _ in tab_toasts)
