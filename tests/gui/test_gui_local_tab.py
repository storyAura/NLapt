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
    def test_top_level_series_match_catalog(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        names = [
            tab.tree.topLevelItem(i).text(0)
            for i in range(tab.tree.topLevelItemCount())
        ]
        assert names == [series.name for series in all_series()]

    def test_family_and_quant_counts(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        for index, series in enumerate(all_series()):
            series_item = tab.tree.topLevelItem(index)
            families = families_for(series.series_id)
            assert series_item.childCount() == len(families)
            for family_index, family in enumerate(families):
                family_item = series_item.child(family_index)
                assert family_item.childCount() == len(family.quants)

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
        assert "CPU 16 核" in tab.hw_label.text()
        assert "RTX 5090" in tab.hw_label.text()

    def test_gpu_full_verdict_on_big_rig(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab._on_hardware_ready(BIG_RIG)
        item = quant_item(tab, "toriigate-0.5", "Q4_K_M")
        assert item.text(3) == "✓ 显存流畅"

    def test_cpu_only_verdict_without_gpu(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab._on_hardware_ready(NO_GPU_RIG)
        assert "未检测到 NVIDIA" in tab.hw_label.text()
        item = quant_item(tab, "toriigate-0.5", "Q4_K_M")
        assert item.text(3) == "▢ 仅内存(慢)"

    def test_not_runnable_on_tiny_rig(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab._on_hardware_ready(TINY_RIG)
        item = quant_item(tab, "gemma4-31b", "Q8_0")
        assert item.text(3) == "✗ 配置不足"
        assert "还差" in item.toolTip(3)

    def test_detection_failure_shows_unknown(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab._on_hardware_ready(UNDETECTED)
        assert tab.hw_label.text() == HW_UNKNOWN
        item = quant_item(tab, "toriigate-0.5", "Q4_K_M")
        assert item.text(3) == "?"

    def test_context_change_recomputes_verdicts(self, qtbot, tab_controller) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab._on_hardware_ready(BIG_RIG)
        tab.context_spin.setValue(32_768)
        item = quant_item(tab, "toriigate-0.5", "Q4_K_M")
        assert item.text(3) != "?"


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
        tab.persist()
        stored = load_local_settings(app_data_dir() / "local_llm.json")
        assert stored.context_length == 2048
        assert stored.parallel == 6
        assert stored.port == 2000
        assert stored.server_path == "C:/llama/llama-server.exe"
        assert stored.family_id == "joycaption-beta-one"
        assert stored.quant_label == "Q4_K"

    def test_prefill_restores_saved_selection(self, qtbot, tab_controller) -> None:
        save_local_settings(
            app_data_dir() / "local_llm.json",
            LocalSettings(
                port=3000, parallel=7, family_id="gemma4-12b", quant_label="Q8_0"
            ),
        )
        tab = make_tab(qtbot, tab_controller)
        assert tab.port_spin.value() == 3000
        assert tab.parallel_spin.value() == 7
        selection = tab.current_selection()
        assert selection is not None
        assert selection[0].family_id == "gemma4-12b"
        assert selection[1].label == "Q8_0"


class TestDownloadFlow:
    def test_download_disabled_without_selection(
        self, qtbot, tab_controller
    ) -> None:
        tab = make_tab(qtbot, tab_controller)
        assert not tab.download_button.isEnabled()

    def test_download_click_runs_to_finish(
        self, qtbot, tab_controller, tab_toasts, monkeypatch
    ) -> None:
        def fake_download(url, dest, *, expected_bytes=None, progress=None, cancel=None, **kw):  # noqa: ANN001, ANN003
            if progress is not None:
                progress(int(expected_bytes or 0), int(expected_bytes or 0))
            return dest

        monkeypatch.setattr("nlapt_gui.local_bridge.download_file", fake_download)
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        with qtbot.waitSignal(tab.bridge.download_finished, timeout=2000):
            tab.download_button.click()
        qtbot.waitUntil(lambda: not tab.progress.isVisible(), timeout=2000)
        assert any("已下载" in text for text, _ in tab_toasts)
        assert tab.download_button.text() != "取消下载"


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
        stored = load_config(config_path())
        assert stored.active_profile == "local"
        profile = stored.profiles[0]
        assert profile.name == "local"
        assert profile.api_type == "openai"
        assert profile.base_url == "http://127.0.0.1:2222/v1"
        assert profile.text_model == "joycaption-beta-one"
        assert profile.vision_model == "joycaption-beta-one"
        assert stored.request.concurrency == 4
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
        stored = load_config(config_path())
        assert stored.profiles[0].vision_model == ""

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
    def test_server_requires_server_path(
        self, qtbot, tab_controller, tab_toasts
    ) -> None:
        tab = make_tab(qtbot, tab_controller)
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        tab.server_button.click()
        assert any("llama-server" in text for text, _ in tab_toasts)

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
        tab.tree.setCurrentItem(quant_item(tab, "toriigate-0.5", "Q4_K_M"))
        tab.server_button.click()
        qtbot.waitUntil(
            lambda: any("启动失败" in text for text, _ in tab_toasts), timeout=2000
        )
        assert tab.server_button.isEnabled()
