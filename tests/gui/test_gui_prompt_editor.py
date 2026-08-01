"""Tests for nlapt_gui.widgets.prompt_editor (设置 ▸ 提示词 tab)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import nlapt_gui.widgets.prompt_editor as module
from nlapt_gui.prompt_store import (
    DEFAULT_PROMPT_NAME,
    VisionPrompts,
    load_vision_prompts,
)
from nlapt_gui.widgets.prompt_editor import PromptsTab


@pytest.fixture()
def toasts_list() -> list[tuple[str, str]]:
    return []


def make_tab(
    qtbot, prompts: VisionPrompts | None = None, sink: list | None = None
) -> PromptsTab:
    tab = PromptsTab(prompts if prompts is not None else VisionPrompts())
    qtbot.addWidget(tab)
    if sink is not None:
        tab.toast_requested.connect(lambda text, kind: sink.append((text, kind)))
    return tab


def _patch_input(monkeypatch, name: str, ok: bool = True) -> None:
    monkeypatch.setattr(
        module.QInputDialog,
        "getText",
        staticmethod(lambda *a, **k: (name, ok)),
    )


def _patch_save_dialog(monkeypatch, path: Path | None) -> None:
    monkeypatch.setattr(
        module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(path) if path else "", "")),
    )


class TestDefaultTemplate:
    def test_default_is_empty_and_readonly(self, qtbot) -> None:
        tab = make_tab(qtbot)
        assert tab.template_combo.currentText() == DEFAULT_PROMPT_NAME
        assert tab.system_edit.toPlainText() == ""
        assert tab.system_edit.isReadOnly()
        assert not tab.save_template_button.isEnabled()
        assert not tab.delete_button.isEnabled()
        assert tab.hint.isVisibleTo(tab)

    def test_current_prompts_ignores_edits_on_default(self, qtbot) -> None:
        tab = make_tab(qtbot)
        tab.user_edit.setPlainText("user text")
        snapshot = tab.current_prompts()
        assert snapshot.active == DEFAULT_PROMPT_NAME
        assert snapshot.system_text() == ""
        assert snapshot.user_prompt == "user text"


class TestTemplateLifecycle:
    def test_new_template_forks_editor_text(self, qtbot, monkeypatch, toasts_list) -> None:
        tab = make_tab(qtbot, sink=toasts_list)
        _patch_input(monkeypatch, "动漫")
        tab.new_button.click()
        assert tab.template_combo.currentText() == "动漫"
        assert not tab.system_edit.isReadOnly()
        stored = load_vision_prompts()
        assert stored.active == "动漫"
        assert "动漫" in stored.prompts
        assert any("已保存模板" in text for text, _ in toasts_list)

    def test_new_rejects_duplicate_name(self, qtbot, monkeypatch, toasts_list) -> None:
        tab = make_tab(
            qtbot, VisionPrompts(prompts={"动漫": "x"}), sink=toasts_list
        )
        _patch_input(monkeypatch, "动漫")
        tab.new_button.click()
        assert any("已存在" in text for text, _ in toasts_list)

    def test_new_cancelled_is_noop(self, qtbot, monkeypatch) -> None:
        tab = make_tab(qtbot)
        _patch_input(monkeypatch, "", ok=False)
        tab.new_button.click()
        assert tab.template_combo.currentText() == DEFAULT_PROMPT_NAME

    def test_save_template_persists_editor_text(self, qtbot, monkeypatch) -> None:
        tab = make_tab(qtbot, VisionPrompts(active="动漫", prompts={"动漫": "old"}))
        tab.system_edit.setPlainText("new system prompt")
        tab.user_edit.setPlainText("new user prompt")
        tab.save_template_button.click()
        stored = load_vision_prompts()
        assert stored.prompts["动漫"] == "new system prompt"
        assert stored.user_prompt == "new user prompt"

    def test_delete_returns_to_default(self, qtbot, toasts_list) -> None:
        tab = make_tab(
            qtbot,
            VisionPrompts(active="动漫", prompts={"动漫": "x", "写实": "y"}),
            sink=toasts_list,
        )
        tab.delete_button.click()
        assert tab.template_combo.currentText() == DEFAULT_PROMPT_NAME
        stored = load_vision_prompts()
        assert "动漫" not in stored.prompts
        assert "写实" in stored.prompts
        assert any("已删除模板" in text for text, _ in toasts_list)

    def test_switching_templates_swaps_editor(self, qtbot) -> None:
        tab = make_tab(
            qtbot, VisionPrompts(active="动漫", prompts={"动漫": "a", "写实": "b"})
        )
        tab.template_combo.setCurrentIndex(tab.template_combo.findText("写实"))
        assert tab.system_edit.toPlainText() == "b"
        tab.template_combo.setCurrentIndex(
            tab.template_combo.findText(DEFAULT_PROMPT_NAME)
        )
        assert tab.system_edit.toPlainText() == ""


class TestExport:
    def test_export_current_writes_txt(
        self, qtbot, tmp_path: Path, monkeypatch, toasts_list
    ) -> None:
        tab = make_tab(
            qtbot,
            VisionPrompts(active="动漫", prompts={"动漫": "sys-text"}),
            sink=toasts_list,
        )
        target = tmp_path / "out.txt"
        _patch_save_dialog(monkeypatch, target)
        tab.export_current_button.click()
        assert target.read_text(encoding="utf-8") == "sys-text"
        assert any("已导出" in text for text, _ in toasts_list)

    def test_export_current_cancelled_writes_nothing(
        self, qtbot, tmp_path: Path, monkeypatch
    ) -> None:
        tab = make_tab(qtbot, VisionPrompts(active="动漫", prompts={"动漫": "x"}))
        _patch_save_dialog(monkeypatch, None)
        tab.export_current_button.click()
        assert list(tmp_path.iterdir()) == []

    def test_export_all_writes_json(
        self, qtbot, tmp_path: Path, monkeypatch, toasts_list
    ) -> None:
        tab = make_tab(
            qtbot,
            VisionPrompts(prompts={"动漫": "a", "写实": "b"}),
            sink=toasts_list,
        )
        target = tmp_path / "all.json"
        _patch_save_dialog(monkeypatch, target)
        tab.export_all_button.click()
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload == {"动漫": "a", "写实": "b"}

    def test_export_all_without_custom_warns(
        self, qtbot, tmp_path: Path, monkeypatch, toasts_list
    ) -> None:
        tab = make_tab(qtbot, sink=toasts_list)
        _patch_save_dialog(monkeypatch, tmp_path / "never.json")
        tab.export_all_button.click()
        assert any("没有自定义模板" in text for text, _ in toasts_list)
        assert not (tmp_path / "never.json").exists()

    def test_export_io_error_toasts(
        self, qtbot, tmp_path: Path, monkeypatch, toasts_list
    ) -> None:
        tab = make_tab(
            qtbot, VisionPrompts(active="动漫", prompts={"动漫": "x"}), sink=toasts_list
        )
        # A directory as the target file forces the OSError path.
        target_dir = tmp_path / "as_dir"
        target_dir.mkdir()
        _patch_save_dialog(monkeypatch, target_dir)
        tab.export_current_button.click()
        assert any("导出失败" in text for text, _ in toasts_list)


class TestCurrentPrompts:
    def test_captures_unsaved_custom_edits(self, qtbot) -> None:
        tab = make_tab(qtbot, VisionPrompts(active="动漫", prompts={"动漫": "old"}))
        tab.system_edit.setPlainText("edited")
        snapshot = tab.current_prompts()
        assert snapshot.prompts["动漫"] == "edited"
        assert snapshot.active == "动漫"


class TestLocalPromptSection:
    def test_unified_by_default_hides_local_editors(self, qtbot) -> None:
        tab = make_tab(qtbot)
        assert tab.local_unified_box.isChecked()
        assert not tab.local_system_edit.isVisibleTo(tab)
        assert not tab.local_user_edit.isVisibleTo(tab)

    def test_unchecking_reveals_and_captures_local_prompts(self, qtbot) -> None:
        tab = make_tab(qtbot)
        tab.local_unified_box.setChecked(False)
        assert tab.local_system_edit.isVisibleTo(tab)
        tab.local_system_edit.setPlainText("loc-sys")
        tab.local_user_edit.setPlainText("loc-user")
        snapshot = tab.current_prompts()
        assert snapshot.local_unified is False
        assert snapshot.local_system == "loc-sys"
        assert snapshot.local_user_prompt == "loc-user"

    def test_prefill_restores_local_fields(self, qtbot) -> None:
        prompts = VisionPrompts(
            local_unified=False, local_system="a", local_user_prompt="b"
        )
        tab = make_tab(qtbot, prompts)
        assert not tab.local_unified_box.isChecked()
        assert tab.local_system_edit.toPlainText() == "a"
        assert tab.local_user_edit.toPlainText() == "b"
        assert tab.local_system_edit.isVisibleTo(tab)
