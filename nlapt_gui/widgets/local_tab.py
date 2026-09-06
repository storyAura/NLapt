"""设置 ▸ 本地推理 tab: catalog tree, run grades, downloads, server.

Landscape layout: catalog tree on the left, carded controls on the right.
Per-quant「能否运行」is a five-level Chinese grade from
:class:`nlapt.local.advisor.RunGrade`. Slow work goes through
:class:`nlapt_gui.local_bridge.LocalBridge`. This tab may write the core
config (设为当前模型 registers a ``local`` OpenAI-compatible profile).
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QUrl
from PySide6.QtGui import QDesktopServices, QShowEvent
from PySide6.QtWidgets import QFileDialog, QTreeWidgetItem, QWidget

from nlapt.core.config import LLMProfile
from nlapt.core.errors import NLaptError, StorageError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.local.advisor import RunAssessment, RunGrade, assess, estimate_memory
from nlapt.local.catalog import (
    CATALOG_SNAPSHOT_DATE,
    ENGINE_FLORENCE,
    ModelFamily,
    QuantFile,
    find_family,
    find_lora,
    find_quant,
    repo_page_url,
)
from nlapt.local.hardware import HardwareInfo, format_bytes
from nlapt.local.presets import PRESET_CUSTOM, presets_for
from nlapt.local.runtime import LLAMA_CPP_TAG
from nlapt.local.server import base_url as local_base_url

from nlapt_gui.api_config import load_app_config, save_app_config
from nlapt_gui.controller import AppController, TOAST_ERR, TOAST_OK, TOAST_WARN
from nlapt_gui.local_bridge import (
    DOWNLOAD_CANCELLED,
    DOWNLOAD_OK,
    LORA_FAMILY_PREFIX,
    SERVER_ERROR,
    SERVER_RUNNING,
    SERVER_STARTING,
    LocalBridge,
    default_models_dir,
    runtime_supported,
)
_LOGGER = get_logger(__name__)

LOCAL_PROFILE_NAME = "local"
LOCAL_API_TYPE = "openai"

ROLE_FAMILY = Qt.ItemDataRole.UserRole
ROLE_QUANT = Qt.ItemDataRole.UserRole + 1
# LoRA combo: item data = safetensors path; this role = curated lora_id.
ROLE_LORA_ID = Qt.ItemDataRole.UserRole + 2

# Landscape sizing: tree left, controls right.
LOCAL_TAB_MIN_W = 840
LOCAL_TAB_MIN_H = 520
TREE_MIN_W = 430
RIGHT_PANE_MIN_W = 330
EXTRA_DIRS_LIST_H = 64
CONTEXT_STEP = 512
PROGRESS_BAR_MAX = 1000

# UI strings.
HW_DETECTING = "正在检测本机硬件…"
HW_UNKNOWN = "硬件检测失败 — 「能否运行」列将显示「未检测」"
HW_NO_GPU = "未检测到 NVIDIA 独显(将以 CPU 推理)"
HW_LINE = (
    "本机硬件:CPU <b>{cores}</b> 核 · 内存 <b>{ram}</b>(可用 {avail})· {gpu}"
)
HW_GPU = "{name} · 显存 <b>{vram}</b>(空闲 {free})"
BTN_DETECT = "重新检测"
COL_MODEL = "模型"
COL_SIZE = "体积"
COL_DOWNLOADS = "热度"
COL_GRADE = "能否运行"
TAG_VISION = "视觉"
TAG_RECOMMENDED = "★ 推荐"
CATALOG_HINT = (
    f"目录快照 {CATALOG_SNAPSHOT_DATE} · 热度为 HuggingFace 月下载量 · "
    "「能否运行」依据本机硬件与右侧 上下文长度 × 并发请求数 估算"
)
DETAIL_EMPTY = "在左侧展开一个系列,选择具体量化档查看评估与操作"
DETAIL_LINE = (
    "<b>{grade}</b> — {sentence}<br>"
    "预计占用 <b>{total}</b>(权重 {weights}{mmproj} + 上下文 {kv} + 开销 {overhead})<br>"
    "显存预算 {gpu} · 内存预算 {ram}"
)
DETAIL_MMPROJ = " + 视觉 {size}"
# 五级中文评级 (用户要求: 直接说能不能跑得动).
GRADE_TEXT: dict[RunGrade, str] = {
    RunGrade.PERFECT: "轻松运行",
    RunGrade.SMOOTH: "流畅运行",
    RunGrade.OK: "可以运行",
    RunGrade.BARELY: "勉强能跑",
    RunGrade.NO: "跑不动",
    RunGrade.UNKNOWN: "未检测",
}
GRADE_SENTENCE: dict[RunGrade, str] = {
    RunGrade.PERFECT: "显存余量充足,可完全载入显存,速度很快",
    RunGrade.SMOOTH: "可完全载入显存,预计流畅运行",
    RunGrade.OK: "需要内存参与(显存不足或无独显),中等速度,可正常使用",
    RunGrade.BARELY: "余量很小,勉强能跑,速度较慢且可能不稳定",
    RunGrade.NO: "超出本机显存 + 内存承受范围,跑不动(还差 {shortfall})",
    RunGrade.UNKNOWN: "硬件信息未知,无法判断",
}
BTN_DOWNLOAD = "下载模型"
BTN_DOWNLOAD_AGAIN = "已就绪 ✓(重新校验)"
BTN_CANCEL_DOWNLOAD = "取消下载"
BTN_PAGE = "打开模型页"
BTN_SERVER_START = "启动本地服务"
BTN_SERVER_STOP = "停止服务"
BTN_APPLY = "设为当前模型"
SERVER_STATUS_STOPPED = "服务未启动"
SERVER_STATUS_STARTING = "正在启动服务(首次加载较慢)…"
SERVER_STATUS_RUNNING = "服务运行中:{url}"
LABEL_MODELS_DIR = "下载目录"
LABEL_SERVER_PATH = "llama-server"
SERVER_PATH_AUTO_PLACEHOLDER = f"自动(内置 llama.cpp {LLAMA_CPP_TAG},首次使用自动下载)"
BTN_BROWSE = "浏览…"
LABEL_CONTEXT = "上下文长度"
LABEL_GPU_LAYERS = "GPU 层数"
LABEL_THREADS = "线程数"
LABEL_PARALLEL = "并发请求数"
LABEL_PORT = "端口"
SPECIAL_GPU_AUTO = "自动(全部)"
SPECIAL_THREADS_AUTO = "自动"
LABEL_FLORENCE_TASK = "指令模式"
TIP_FLORENCE_TASK = (
    "Florence-2 PromptGen 由内置指令驱动(不使用推理提示词):"
    "选择输出标签、各级标题或构图分析"
)
LABEL_FLORENCE_LORA = "LoRA"
LORA_NONE_LABEL = "不使用 LoRA"
TIP_FLORENCE_LORA = (
    "推理时把 PEFT LoRA(safetensors,同目录需有 adapter_config.json)"
    "合并进模型;仅支持与当前模型同架构训练的 LoRA"
)
CAPTION_PICK_LORA = "选择 LoRA 文件(同目录需有 adapter_config.json)"
FILTER_LORA = "LoRA 权重 (*.safetensors);;所有文件 (*)"
BTN_DOWNLOAD_LORA = "下载"
LORA_MISSING_SUFFIX = "(未下载)"
TOAST_LORA_OK = "已就绪 LoRA:{name}"
LABEL_PROMPT_PRESET = "提示词预设"
PRESET_CUSTOM_LABEL = "自定义(使用推理提示词)"
TIP_PROMPT_PRESET = (
    "该模型按官方固定指令训练,使用预设可获得稳定输出;"
    "选择「自定义」则沿用 设置 ▸ 提示词 里的推理提示词"
)
TIP_FLORENCE_NO_SERVER = "该模型免启动服务 — 推理时自动加载,无需 llama-server"
TIP_FLORENCE_NO_APPLY = "该模型仅用于图片打标,不能作为翻译 / 重写的当前模型"
LABEL_EXTRA_DIRS = "复用目录(也在这些目录中查找已下载的模型)"
BTN_ADD_DIR = "添加"
BTN_REMOVE_DIR = "移除"
CAPTION_PICK_EXTRA_DIR = "选择复用模型目录"
TOAST_SELECT_QUANT = "请先在列表中选择一个量化档"
TOAST_DOWNLOAD_BUSY = "已有下载任务正在进行"
TOAST_DOWNLOAD_OK = "已就绪 {name} · {quant}"
TOAST_DOWNLOAD_CANCELLED = "已取消下载(已下载部分保留,可续传)"
TOAST_DOWNLOAD_FAIL = "下载失败: {message}"
TOAST_NEED_SERVER_PATH = "请先选择 llama-server 可执行文件"
TOAST_NEED_DOWNLOAD = "该量化档尚未下载完成"
TOAST_SERVER_RUNNING = "本地服务已就绪: {url}"
TOAST_SERVER_FAIL = "本地服务启动失败: {message}"
TOAST_SERVER_STOPPED = "本地服务已停止"
TOAST_APPLIED = "已切换到本地模型 {name}(并发 {parallel})"
TOAST_CONFIG_UNREADABLE = "无法读取现有配置,已取消写入以避免覆盖其它设置"
TOAST_SETTINGS_FAILED = "本地推理设置保存失败: {message}"
DOWNLOAD_FORMAT = "{done} / {total}"

FILTER_EXECUTABLE = "可执行文件 (*.exe);;所有文件 (*)"

from nlapt_gui.widgets.local_tab_sections import (  # noqa: E402
    build_local_tab_ui,
    populate_catalog_tree,
    quant_items,
    rebuild_lora_combo,
    rebuild_task_combo,
    refresh_lora_buttons,
)


class LocalTab(QWidget):
    """The 本地推理 tab (landscape layout, five-level run grades)."""

    def __init__(
        self,
        controller: AppController,
        *,
        bridge: LocalBridge | None = None,
        pool: QThreadPool | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self.bridge = (
            bridge if bridge is not None else LocalBridge(pool=pool, parent=self)
        )
        # Last HardwareInfo received via hardware_ready (single UI-side source).
        self._hardware: HardwareInfo | None = self.bridge.hardware
        self._detect_started = False
        self._build_ui()
        self._populate_tree()
        self._prefill_from_settings()
        self._connect()
        self._refresh_selection_ui()

    # -- construction ------------------------------------------------------------------
    def _build_ui(self) -> None:
        build_local_tab_ui(self)

    def _populate_tree(self) -> None:
        populate_catalog_tree(self)

    def _connect(self) -> None:
        self.detect_button.clicked.connect(self.refresh_hardware)
        self.tree.currentItemChanged.connect(self._on_selection_changed)
        self.context_spin.valueChanged.connect(self._refresh_verdicts)
        # KV memory scales with ctx × parallel (per-slot context semantics).
        self.parallel_spin.valueChanged.connect(self._refresh_verdicts)
        self.download_button.clicked.connect(self._on_download_clicked)
        self.page_button.clicked.connect(self._on_page_clicked)
        self.server_button.clicked.connect(self._on_server_clicked)
        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.models_dir_browse.clicked.connect(self._browse_models_dir)
        self.server_path_browse.clicked.connect(self._browse_server_path)
        self.extra_add_button.clicked.connect(self._add_extra_dir)
        self.extra_remove_button.clicked.connect(self._remove_extra_dir)
        self.lora_add_button.clicked.connect(self._add_lora)
        self.lora_remove_button.clicked.connect(self._remove_lora)
        self.lora_download_button.clicked.connect(self._on_lora_download_clicked)
        self.florence_lora_combo.currentIndexChanged.connect(self._on_lora_changed)
        bridge = self.bridge
        bridge.hardware_ready.connect(self._on_hardware_ready)
        bridge.download_progress.connect(self._on_download_progress)
        bridge.download_finished.connect(self._on_download_finished)
        bridge.server_changed.connect(self._on_server_changed)

    def _prefill_from_settings(self) -> None:
        settings = self.bridge.settings
        self.models_dir_edit.setText(settings.models_dir)
        self.models_dir_edit.setPlaceholderText(str(default_models_dir()))
        self.server_path_edit.setText(settings.server_path)
        if runtime_supported():
            self.server_path_edit.setPlaceholderText(SERVER_PATH_AUTO_PLACEHOLDER)
        self.extra_dirs_list.addItems(list(settings.extra_dirs))
        task_index = self.florence_task_combo.findData(settings.florence_task)
        if task_index >= 0:
            self.florence_task_combo.setCurrentIndex(task_index)
        # LoRAs: user-registered files (persisted) + the current pick. A
        # selected path outside the list (e.g. a curated LoRA) stays
        # selectable as a plain entry until a family rebuild re-labels it.
        self._user_loras = list(settings.florence_loras)
        self._rebuild_lora_combo(None)
        selected_lora = settings.florence_lora
        lora_index = self.florence_lora_combo.findData(selected_lora)
        if selected_lora and lora_index < 0:
            self.florence_lora_combo.addItem(Path(selected_lora).stem, selected_lora)
            lora_index = self.florence_lora_combo.count() - 1
        self.florence_lora_combo.setCurrentIndex(max(lora_index, 0))
        self.context_spin.setValue(settings.context_length)
        self.gpu_layers_spin.setValue(settings.gpu_layers)
        self.threads_spin.setValue(settings.threads)
        self.parallel_spin.setValue(settings.parallel)
        self.port_spin.setValue(settings.port)
        if settings.family_id and settings.quant_label:
            self._select_quant_item(settings.family_id, settings.quant_label)
        if self.bridge.server_running():
            # Reflect an already-running server WITHOUT the "just started" toast.
            self.server_button.setText(BTN_SERVER_STOP)
            self.server_status.setText(
                SERVER_STATUS_RUNNING.format(url=self.bridge.server_base_url())
            )
        active = self.bridge.active_download()
        if active is not None:
            # Re-attach to a download started before this tab existed: show
            # the live progress bar at its current position; the hub keeps
            # feeding _on_download_progress, and _refresh_selection_ui turns
            # the button into 取消下载 via is_downloading().
            family_id, quant_label, done, total = active
            self._select_quant_item(family_id, quant_label)
            self.progress.setVisible(True)
            self._on_download_progress(family_id, quant_label, done, total)

    def _select_quant_item(self, family_id: str, quant_label: str) -> None:
        for item in self._quant_items():
            if (
                item.data(0, ROLE_FAMILY) == family_id
                and item.data(0, ROLE_QUANT) == quant_label
            ):
                parent = item.parent()
                while parent is not None:
                    parent.setExpanded(True)
                    parent = parent.parent()
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)
                return

    # -- lazy hardware detection -------------------------------------------------------
    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        if not self._detect_started:
            self.refresh_hardware()

    def refresh_hardware(self) -> None:
        """(Re-)probe the machine asynchronously."""
        self._detect_started = True
        self.hw_label.setText(HW_DETECTING)
        self.bridge.detect()

    # -- persistence -------------------------------------------------------------------
    def persist(self) -> None:
        """Write the form + selection into the local settings file."""
        selection = self.current_selection()
        try:
            self.bridge.update_settings(
                models_dir=self.models_dir_edit.text().strip(),
                extra_dirs=self._extra_dirs(),
                server_path=self.server_path_edit.text().strip(),
                context_length=self.context_spin.value(),
                gpu_layers=self.gpu_layers_spin.value(),
                threads=self.threads_spin.value(),
                parallel=self.parallel_spin.value(),
                port=self.port_spin.value(),
                family_id=selection[0].family_id if selection else "",
                quant_label=selection[1].label if selection else "",
                florence_task=self.florence_task_combo.currentData(),
                florence_lora=self.florence_lora_combo.currentData() or "",
                florence_loras=self._florence_loras(),
                prompt_preset=self._current_preset_choice(),
            )
        except NLaptError as exc:
            _LOGGER.exception("could not persist local settings")
            self._toast(TOAST_SETTINGS_FAILED.format(message=exc.message), TOAST_ERR)

    def _extra_dirs(self) -> tuple[str, ...]:
        return tuple(
            self.extra_dirs_list.item(i).text()
            for i in range(self.extra_dirs_list.count())
        )

    def _florence_loras(self) -> tuple[str, ...]:
        """User-registered LoRA paths (curated entries live in the catalog)."""
        return tuple(self._user_loras)

    def _current_family(self) -> ModelFamily | None:
        selection = self.current_selection()
        return selection[0] if selection is not None else None

    def _add_lora(self) -> None:
        chosen, _selected_filter = QFileDialog.getOpenFileName(
            self, CAPTION_PICK_LORA, "", FILTER_LORA
        )
        if not chosen:
            return
        if chosen not in self._user_loras:
            self._user_loras.append(chosen)
        self._rebuild_lora_combo(self._current_family())
        index = self.florence_lora_combo.findData(chosen)
        self.florence_lora_combo.setCurrentIndex(max(index, 0))

    def _remove_lora(self) -> None:
        """Drop the selected USER LoRA (curated entries are not removable)."""
        combo = self.florence_lora_combo
        index = combo.currentIndex()
        if index <= 0 or combo.itemData(index, ROLE_LORA_ID):
            return
        path = combo.itemData(index)
        if path in self._user_loras:
            self._user_loras.remove(path)
        combo.removeItem(index)

    def _rebuild_lora_combo(self, family: ModelFamily | None) -> None:
        rebuild_lora_combo(self, family)

    def _rebuild_task_combo(self, family: ModelFamily) -> None:
        rebuild_task_combo(self, family)

    def _refresh_lora_buttons(self) -> None:
        refresh_lora_buttons(self)

    def _on_lora_changed(self, _index: int) -> None:
        """Curated pick: surface the download button + jump to its 指令."""
        self._refresh_lora_buttons()
        combo = self.florence_lora_combo
        lora_id = combo.itemData(combo.currentIndex(), ROLE_LORA_ID)
        if not lora_id:
            return
        entry = find_lora(str(lora_id))
        task_index = self.florence_task_combo.findData(entry.task)
        if entry.task and task_index >= 0:
            self.florence_task_combo.setCurrentIndex(task_index)

    def _on_lora_download_clicked(self) -> None:
        combo = self.florence_lora_combo
        lora_id = combo.itemData(combo.currentIndex(), ROLE_LORA_ID)
        if not lora_id:
            return
        if not self.bridge.start_lora_download(str(lora_id)):
            self._toast(TOAST_DOWNLOAD_BUSY, TOAST_WARN)
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._refresh_lora_buttons()

    # -- selection ---------------------------------------------------------------------
    def current_selection(self) -> tuple[ModelFamily, QuantFile] | None:
        """The selected (family, quant), or None when no quant row is current."""
        item = self.tree.currentItem()
        if item is None:
            return None
        family_id = item.data(0, ROLE_FAMILY)
        quant_label = item.data(0, ROLE_QUANT)
        if not family_id or not quant_label:
            return None
        family = find_family(str(family_id))
        return family, find_quant(family, str(quant_label))

    def _quant_items(self) -> tuple[QTreeWidgetItem, ...]:
        return quant_items(self)

    # -- grades ------------------------------------------------------------------------
    def _estimate_context(self) -> int:
        """Context the server will actually allocate: per-slot ctx × slots."""
        return self.context_spin.value() * max(1, self.parallel_spin.value())

    def _refresh_verdicts(self) -> None:
        hardware = self._hardware
        context = self._estimate_context()
        for item in self._quant_items():
            family = find_family(str(item.data(0, ROLE_FAMILY)))
            quant = find_quant(family, str(item.data(0, ROLE_QUANT)))
            if hardware is None:
                item.setText(3, GRADE_TEXT[RunGrade.UNKNOWN])
                continue
            result = assess(family, quant, hardware, context_length=context)
            item.setText(3, GRADE_TEXT[result.grade])
            item.setToolTip(3, self._grade_sentence(result))
        self._refresh_selection_ui()

    @staticmethod
    def _grade_sentence(result: RunAssessment) -> str:
        sentence = GRADE_SENTENCE[result.grade]
        if result.grade is RunGrade.NO:
            sentence = sentence.format(shortfall=format_bytes(result.shortfall_bytes))
        return sentence

    def _on_hardware_ready(self, info: object) -> None:
        if not isinstance(info, HardwareInfo):
            return
        self._hardware = info
        if info.ram_total_bytes <= 0:
            self.hw_label.setText(HW_UNKNOWN)
        else:
            gpu = info.best_gpu()
            gpu_text = (
                HW_NO_GPU
                if gpu is None
                else HW_GPU.format(
                    name=gpu.name,
                    vram=format_bytes(gpu.vram_total_bytes),
                    free=format_bytes(gpu.vram_free_bytes),
                )
            )
            self.hw_label.setText(
                HW_LINE.format(
                    cores=info.cpu_cores,
                    ram=format_bytes(info.ram_total_bytes),
                    avail=format_bytes(info.ram_available_bytes),
                    gpu=gpu_text,
                )
            )
        self._refresh_verdicts()

    # -- selection-driven UI -----------------------------------------------------------
    def _on_selection_changed(self, *_args: object) -> None:
        self._refresh_selection_ui()

    def _refresh_selection_ui(self) -> None:
        selection = self.current_selection()
        downloading = self.bridge.is_downloading()
        if selection is None:
            self.detail_label.setText(DETAIL_EMPTY)
            self.download_button.setText(
                BTN_CANCEL_DOWNLOAD if downloading else BTN_DOWNLOAD
            )
            self.download_button.setEnabled(downloading)
            self.page_button.setEnabled(False)
            self.server_button.setEnabled(self.bridge.server_running())
            self.apply_button.setEnabled(False)
            self._form.setRowVisible(self.florence_task_combo, False)
            self._form.setRowVisible(self.florence_lora_holder, False)
            self._form.setRowVisible(self.preset_combo, False)
            return
        family, quant = selection
        hardware = self._hardware
        context = self._estimate_context()
        estimate = estimate_memory(family, quant, context_length=context)
        mmproj_part = (
            DETAIL_MMPROJ.format(size=format_bytes(estimate.mmproj_bytes))
            if estimate.mmproj_bytes
            else ""
        )
        if hardware is not None:
            result = assess(family, quant, hardware, context_length=context)
            grade_word = GRADE_TEXT[result.grade]
            sentence = self._grade_sentence(result)
            budgets = (
                format_bytes(result.gpu_budget_bytes),
                format_bytes(result.ram_budget_bytes),
            )
        else:
            grade_word = GRADE_TEXT[RunGrade.UNKNOWN]
            sentence = GRADE_SENTENCE[RunGrade.UNKNOWN]
            budgets = ("?", "?")
        self.detail_label.setText(
            DETAIL_LINE.format(
                grade=grade_word,
                sentence=sentence,
                total=format_bytes(estimate.total_bytes),
                weights=format_bytes(estimate.weights_bytes),
                mmproj=mmproj_part,
                kv=format_bytes(estimate.kv_cache_bytes),
                overhead=format_bytes(estimate.overhead_bytes),
                gpu=budgets[0],
                ram=budgets[1],
            )
        )
        downloaded = self.bridge.is_downloaded(family, quant)
        if downloading:
            self.download_button.setText(BTN_CANCEL_DOWNLOAD)
            self.download_button.setEnabled(True)
        else:
            self.download_button.setText(
                BTN_DOWNLOAD_AGAIN if downloaded else BTN_DOWNLOAD
            )
            self.download_button.setEnabled(True)
        self.page_button.setEnabled(True)
        is_florence = family.engine == ENGINE_FLORENCE
        self._form.setRowVisible(self.florence_task_combo, is_florence)
        self._form.setRowVisible(self.florence_lora_holder, is_florence)
        if is_florence:
            self._rebuild_task_combo(family)
            self._rebuild_lora_combo(family)
        has_presets = bool(presets_for(family.family_id))
        self._form.setRowVisible(self.preset_combo, has_presets)
        if has_presets:
            self._populate_presets(family)
        if is_florence:
            # Florence loads in-process on demand: the server button only
            # remains usable to STOP an already-running llama server, and
            # 设为当前模型 (a text-LLM concern) does not apply.
            self.server_button.setEnabled(self.bridge.server_running())
            self.server_button.setToolTip(TIP_FLORENCE_NO_SERVER)
            self.apply_button.setEnabled(False)
            self.apply_button.setToolTip(TIP_FLORENCE_NO_APPLY)
        else:
            self.server_button.setEnabled(True)
            self.server_button.setToolTip("")
            self.apply_button.setEnabled(downloaded)
            self.apply_button.setToolTip("")

    # -- prompt presets ----------------------------------------------------------------
    def _populate_presets(self, family: ModelFamily) -> None:
        """Fill the 预设 combo for a family, restoring the persisted choice."""
        if family.family_id == self._preset_family_id:
            return
        self._preset_family_id = family.family_id
        self.preset_combo.clear()
        for preset in presets_for(family.family_id):
            self.preset_combo.addItem(preset.label, preset.preset_id)
        self.preset_combo.addItem(PRESET_CUSTOM_LABEL, PRESET_CUSTOM)
        index = self.preset_combo.findData(self.bridge.settings.prompt_preset)
        # Unknown/empty stored ids land on the family's default (first) preset.
        self.preset_combo.setCurrentIndex(index if index >= 0 else 0)

    def _current_preset_choice(self) -> str:
        """Combo choice when populated; otherwise keep the stored value."""
        data = self.preset_combo.currentData()
        if data is None:
            return self.bridge.settings.prompt_preset
        return str(data)

    # -- downloads ---------------------------------------------------------------------
    def _on_download_clicked(self) -> None:
        if self.bridge.is_downloading():
            self.bridge.cancel_download()
            return
        selection = self.current_selection()
        if selection is None:
            self._toast(TOAST_SELECT_QUANT, TOAST_WARN)
            return
        family, quant = selection
        self.persist()
        if not self.bridge.start_download(family.family_id, quant.label):
            self._toast(TOAST_DOWNLOAD_BUSY, TOAST_WARN)
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, PROGRESS_BAR_MAX)
        self.progress.setValue(0)
        self._refresh_selection_ui()

    def _on_download_progress(
        self, _family_id: str, _quant_label: str, done: object, total: object
    ) -> None:
        if not isinstance(done, int):
            return
        if isinstance(total, int) and total > 0:
            self.progress.setRange(0, PROGRESS_BAR_MAX)
            self.progress.setValue(
                min(PROGRESS_BAR_MAX, round(done / total * PROGRESS_BAR_MAX))
            )
            self.progress.setFormat(
                DOWNLOAD_FORMAT.format(
                    done=format_bytes(done), total=format_bytes(total)
                )
            )
        else:
            self.progress.setRange(0, 0)

    def _on_download_finished(
        self, family_id: str, quant_label: str, status: str, message: str
    ) -> None:
        self.progress.setVisible(False)
        if status == DOWNLOAD_OK:
            if family_id.startswith(LORA_FAMILY_PREFIX):
                try:
                    name = find_lora(family_id[len(LORA_FAMILY_PREFIX) :]).name
                except NLaptError:
                    name = family_id
                self._toast(TOAST_LORA_OK.format(name=name), TOAST_OK)
                self._refresh_selection_ui()
                return
            try:
                family = find_family(family_id)
                name = family.name
            except NLaptError:
                name = family_id
            self._toast(
                TOAST_DOWNLOAD_OK.format(name=name, quant=quant_label), TOAST_OK
            )
        elif status == DOWNLOAD_CANCELLED:
            self._toast(TOAST_DOWNLOAD_CANCELLED, TOAST_WARN)
        else:
            self._toast(TOAST_DOWNLOAD_FAIL.format(message=message), TOAST_ERR)
        self._refresh_selection_ui()

    def _on_page_clicked(self) -> None:
        selection = self.current_selection()
        if selection is None:
            return
        QDesktopServices.openUrl(QUrl(repo_page_url(selection[0].repo_id)))

    # -- server ------------------------------------------------------------------------
    def _on_server_clicked(self) -> None:
        if self.bridge.server_running():
            self.server_button.setEnabled(False)
            self.bridge.stop_server()
            return
        selection = self.current_selection()
        if selection is None:
            self._toast(TOAST_SELECT_QUANT, TOAST_WARN)
            return
        family, quant = selection
        if family.engine == ENGINE_FLORENCE:
            self._toast(TIP_FLORENCE_NO_SERVER, TOAST_WARN)
            return
        self.persist()
        # A manual path OR the auto-provisioned runtime works; only a
        # platform with neither still needs a manual pick.
        if not self.bridge.settings.server_path.strip() and not runtime_supported():
            self._toast(TOAST_NEED_SERVER_PATH, TOAST_WARN)
            return
        if not self.bridge.is_downloaded(family, quant):
            self._toast(TOAST_NEED_DOWNLOAD, TOAST_WARN)
            return
        self.bridge.start_server(family.family_id, quant.label)

    def _on_server_changed(self, state: str, detail: str) -> None:
        if state == SERVER_STARTING:
            self.server_button.setEnabled(False)
            self.server_status.setText(SERVER_STATUS_STARTING)
            return
        self.server_button.setEnabled(True)
        if state == SERVER_RUNNING:
            self.server_button.setText(BTN_SERVER_STOP)
            self.server_status.setText(SERVER_STATUS_RUNNING.format(url=detail))
            self._toast(TOAST_SERVER_RUNNING.format(url=detail), TOAST_OK)
        elif state == SERVER_ERROR:
            self.server_button.setText(BTN_SERVER_START)
            self.server_status.setText(SERVER_STATUS_STOPPED)
            self._toast(TOAST_SERVER_FAIL.format(message=detail), TOAST_ERR)
        else:  # stopped
            self.server_button.setText(BTN_SERVER_START)
            self.server_status.setText(SERVER_STATUS_STOPPED)
            self._toast(TOAST_SERVER_STOPPED, TOAST_OK)

    # -- apply as active profile -------------------------------------------------------
    def _on_apply_clicked(self) -> None:
        selection = self.current_selection()
        if selection is None:
            self._toast(TOAST_SELECT_QUANT, TOAST_WARN)
            return
        family, _quant = selection
        self.persist()
        self._apply_profile(family)

    def _apply_profile(self, family: ModelFamily) -> None:
        """Register/refresh the ``local`` profile and make it active."""
        try:
            existing = load_app_config()
        except (ValidationError, StorageError):
            _LOGGER.exception("could not read config before applying local profile")
            self._toast(TOAST_CONFIG_UNREADABLE, TOAST_ERR)
            return
        settings = self.bridge.settings
        profile = LLMProfile(
            name=LOCAL_PROFILE_NAME,
            api_type=LOCAL_API_TYPE,
            base_url=local_base_url(settings.port),
            api_key="",
            text_model=family.family_id,
            vision_model=family.family_id if family.vision else "",
        )
        others = tuple(p for p in existing.profiles if p.name != LOCAL_PROFILE_NAME)
        config = _dc_replace(
            existing,
            profiles=(profile, *others),
            active_profile=LOCAL_PROFILE_NAME,
            request=_dc_replace(existing.request, concurrency=settings.parallel),
        )
        try:
            save_app_config(config)
        except NLaptError as exc:
            _LOGGER.exception("could not save config while applying local profile")
            self._toast(TOAST_SETTINGS_FAILED.format(message=exc.message), TOAST_ERR)
            return
        self._controller.reload_config(config)
        self._toast(
            TOAST_APPLIED.format(name=family.name, parallel=settings.parallel),
            TOAST_OK,
        )

    # -- browse ------------------------------------------------------------------------
    def _browse_models_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, LABEL_MODELS_DIR, self.models_dir_edit.text().strip()
        )
        if chosen:
            self.models_dir_edit.setText(chosen)

    def _browse_server_path(self) -> None:
        chosen, _selected_filter = QFileDialog.getOpenFileName(
            self, LABEL_SERVER_PATH, "", FILTER_EXECUTABLE
        )
        if chosen:
            self.server_path_edit.setText(chosen)

    def _add_extra_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, CAPTION_PICK_EXTRA_DIR, "")
        if chosen and chosen not in self._extra_dirs():
            self.extra_dirs_list.addItem(chosen)

    def _remove_extra_dir(self) -> None:
        row = self.extra_dirs_list.currentRow()
        if row >= 0:
            self.extra_dirs_list.takeItem(row)

    # -- misc --------------------------------------------------------------------------
    def _toast(self, text: str, kind: str) -> None:
        self._controller.toast_requested.emit(text, kind)
