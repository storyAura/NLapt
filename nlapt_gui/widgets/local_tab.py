"""设置 ▸ 本地推理 tab: catalog tree, run grades, downloads, server.

Landscape composition (用户要求 横构图): the catalog tree fills the left
side; the right side stacks the selection detail, the action buttons, the
runtime settings (下载目录 / 复用目录 / llama-server / 上下文 / GPU 层 /
线程 / 并发 / 端口) and the hint. Per-quant「能否运行」is a plain
five-level Chinese grade from :class:`nlapt.local.advisor.RunGrade`
(轻松运行 / 流畅运行 / 可以运行 / 勉强能跑 / 跑不动).

Model files download into the primary 下载目录 (default: inside the app,
``models/``) and are FOUND in the primary + any 复用目录, so models already
downloaded by other tools are reused instead of re-downloaded.

All slow work goes through :class:`nlapt_gui.local_bridge.LocalBridge`.
Like the settings dialog that hosts it, this tab is part of the sanctioned
exception that may write the core config directly (设为当前模型 registers a
``local`` OpenAI-compatible profile pointing at the llama-server).
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace

from PySide6.QtCore import Qt, QThreadPool, QUrl
from PySide6.QtGui import QDesktopServices, QShowEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nlapt.core.config import LLMProfile, load_config, save_config
from nlapt.core.errors import NLaptError, StorageError, ValidationError
from nlapt.diagnostics import get_logger
from nlapt.local.advisor import RunAssessment, RunGrade, assess, estimate_memory
from nlapt.local.catalog import (
    CATALOG_SNAPSHOT_DATE,
    ENGINE_FLORENCE,
    ModelFamily,
    QuantFile,
    all_series,
    families_for,
    find_family,
    find_quant,
    repo_page_url,
)
from nlapt.local.florence import FLORENCE_TASK_LABELS
from nlapt.local.hardware import HardwareInfo, format_bytes
from nlapt.local.presets import PRESET_CUSTOM, presets_for
from nlapt.local.server import base_url as local_base_url
from nlapt.local.settings import (
    CONTEXT_RANGE,
    GPU_LAYERS_RANGE,
    PARALLEL_RANGE,
    PORT_RANGE,
    THREADS_RANGE,
)

from nlapt.local.runtime import LLAMA_CPP_TAG

from nlapt_gui.controller import AppController, TOAST_ERR, TOAST_OK, TOAST_WARN
from nlapt_gui.local_bridge import (
    DOWNLOAD_CANCELLED,
    DOWNLOAD_OK,
    SERVER_ERROR,
    SERVER_RUNNING,
    SERVER_STARTING,
    LocalBridge,
    default_models_dir,
    runtime_supported,
)
from nlapt_gui.resources import config_path

_LOGGER = get_logger(__name__)

LOCAL_PROFILE_NAME = "local"
LOCAL_API_TYPE = "openai"

ROLE_FAMILY = Qt.ItemDataRole.UserRole
ROLE_QUANT = Qt.ItemDataRole.UserRole + 1

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
SERVER_HINT = (
    "本地服务基于 llama.cpp 的 llama-server:程序自带完整推理能力,"
    "首次下载模型 / 启动服务时会自动获取官方运行时(上方留空即用自动版本,"
    "也可手动指定可执行文件)。模型就绪后,编辑区与文件夹右键的「本地推理」"
    "可直接使用;「设为当前模型」额外把翻译 / 重译也切到本地模型。"
    "Florence-2 PromptGen 模型例外:免服务、免运行时,按「指令模式」直接打标。"
)
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


def _format_downloads(count: int) -> str:
    """Compact Chinese download count: 1_491_605 -> '149.2 万'."""
    if count >= 10_000:
        return f"{count / 10_000:.1f} 万"
    return str(count)


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
        self.setMinimumSize(LOCAL_TAB_MIN_W, LOCAL_TAB_MIN_H)
        self.hw_label = QLabel(HW_DETECTING, self)
        self.hw_label.setTextFormat(Qt.TextFormat.RichText)
        self.hw_label.setWordWrap(True)
        self.detect_button = QPushButton(BTN_DETECT, self)
        self.detect_button.setProperty("variant", "outline")
        hw_row = QHBoxLayout()
        hw_row.addWidget(self.hw_label, 1)
        hw_row.addWidget(self.detect_button, 0, Qt.AlignmentFlag.AlignTop)

        # -- left: catalog tree -------------------------------------------------------
        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels([COL_MODEL, COL_SIZE, COL_DOWNLOADS, COL_GRADE])
        self.tree.setMinimumWidth(TREE_MIN_W)
        self.tree.setRootIsDecorated(True)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

        self.catalog_hint = QLabel(CATALOG_HINT, self)
        self.catalog_hint.setProperty("muted", True)
        self.catalog_hint.setWordWrap(True)

        left_pane = QVBoxLayout()
        left_pane.setSpacing(6)
        left_pane.addWidget(self.tree, 1)
        left_pane.addWidget(self.catalog_hint)

        # -- right: detail, actions, runtime settings ---------------------------------
        self.detail_label = QLabel(DETAIL_EMPTY, self)
        self.detail_label.setTextFormat(Qt.TextFormat.RichText)
        self.detail_label.setProperty("muted", True)
        self.detail_label.setWordWrap(True)

        self.download_button = QPushButton(BTN_DOWNLOAD, self)
        self.download_button.setProperty("variant", "accent")
        self.page_button = QPushButton(BTN_PAGE, self)
        self.page_button.setProperty("variant", "ghost")
        action_row_top = QHBoxLayout()
        action_row_top.addWidget(self.download_button)
        action_row_top.addWidget(self.page_button)
        action_row_top.addStretch(1)
        self.server_button = QPushButton(BTN_SERVER_START, self)
        self.server_button.setProperty("variant", "outline")
        self.apply_button = QPushButton(BTN_APPLY, self)
        self.apply_button.setProperty("variant", "outline")
        action_row_bottom = QHBoxLayout()
        action_row_bottom.addWidget(self.server_button)
        action_row_bottom.addWidget(self.apply_button)
        action_row_bottom.addStretch(1)

        self.progress = QProgressBar(self)
        self.progress.setVisible(False)
        self.server_status = QLabel(SERVER_STATUS_STOPPED, self)
        self.server_status.setProperty("muted", True)
        self.server_status.setWordWrap(True)

        self.models_dir_edit = QLineEdit(self)
        self.models_dir_browse = QPushButton(BTN_BROWSE, self)
        self.models_dir_browse.setProperty("variant", "ghost")
        models_dir_row = QHBoxLayout()
        models_dir_row.addWidget(self.models_dir_edit, 1)
        models_dir_row.addWidget(self.models_dir_browse)
        self.server_path_edit = QLineEdit(self)
        self.server_path_browse = QPushButton(BTN_BROWSE, self)
        self.server_path_browse.setProperty("variant", "ghost")
        server_path_row = QHBoxLayout()
        server_path_row.addWidget(self.server_path_edit, 1)
        server_path_row.addWidget(self.server_path_browse)

        self.context_spin = QSpinBox(self)
        self.context_spin.setRange(*CONTEXT_RANGE)
        self.context_spin.setSingleStep(CONTEXT_STEP)
        self.gpu_layers_spin = QSpinBox(self)
        self.gpu_layers_spin.setRange(*GPU_LAYERS_RANGE)
        self.gpu_layers_spin.setSpecialValueText(SPECIAL_GPU_AUTO)
        self.threads_spin = QSpinBox(self)
        self.threads_spin.setRange(*THREADS_RANGE)
        self.threads_spin.setSpecialValueText(SPECIAL_THREADS_AUTO)
        self.parallel_spin = QSpinBox(self)
        self.parallel_spin.setRange(*PARALLEL_RANGE)
        self.port_spin = QSpinBox(self)
        self.port_spin.setRange(*PORT_RANGE)

        self.florence_task_combo = QComboBox(self)
        for token, label in FLORENCE_TASK_LABELS.items():
            self.florence_task_combo.addItem(label, token)
        self.florence_task_combo.setToolTip(TIP_FLORENCE_TASK)

        # 提示词预设 row: only shown for caption specialists with official
        # presets (JoyCaption / ToriiGate); populated per selected family.
        self.preset_combo = QComboBox(self)
        self.preset_combo.setToolTip(TIP_PROMPT_PRESET)
        self._preset_family_id = ""

        form = QFormLayout()
        form.setVerticalSpacing(6)
        # 指令模式 row: only meaningful (and only shown) for Florence models.
        form.addRow(LABEL_FLORENCE_TASK, self.florence_task_combo)
        form.addRow(LABEL_PROMPT_PRESET, self.preset_combo)
        self._form = form
        form.setRowVisible(self.florence_task_combo, False)
        form.setRowVisible(self.preset_combo, False)
        form.addRow(LABEL_MODELS_DIR, models_dir_row)
        form.addRow(LABEL_SERVER_PATH, server_path_row)
        form.addRow(LABEL_CONTEXT, self.context_spin)
        form.addRow(LABEL_GPU_LAYERS, self.gpu_layers_spin)
        form.addRow(LABEL_THREADS, self.threads_spin)
        form.addRow(LABEL_PARALLEL, self.parallel_spin)
        form.addRow(LABEL_PORT, self.port_spin)

        self.extra_dirs_label = QLabel(LABEL_EXTRA_DIRS, self)
        self.extra_dirs_label.setProperty("muted", True)
        self.extra_dirs_label.setWordWrap(True)
        self.extra_dirs_list = QListWidget(self)
        self.extra_dirs_list.setFixedHeight(EXTRA_DIRS_LIST_H)
        self.extra_add_button = QPushButton(BTN_ADD_DIR, self)
        self.extra_add_button.setProperty("variant", "ghost")
        self.extra_remove_button = QPushButton(BTN_REMOVE_DIR, self)
        self.extra_remove_button.setProperty("variant", "ghost")
        extra_buttons = QVBoxLayout()
        extra_buttons.setSpacing(4)
        extra_buttons.addWidget(self.extra_add_button)
        extra_buttons.addWidget(self.extra_remove_button)
        extra_buttons.addStretch(1)
        extra_row = QHBoxLayout()
        extra_row.addWidget(self.extra_dirs_list, 1)
        extra_row.addLayout(extra_buttons)

        self.server_hint = QLabel(SERVER_HINT, self)
        self.server_hint.setProperty("muted", True)
        self.server_hint.setWordWrap(True)

        right_pane = QVBoxLayout()
        right_pane.setSpacing(8)
        right_pane.addWidget(self.detail_label)
        right_pane.addLayout(action_row_top)
        right_pane.addLayout(action_row_bottom)
        right_pane.addWidget(self.progress)
        right_pane.addWidget(self.server_status)
        right_pane.addLayout(form)
        right_pane.addWidget(self.extra_dirs_label)
        right_pane.addLayout(extra_row)
        right_pane.addWidget(self.server_hint)
        right_pane.addStretch(1)
        right_holder = QWidget(self)
        right_holder.setMinimumWidth(RIGHT_PANE_MIN_W)
        right_holder.setLayout(right_pane)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addLayout(left_pane, 3)
        body.addWidget(right_holder, 2)

        column = QVBoxLayout(self)
        column.setSpacing(8)
        column.addLayout(hw_row)
        column.addLayout(body, 1)

    def _populate_tree(self) -> None:
        self.tree.clear()
        for series in all_series():
            series_item = QTreeWidgetItem([series.name, "", "", ""])
            series_item.setToolTip(0, series.description)
            series_item.setFlags(series_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._embolden(series_item, 0)
            self.tree.addTopLevelItem(series_item)
            for family in families_for(series.series_id):
                name = f"{family.name} · {family.params_label}"
                if family.vision:
                    name = f"{name} · {TAG_VISION}"
                family_item = QTreeWidgetItem(
                    [name, "", _format_downloads(family.downloads), ""]
                )
                tooltip = family.notes or family.name
                family_item.setToolTip(
                    0, f"{tooltip}\n{family.repo_id} · {family.license}"
                )
                family_item.setFlags(
                    family_item.flags() & ~Qt.ItemFlag.ItemIsSelectable
                )
                family_item.setData(0, ROLE_FAMILY, family.family_id)
                self._embolden(family_item, 0)
                series_item.addChild(family_item)
                extras_bytes = sum(extra.size_bytes for extra in family.extra_files)
                for quant in family.quants:
                    label = quant.label
                    if quant.recommended:
                        label = f"{label}  {TAG_RECOMMENDED}"
                    quant_item = QTreeWidgetItem(
                        [
                            label,
                            format_bytes(quant.size_bytes + extras_bytes),
                            "",
                            GRADE_TEXT[RunGrade.UNKNOWN],
                        ]
                    )
                    quant_item.setData(0, ROLE_FAMILY, family.family_id)
                    quant_item.setData(0, ROLE_QUANT, quant.label)
                    self._embolden(quant_item, 3)
                    family_item.addChild(quant_item)
            series_item.setExpanded(True)

    @staticmethod
    def _embolden(item: QTreeWidgetItem, column: int) -> None:
        """Bold one cell (重要字眼需要明显 — names and grades stand out)."""
        font = item.font(column)
        font.setBold(True)
        item.setFont(column, font)

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
        items: list[QTreeWidgetItem] = []
        for series_index in range(self.tree.topLevelItemCount()):
            series_item = self.tree.topLevelItem(series_index)
            for family_index in range(series_item.childCount()):
                family_item = series_item.child(family_index)
                items.extend(
                    family_item.child(i) for i in range(family_item.childCount())
                )
        return tuple(items)

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
            existing = load_config(config_path())
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
            save_config(config_path(), config)
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
