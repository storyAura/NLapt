"""Carded right-pane construction for the 本地推理 settings tab.

Widgets are assigned onto the owning :class:`LocalTab` so existing
attribute names (``tree``, ``detail_label``, ``*_spin``, …) stay stable.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nlapt.local.advisor import RunGrade
from nlapt.local.catalog import (
    ModelFamily,
    all_series,
    families_for,
    find_lora,
    loras_for_family,
)
from nlapt.local.florence import FLORENCE_TASK_LABELS
from nlapt.local.hardware import format_bytes
from nlapt.local.settings import (
    CONTEXT_RANGE,
    GPU_LAYERS_RANGE,
    PARALLEL_RANGE,
    PORT_RANGE,
    THREADS_RANGE,
)

from nlapt_gui.widgets.local_tab import (
    BTN_ADD_DIR,
    BTN_APPLY,
    BTN_BROWSE,
    BTN_DETECT,
    BTN_DOWNLOAD,
    BTN_DOWNLOAD_LORA,
    BTN_PAGE,
    BTN_REMOVE_DIR,
    BTN_SERVER_START,
    CATALOG_HINT,
    COL_DOWNLOADS,
    COL_GRADE,
    COL_MODEL,
    COL_SIZE,
    CONTEXT_STEP,
    DETAIL_EMPTY,
    EXTRA_DIRS_LIST_H,
    GRADE_TEXT,
    HW_DETECTING,
    LABEL_CONTEXT,
    LABEL_EXTRA_DIRS,
    LABEL_FLORENCE_LORA,
    LABEL_FLORENCE_TASK,
    LABEL_GPU_LAYERS,
    LABEL_MODELS_DIR,
    LABEL_PARALLEL,
    LABEL_PORT,
    LABEL_PROMPT_PRESET,
    LABEL_SERVER_PATH,
    LABEL_THREADS,
    LOCAL_TAB_MIN_H,
    LOCAL_TAB_MIN_W,
    LORA_MISSING_SUFFIX,
    LORA_NONE_LABEL,
    RIGHT_PANE_MIN_W,
    ROLE_FAMILY,
    ROLE_LORA_ID,
    ROLE_QUANT,
    SERVER_STATUS_STOPPED,
    SPECIAL_GPU_AUTO,
    SPECIAL_THREADS_AUTO,
    TAG_RECOMMENDED,
    TAG_VISION,
    TIP_FLORENCE_LORA,
    TIP_FLORENCE_TASK,
    TIP_PROMPT_PRESET,
    TREE_MIN_W,
)

SECTION_STATUS = "模型状态"
SECTION_MODEL = "模型设置"
SECTION_RUNTIME = "运行参数"
SECTION_PATHS = "目录"
CARD_PAD = 10
CARD_GAP = 10


def section_card(title: str, parent: QWidget) -> tuple[QFrame, QVBoxLayout]:
    """Token-styled surface card with a section title and an inner body layout."""
    card = QFrame(parent)
    card.setProperty("surfaceCard", True)
    outer = QVBoxLayout(card)
    outer.setContentsMargins(CARD_PAD, CARD_PAD, CARD_PAD, CARD_PAD)
    outer.setSpacing(8)
    heading = QLabel(title, card)
    heading.setProperty("sectionTitle", True)
    outer.addWidget(heading)
    body = QVBoxLayout()
    body.setSpacing(8)
    outer.addLayout(body)
    return card, body


def _browse_row(edit: QLineEdit, button: QPushButton) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(edit, 1)
    row.addWidget(button)
    return row


def build_local_tab_ui(tab: QWidget) -> None:
    """Assemble hardware row + catalog tree + carded right pane on ``tab``."""
    tab.setMinimumSize(LOCAL_TAB_MIN_W, LOCAL_TAB_MIN_H)
    tab.hw_label = QLabel(HW_DETECTING, tab)
    tab.hw_label.setTextFormat(Qt.TextFormat.RichText)
    tab.hw_label.setWordWrap(True)
    tab.detect_button = QPushButton(BTN_DETECT, tab)
    tab.detect_button.setProperty("variant", "outline")
    hw_row = QHBoxLayout()
    hw_row.addWidget(tab.hw_label, 1)
    hw_row.addWidget(tab.detect_button, 0, Qt.AlignmentFlag.AlignTop)

    tab.tree = QTreeWidget(tab)
    tab.tree.setColumnCount(4)
    tab.tree.setHeaderLabels([COL_MODEL, COL_SIZE, COL_DOWNLOADS, COL_GRADE])
    tab.tree.setMinimumWidth(TREE_MIN_W)
    tab.tree.setRootIsDecorated(True)
    header = tab.tree.header()
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for column in (1, 2, 3):
        header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

    tab.catalog_hint = QLabel(CATALOG_HINT, tab)
    tab.catalog_hint.setProperty("muted", True)
    tab.catalog_hint.setWordWrap(True)
    left_pane = QVBoxLayout()
    left_pane.setSpacing(6)
    left_pane.addWidget(tab.tree, 1)
    left_pane.addWidget(tab.catalog_hint)

    tab.detail_label = QLabel(DETAIL_EMPTY, tab)
    tab.detail_label.setTextFormat(Qt.TextFormat.RichText)
    tab.detail_label.setProperty("muted", True)
    tab.detail_label.setWordWrap(True)
    tab.download_button = QPushButton(BTN_DOWNLOAD, tab)
    tab.download_button.setProperty("variant", "accent")
    tab.page_button = QPushButton(BTN_PAGE, tab)
    tab.page_button.setProperty("variant", "ghost")
    tab.server_button = QPushButton(BTN_SERVER_START, tab)
    tab.server_button.setProperty("variant", "outline")
    tab.apply_button = QPushButton(BTN_APPLY, tab)
    tab.apply_button.setProperty("variant", "outline")
    tab.progress = QProgressBar(tab)
    tab.progress.setVisible(False)
    tab.server_status = QLabel(SERVER_STATUS_STOPPED, tab)
    tab.server_status.setProperty("muted", True)
    tab.server_status.setWordWrap(True)

    actions_top = QHBoxLayout()
    actions_top.addWidget(tab.download_button)
    actions_top.addWidget(tab.page_button)
    actions_top.addStretch(1)
    actions_bottom = QHBoxLayout()
    actions_bottom.addWidget(tab.server_button)
    actions_bottom.addWidget(tab.apply_button)
    actions_bottom.addStretch(1)

    status_card, status_body = section_card(SECTION_STATUS, tab)
    status_body.addWidget(tab.detail_label)
    status_body.addLayout(actions_top)
    status_body.addLayout(actions_bottom)
    status_body.addWidget(tab.progress)
    status_body.addWidget(tab.server_status)

    tab.florence_task_combo = QComboBox(tab)
    for token, label in FLORENCE_TASK_LABELS.items():
        tab.florence_task_combo.addItem(label, token)
    tab.florence_task_combo.setToolTip(TIP_FLORENCE_TASK)
    tab.florence_lora_combo = QComboBox(tab)
    tab.florence_lora_combo.addItem(LORA_NONE_LABEL, "")
    tab.florence_lora_combo.setToolTip(TIP_FLORENCE_LORA)
    tab.lora_download_button = QPushButton(BTN_DOWNLOAD_LORA, tab)
    tab.lora_download_button.setProperty("variant", "ghost")
    tab.lora_download_button.setVisible(False)
    tab.lora_add_button = QPushButton(BTN_ADD_DIR, tab)
    tab.lora_add_button.setProperty("variant", "ghost")
    tab.lora_remove_button = QPushButton(BTN_REMOVE_DIR, tab)
    tab.lora_remove_button.setProperty("variant", "ghost")
    lora_row = QHBoxLayout()
    lora_row.setContentsMargins(0, 0, 0, 0)
    lora_row.addWidget(tab.florence_lora_combo, 1)
    lora_row.addWidget(tab.lora_download_button)
    lora_row.addWidget(tab.lora_add_button)
    lora_row.addWidget(tab.lora_remove_button)
    tab.florence_lora_holder = QWidget(tab)
    tab.florence_lora_holder.setLayout(lora_row)
    tab.preset_combo = QComboBox(tab)
    tab.preset_combo.setToolTip(TIP_PROMPT_PRESET)
    tab._preset_family_id = ""

    form = QFormLayout()
    form.setVerticalSpacing(6)
    form.addRow(LABEL_FLORENCE_TASK, tab.florence_task_combo)
    form.addRow(LABEL_FLORENCE_LORA, tab.florence_lora_holder)
    form.addRow(LABEL_PROMPT_PRESET, tab.preset_combo)
    tab._form = form
    form.setRowVisible(tab.florence_task_combo, False)
    form.setRowVisible(tab.florence_lora_holder, False)
    form.setRowVisible(tab.preset_combo, False)
    model_card, model_body = section_card(SECTION_MODEL, tab)
    model_body.addLayout(form)

    tab.context_spin = QSpinBox(tab)
    tab.context_spin.setRange(*CONTEXT_RANGE)
    tab.context_spin.setSingleStep(CONTEXT_STEP)
    tab.gpu_layers_spin = QSpinBox(tab)
    tab.gpu_layers_spin.setRange(*GPU_LAYERS_RANGE)
    tab.gpu_layers_spin.setSpecialValueText(SPECIAL_GPU_AUTO)
    tab.threads_spin = QSpinBox(tab)
    tab.threads_spin.setRange(*THREADS_RANGE)
    tab.threads_spin.setSpecialValueText(SPECIAL_THREADS_AUTO)
    tab.parallel_spin = QSpinBox(tab)
    tab.parallel_spin.setRange(*PARALLEL_RANGE)
    tab.port_spin = QSpinBox(tab)
    tab.port_spin.setRange(*PORT_RANGE)

    runtime = QGridLayout()
    runtime.setHorizontalSpacing(8)
    runtime.setVerticalSpacing(6)
    pairs = (
        (LABEL_CONTEXT, tab.context_spin),
        (LABEL_GPU_LAYERS, tab.gpu_layers_spin),
        (LABEL_THREADS, tab.threads_spin),
        (LABEL_PARALLEL, tab.parallel_spin),
        (LABEL_PORT, tab.port_spin),
    )
    for index, (label, widget) in enumerate(pairs):
        row, col = divmod(index, 2)
        caption = QLabel(label, tab)
        caption.setProperty("muted", True)
        runtime.addWidget(caption, row * 2, col)
        runtime.addWidget(widget, row * 2 + 1, col)
    runtime_card, runtime_body = section_card(SECTION_RUNTIME, tab)
    runtime_body.addLayout(runtime)

    tab.models_dir_edit = QLineEdit(tab)
    tab.models_dir_browse = QPushButton(BTN_BROWSE, tab)
    tab.models_dir_browse.setProperty("variant", "ghost")
    tab.server_path_edit = QLineEdit(tab)
    tab.server_path_browse = QPushButton(BTN_BROWSE, tab)
    tab.server_path_browse.setProperty("variant", "ghost")
    tab.extra_dirs_label = QLabel(LABEL_EXTRA_DIRS, tab)
    tab.extra_dirs_label.setProperty("muted", True)
    tab.extra_dirs_label.setWordWrap(True)
    tab.extra_dirs_list = QListWidget(tab)
    tab.extra_dirs_list.setFixedHeight(EXTRA_DIRS_LIST_H)
    tab.extra_add_button = QPushButton(BTN_ADD_DIR, tab)
    tab.extra_add_button.setProperty("variant", "ghost")
    tab.extra_remove_button = QPushButton(BTN_REMOVE_DIR, tab)
    tab.extra_remove_button.setProperty("variant", "ghost")
    extra_buttons = QVBoxLayout()
    extra_buttons.setSpacing(4)
    extra_buttons.addWidget(tab.extra_add_button)
    extra_buttons.addWidget(tab.extra_remove_button)
    extra_buttons.addStretch(1)
    extra_row = QHBoxLayout()
    extra_row.addWidget(tab.extra_dirs_list, 1)
    extra_row.addLayout(extra_buttons)

    paths_card, paths_body = section_card(SECTION_PATHS, tab)
    paths_form = QFormLayout()
    paths_form.setVerticalSpacing(6)
    paths_form.addRow(LABEL_MODELS_DIR, _browse_row(tab.models_dir_edit, tab.models_dir_browse))
    paths_form.addRow(LABEL_SERVER_PATH, _browse_row(tab.server_path_edit, tab.server_path_browse))
    paths_body.addLayout(paths_form)
    paths_body.addWidget(tab.extra_dirs_label)
    paths_body.addLayout(extra_row)

    stack = QWidget(tab)
    stack_lay = QVBoxLayout(stack)
    stack_lay.setContentsMargins(0, 0, 0, 0)
    stack_lay.setSpacing(CARD_GAP)
    stack_lay.addWidget(status_card)
    stack_lay.addWidget(model_card)
    stack_lay.addWidget(runtime_card)
    stack_lay.addWidget(paths_card)
    stack_lay.addStretch(1)

    scroll = QScrollArea(tab)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(stack)
    scroll.setMinimumWidth(RIGHT_PANE_MIN_W)

    body = QHBoxLayout()
    body.setSpacing(12)
    body.addLayout(left_pane, 3)
    body.addWidget(scroll, 2)

    column = QVBoxLayout(tab)
    column.setSpacing(8)
    column.addLayout(hw_row)
    column.addLayout(body, 1)


def format_downloads(count: int) -> str:
    """Compact Chinese download count: 1_491_605 -> '149.2 万'."""
    if count >= 10_000:
        return f"{count / 10_000:.1f} 万"
    return str(count)


def _embolden(item: QTreeWidgetItem, column: int) -> None:
    """Bold one cell (names and grades stand out)."""
    font = item.font(column)
    font.setBold(True)
    item.setFont(column, font)


def _family_item(family: ModelFamily) -> QTreeWidgetItem:
    """One family row; a single-quant family is itself the quant row."""
    name = f"{family.name} · {family.params_label}"
    if family.vision:
        name = f"{name} · {TAG_VISION}"
    tooltip = family.notes or family.name
    extras_bytes = sum(extra.size_bytes for extra in family.extra_files)
    if len(family.quants) == 1:
        quant = family.quants[0]
        merged = QTreeWidgetItem(
            [
                name,
                format_bytes(quant.size_bytes + extras_bytes),
                format_downloads(family.downloads),
                GRADE_TEXT[RunGrade.UNKNOWN],
            ]
        )
        merged.setToolTip(0, f"{tooltip}\n{family.repo_id} · {family.license}")
        merged.setData(0, ROLE_FAMILY, family.family_id)
        merged.setData(0, ROLE_QUANT, quant.label)
        _embolden(merged, 0)
        _embolden(merged, 3)
        return merged
    family_item = QTreeWidgetItem(
        [name, "", format_downloads(family.downloads), ""]
    )
    family_item.setToolTip(0, f"{tooltip}\n{family.repo_id} · {family.license}")
    family_item.setFlags(family_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
    family_item.setData(0, ROLE_FAMILY, family.family_id)
    _embolden(family_item, 0)
    for quant in family.quants:
        label = f"{quant.label}  {TAG_RECOMMENDED}" if quant.recommended else quant.label
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
        _embolden(quant_item, 3)
        family_item.addChild(quant_item)
    return family_item


def populate_catalog_tree(tab: QWidget) -> None:
    """Build the catalog tree, collapsing single-child chains (不分层)."""
    tab.tree.clear()
    for series in all_series():
        family_items = [
            _family_item(family) for family in families_for(series.series_id)
        ]
        if len(family_items) == 1:
            tab.tree.addTopLevelItem(family_items[0])
            family_items[0].setExpanded(True)
            continue
        series_item = QTreeWidgetItem([series.name, "", "", ""])
        series_item.setToolTip(0, series.description)
        series_item.setFlags(series_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        _embolden(series_item, 0)
        tab.tree.addTopLevelItem(series_item)
        for family_item in family_items:
            series_item.addChild(family_item)
        series_item.setExpanded(True)


def quant_items(tab: QWidget) -> tuple[QTreeWidgetItem, ...]:
    """Every selectable quant row, at whatever depth flattening left it."""
    items: list[QTreeWidgetItem] = []

    def walk(item: QTreeWidgetItem) -> None:
        if item.data(0, ROLE_QUANT):
            items.append(item)
        for index in range(item.childCount()):
            walk(item.child(index))

    for index in range(tab.tree.topLevelItemCount()):
        walk(tab.tree.topLevelItem(index))
    return tuple(items)


def refresh_lora_buttons(tab: QWidget) -> None:
    """下载 appears only for a curated LoRA that is not on disk yet."""
    combo = tab.florence_lora_combo
    lora_id = combo.itemData(combo.currentIndex(), ROLE_LORA_ID)
    show_download = bool(lora_id) and not tab.bridge.is_lora_downloaded(
        find_lora(str(lora_id))
    )
    tab.lora_download_button.setVisible(show_download)
    tab.lora_download_button.setEnabled(not tab.bridge.is_downloading())


def rebuild_lora_combo(tab: QWidget, family: ModelFamily | None) -> None:
    """[不使用] + curated LoRAs of ``family`` + user files; selection kept."""
    combo = tab.florence_lora_combo
    selected = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    combo.addItem(LORA_NONE_LABEL, "")
    if family is not None:
        for entry in loras_for_family(family.family_id):
            label = entry.name
            if not tab.bridge.is_lora_downloaded(entry):
                label += LORA_MISSING_SUFFIX
            combo.addItem(label, str(tab.bridge.lora_adapter_file(entry)))
            index = combo.count() - 1
            combo.setItemData(index, entry.lora_id, ROLE_LORA_ID)
            combo.setItemData(index, entry.notes, Qt.ItemDataRole.ToolTipRole)
    for lora in tab._user_loras:
        combo.addItem(Path(lora).stem, lora)
    index = combo.findData(selected) if selected else 0
    combo.setCurrentIndex(index if index >= 0 else 0)
    combo.blockSignals(False)
    refresh_lora_buttons(tab)


def rebuild_task_combo(tab: QWidget, family: ModelFamily) -> None:
    """Only the 指令 tokens ``family`` was trained on; selection kept."""
    combo = tab.florence_task_combo
    allowed = family.florence_tasks or tuple(FLORENCE_TASK_LABELS)
    current = combo.currentData()
    combo.clear()
    for token in allowed:
        combo.addItem(FLORENCE_TASK_LABELS[token], token)
    index = combo.findData(current)
    combo.setCurrentIndex(index if index >= 0 else 0)
