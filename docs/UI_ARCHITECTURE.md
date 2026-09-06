# NLapt GUI Migration Contract (v1)

Binding contract for implementing the **CaptionForge** design
(`NLapt-head/untitled/project/标注编辑器 CaptionForge.dc.html`) as a real PySide6 desktop
UI wired to the existing `nlapt` core library (`docs/ARCHITECTURE.md`). Recreate the
design faithfully (layout, spacing, radii, colors, behaviors); do not copy the
prototype's internal structure.

Stack: **Python 3.12 + PySide6 6.11 (QtWidgets)**. Tests: pytest + pytest-qt, offscreen.
UI text is Chinese exactly as in the design; code/comments/docstrings in English.

## Global rules

- All rules of `docs/ARCHITECTURE.md` (typing, immutability for data types, ≤800-line
  files, no magic values, typed errors, loggers via `nlapt.diagnostics.get_logger`).
- **No hardcoded colors anywhere** — every color comes from `nlapt_gui.theme.tokens`
  through the generated QSS or token lookups (design mandate + spec 3.3).
- **UI never touches the filesystem or core packages directly** — everything goes
  through `AppController` (the only import of `nlapt.app` in the GUI layer).
- Disk/LLM work runs in `QThreadPool` workers; results come back via Qt signals
  (queued). Never block the GUI thread; never call Qt widget methods from workers.
- Widgets communicate ONLY via `AppController` signals/methods — no widget-to-widget
  references except parent→child composition.
- Test files: `tests/gui/test_gui_<module>.py` (globally unique names). `tests/gui/conftest.py`
  sets `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` before Qt import and
  provides shared fixtures (tmp dataset factory using Pillow-generated 4×3 PNGs).
- Fonts: font stacks only (no bundled fonts): UI `"IBM Plex Sans","Segoe UI","Microsoft YaHei"`;
  mono `"IBM Plex Mono","Consolas"`. Constants `FONT_STACK` / `MONO_STACK` in theme/tokens.py.

## Package layout & ownership

```
nlapt_gui/
  __init__.py            # __version__ re-export from nlapt              [foundation]
  __main__.py            # main(): QApplication, theme, controller, window [integrator]
  resources.py           # resource_path()/app_data_dir() (PyInstaller-aware) [foundation]
  settings.py            # UISettings frozen dataclass + JSON load/save   [foundation]
  workers.py             # FunctionWorker(QRunnable) with done/error signals [foundation]
  controller.py          # AppController(QObject) — UI⇄core bridge        [foundation]
  history_model.py       # per-file labeled caption history (panel data)  [foundation]
  translate_bridge.py    # async LLM segment translation                  [agent C]
  theme/
    tokens.py            # THEMES/ACCENT_OPTIONS/sizes/font stacks        [foundation]
    qss.py               # build_qss(theme, accent) -> str                [foundation]
    manager.py           # ThemeManager(QObject): apply/persist/signal    [foundation]
  widgets/
    toast.py             # ToastOverlay                                   [agent C]
    collapsible.py       # animated CollapsibleSection                    [agent C]
    flow_layout.py       # FlowLayout for chips                           [agent B]
    file_panel.py        # left column                                    [agent A]
    thumbnails.py        # ThumbnailLoader (QThreadPool + LRU cache)      [agent A]
    thumb_cells.py       # grid cell / list row widgets                   [agent A]
    preview_panel.py     # middle header bar + single/multi preview+zoom  [agent A]
    editor_panel.py      # mode tabs, editor blocks, floating seg toolbar [agent B]
    chips_editor.py      # chips mode                                     [agent B]
    sents_editor.py      # sentences mode                                 [agent B]
    text_editor.py       # free-text mode + selection action bar         [agent B]
    tools_panel.py       # right column: 4 collapsible sections           [agent C]
    sections/__init__.py                                                  [agent C]
    sections/find_replace.py                                              [agent C]
    sections/prefix_suffix.py                                             [agent C]
    sections/translate.py                                                 [agent C]
    sections/history.py                                                   [agent C]
    settings_dialog.py   # minimal LLM profile dialog (工具 menu)          [agent C]
    title_bar.py         # frameless title bar + menus + theme popup      [integrator]
    status_bar.py        # bottom status strip                            [integrator]
    main_window.py       # MainWindow assembly + shortcuts + drag/resize  [integrator]
packaging/
  nlapt.spec             # PyInstaller spec (reserved)                    [integrator]
  build.ps1              # windows build script                           [integrator]
  README.md              # packaging guide                                [integrator]
tests/gui/               # per-owner test files (see ownership above)
```

## Design tokens (theme/tokens.py) — exact values from the design

```python
FONT_STACK = ("IBM Plex Sans", "Segoe UI", "Microsoft YaHei", "PingFang SC")
MONO_STACK = ("IBM Plex Mono", "Consolas", "monospace")
DEFAULT_THEME = "雾灰"; DEFAULT_ACCENT = "#0E9384"
ACCENT_OPTIONS = ("#0E9384", "#4F63E7", "#D9634A", "#B58326")
THUMB_MIN_RANGE = (72, 150); DEFAULT_THUMB_MIN = 96
EDITOR_H_RANGE = (170, 620); DEFAULT_EDITOR_H = 330
ZOOM_RANGE = (40, 260); ZOOM_STEP = 20
MIN_WINDOW = (1360, 760)

@dataclass(frozen=True)
class ThemeTokens:
    name: str; scheme: str  # 'light' | 'dark'
    bg: str; panel: str; surface: str; surface2: str
    bd: str; bd2: str; text: str; text2: str; text3: str
    accent: str; accent2: str; onaccent: str
    ok: str; warn: str; danger: str; scroll: str

THEMES: Mapping[str, ThemeTokens]  # keys: 明亮 雾灰 石墨 深邃 墨黑
```

Values (bg/panel/surface/surface2/bd/bd2/text/text2/text3/accent/accent2/onaccent/ok/warn/danger/scroll):

- 明亮 light: `#F5F5F6 #FBFBFC #FFFFFF #F0F0F2 #E4E5E9 #CACCD3 #1B1D22 #54575F #8B8E96 #0E9384 #0B7C70 #FFFFFF #188E4E #C97A10 #D4453A #CDCFD6`
- 雾灰 light: `#E7E8EA #EFEFF1 #F7F7F8 #E2E3E6 #D6D7DB #B9BBC2 #212327 #565962 #8F929B #0E9384 #0B7C70 #FFFFFF #188E4E #C97A10 #D4453A #BFC1C8`
- 石墨 dark: `#292B30 #303237 #393C42 #43464D #484B53 #5B5F69 #ECEDEF #B3B6BD #83868F #31C0AF #4FD2C3 #07211D #46C57E #E8A33D #E9705F #54575F`
- 深邃 dark: `#16171B #1C1E23 #24262C #2D3037 #32353D #464A55 #E7E8EC #A5A8B1 #6E717B #31C0AF #55D4C5 #07211D #46C57E #E8A33D #E9705F #3A3D46`
- 墨黑 dark: `#0C0D0F #111215 #18191D #212227 #282A30 #3D4048 #E5E6E9 #9DA0A9 #63666F #38C9B7 #5CDACB #052019 #4BCB82 #EDAA45 #EE7767 #31343B`

Helpers: `mix(color_a, color_b, pct) -> str` (srgb mix, replaces CSS `color-mix`),
`accent_soft(tokens) -> str` (14% accent over transparent → pre-mixed with bg),
`with_accent(tokens, accent) -> ThemeTokens` (accent override; accent2 = 82% accent mixed
with text, matching the prototype).

`qss.py`: `build_qss(tokens: ThemeTokens) -> str` — one global stylesheet: panels,
buttons (dynamic property `variant` in {"accent","ghost","outline","danger-ghost"}),
inputs (focus ring via border color; Qt has no box-shadow — use 1.5px accent border),
scrollbars (10px, rounded thumb `scroll` color), pills (property `pill=true`), segmented
controls (property `seg=true`, `segActive=true`), chips, collapsible headers, toasts.
Toggle chips (`toggleChip=true` + `chipOn=true`) have full interaction states: hover
(bd2 border), pressed (accent25 bg, instant), checked/`chipOn` (soft bg + solid accent
border + accent text, 600 weight); QSS also matches `:checked` so feedback lands the
instant Qt flips the button, before the property repolish.
Radii from the design: window sections 11px, buttons 6–8px, inputs 7px, chips 999px.
CAUTION: `url()` values containing `;` (the chevron data URI) MUST be quoted — unquoted,
Qt stops parsing at that declaration and silently drops every later rule; the
render-level regression test `test_stylesheet_stays_valid_to_the_last_rule` guards this.
`manager.py`: `ThemeManager(QObject)` — `apply(theme_name, accent)` sets QSS on
QApplication + Qt palette (dark scheme for native menus), emits `theme_changed(ThemeTokens)`,
persists via UISettings.

## settings.py

```python
@dataclass(frozen=True)
class UISettings:
    theme: str = DEFAULT_THEME
    accent: str = DEFAULT_ACCENT
    thumb_min: int = DEFAULT_THUMB_MIN
    view_mode: str = "mid"            # list|mid|big
    editor_h: int = DEFAULT_EDITOR_H
    folder_open: Mapping[str, bool] = field(default_factory=dict)
    last_root: str = ""
    sections: Mapping[str, bool] = field(default_factory=lambda: {"fr": True, "ps": False, "tr": False, "hist": True})
def load_ui_settings(path: Path | None = None) -> UISettings   # default path=app_data_dir()/ui_settings.json; corrupt -> defaults + warning
def save_ui_settings(settings: UISettings, path: Path | None = None) -> None  # atomic via nlapt.storage.atomic
```

`resources.py`: `resource_path(rel: str) -> Path` (handles PyInstaller `sys._MEIPASS`),
`app_data_dir() -> Path` (`%APPDATA%/NLapt` on Windows else `~/.config/nlapt`; created on demand).

## workers.py

```python
class FunctionWorker(QRunnable):
    """Run fn(*args) on QThreadPool; emits signals.done(result) / signals.error(str) on the GUI thread."""
def run_async(pool: QThreadPool, fn, *args, on_done=None, on_error=None) -> None
```

## controller.py — AppController (the single UI⇄core bridge)

```python
class AppController(QObject):
    # signals
    dataset_opened = Signal(object)        # DatasetScanResult
    dataset_open_failed = Signal(str)
    current_changed = Signal(str)          # key ('' when none)
    caption_changed = Signal(str)          # key — text/dirty changed (any source)
    selection_changed = Signal()
    filter_changed = Signal(str)
    mode_changed = Signal(str)             # chips|sents|text
    view_mode_changed = Signal(str)        # list|mid|big
    files_saved = Signal(tuple)            # keys just saved
    batch_finished = Signal(str, object)   # description, BatchReport
    toast_requested = Signal(str, str)     # text, kind: ok|warn|err|info
    busy_changed = Signal(bool)

    def __init__(self, app: NLaptApp | None = None, *, settings: UISettings | None = None,
                 pool: QThreadPool | None = None) -> None
    # dataset
    def open_dataset(self, root: Path) -> None     # async; wires core EventBus→Qt signals; toast on encoding issues/txt conflicts
    def refresh(self) -> None                      # rescan same root
    root: Path | None (property); dataset_label() -> tuple[str, str]  # (folder name, path str)
    # data access
    def keys(self) -> tuple[str, ...]
    def filtered_keys(self) -> tuple[str, ...]     # name OR caption contains filter (case-insensitive); empty filter -> all
    def set_filter(self, text: str) -> None
    def folders(self) -> tuple[str, ...]           # relative dirs in natural order; root files under FOLDER_ROOT_LABEL="根目录"
    def folder_of(self, key: str) -> str
    def record(self, key: str) -> CaptionRecord    # .dirty drives all 未保存 dots
    def image_path(self, key: str) -> Path
    def image_meta(self, key: str) -> str          # "480 × 640 · PNG · 1.2 MB" (cached, lazy via QImageReader/stat)
    def segments(self, key: str) -> tuple[str, ...]        # nlapt.captions.chips.split_chips
    def char_seg_info(self, key: str) -> str               # "n 字符 · m 段"
    # current / nav
    current_key: str | None (property)
    def set_current(self, key: str) -> None
    def nav(self, delta: int) -> None              # order: selected if multi_mode else filtered; wraps
    def pos_label(self) -> str                     # "i / n" per design rules
    # selection (ordered by keys() order; anchor for shift)
    def selected_keys(self) -> tuple[str, ...]
    def is_selected(self, key) -> bool
    def set_anchor(self, key) -> None
    def toggle_selected(self, key) -> None         # selecting sets anchor; if multi and current not in sel -> current=first selected
    def select_range_to(self, key) -> None         # anchor..key over filtered_keys
    def deselect(self, key) -> None
    def select_all(self) -> None / clear_selection(self) -> None
    def multi_mode(self) -> bool                   # len(selected) >= 2
    def editor_keys(self) -> tuple[str, ...]       # selected[:4] if multi else (current,) — drives preview grid + editor blocks
    # editing (every call pushes labeled UI history + emits caption_changed)
    def set_caption(self, key: str, text: str, label: str) -> None
    def commit_segments(self, key: str, segs: Sequence[str], label: str) -> None   # join_chips
    def undo_current(self) -> None                 # app.undo; toast '已撤销'/'没有可撤销的操作'
    def save_current(self) -> None                 # toast '已保存 <name>.txt' / '没有未保存的更改'
    def save_all(self) -> None                     # toast '已保存 n 个文件' / none-msg
    def copy_caption(self) -> None                 # QGuiApplication.clipboard; toast
    # editor mode & view mode & counts
    mode: str (property + set_mode); view_mode: str (property + set_view_mode)
    def dirty_count(self) -> int; def stat_line(self) -> str  # "n 张图片 · 已选 k · 未保存 d"
    # tool scopes
    def scope_keys(self, scope: str) -> tuple[str, ...]   # current|selected|all
    def count_matches(self, find: str, case_sensitive: bool, scope: str) -> tuple[int, int]  # (total, files) via ops.find_replace.scope_stats on in-memory texts
    def replace_all(self, find: str, replace: str, case_sensitive: bool, scope: str) -> None
        # FindReplaceSpec(regex=False, whole_word=False); async app.apply_operation (snapshot+oplog);
        # toasts: '请输入查找内容'/'尚未选择任何图片'/'没有找到匹配内容'/'已在 k 个文件中替换 m 处'
        # after batch: refresh UI history entries for changed keys (label '查找替换')
    def apply_prefix_suffix(self, text: str, position: str, as_tag: bool, scope: str) -> None
        # PrefixSuffixSpec(prefix or suffix, joiner=', ' if as_tag else '', skip_if_present=as_tag)
        # toasts per design; label '添加前缀「x」'/'添加后缀「x」'
    # services
    history: FileHistory (property)
    def make_translator_or_none(self) -> Translator | None   # None when unconfigured; cached
    def close(self) -> None    # save session (crash-safe drafts), persist UISettings
```

Implementation notes: core `EventBus` handlers may fire on worker threads — bridge by
emitting Qt signals only (auto-queued). `set_caption` routes through `app.edit`
(dirty flag, undo stack, index update) then `history.push(key, label, text)`.
Filter uses in-memory records, not `app.search` (design filters name OR caption substring).

## history_model.py — right-panel labeled history (UI-level)

```python
HISTORY_LIMIT = 40
@dataclass(frozen=True)
class HistoryEntry:
    time_label: str      # 'HH:MM:SS'
    label: str           # e.g. '载入原始标注', '编辑分段「long hair」', '查找替换 ×3'
    caption: str
class FileHistory(QObject):
    changed = Signal(str)                       # key
    def seed(self, key: str, caption: str) -> None      # entry '载入原始标注' (on dataset open)
    def push(self, key: str, label: str, caption: str) -> None   # no-op if caption == entries[0].caption
    def entries(self, key: str) -> tuple[HistoryEntry, ...]      # newest first; [0] is current
    def revert_caption(self, key: str, index: int) -> str        # drops entries[:index], returns new current caption
    def clear_keep_current(self, key: str) -> None
```
Revert flow (history section): `new_text = history.revert_caption(key, i)` then
`controller.set_caption_no_history(key, new_text)` — add that internal method (applies
edit without pushing) so revert doesn't duplicate entries; toast `已回退到 HH:MM:SS`.

## Left panel (file_panel.py + thumb_cells.py + thumbnails.py) — [agent A]

Faithful to design: header (folder icon, dataset name bold 12.5, mono path 10 ellipsis,
refresh + open-folder icon buttons), search input with magnifier icon
(placeholder `搜索文件名 / 标签…`), row [view segmented (list/mid/big icons) | spacer |
全选 checkbox | `已选 {n}` | `清除` link], scrollable folder groups, footer stat line +
warn-dot legend `未保存`.

- Folder group header: rotating ▾ (0/-90deg), folder icon, name, `n 张` or `k/n 张` when
  filtered; click toggles (persist `folder_open`). Collapse/expand animated
  (QPropertyAnimation on maximumHeight is acceptable for the grid-rows animation).
- Grid cell: 3:4 thumb, rounded 9px, 2px border (accent when current + soft glow),
  bottom gradient overlay with mono name + `n段`, checkbox 17px top-left
  (semi-transparent surface, accent when checked), warn dirty dot top-right.
  Grid columns: auto-fill minmax(`thumb_min`,1fr) for mid; minmax(150,1fr) for big —
  implement with a reflowing grid layout responding to width.
- List row: checkbox, 34×44 thumb, mono name + dirty dot, caption preview
  (first 4 segments joined ', '; 10.5px text3), `n段` right-aligned.
- Click behaviors (exact): plain=set_current+set_anchor; Shift=select_range_to;
  Alt=deselect; checkbox click=toggle_selected (stopPropagation).
- `没有匹配的文件` empty state.
- `ThumbnailLoader`: `request(key, path, target_h)` → `ready(key, QPixmap)`; LRU ≤512
  entries; QImageReader with scaled read; never blocks GUI thread.

## Middle column (preview_panel.py — agent A; editor_panel.py + editors — agent B)

Header bar (46px): mono filename, meta pill (real `image_meta`), 未保存 pill (warn) when
dirty, multi pill `对比 · 已选 {n} · 切换锁定在选中集` when multi_mode; then undo btn
(tooltip `撤销 (Ctrl+Z)`), divider, prev/next btns (`上一张 (Alt+↑)`/`下一张 (Alt+↓)`),
mono pos label, divider, `复制` outline btn, `保存` accent btn, `全部保存` outline btn.

Preview: single → centered image, fit-or-percent zoom; floating bottom-right zoom pill
(− / label / + / divider / 适应), ZOOM_RANGE/STEP per tokens. multi (≥2 selected) → 2×2
grid of first 4 (contain-fit, name pill bottom-left, `编辑中` accent badge on current,
dirty dot, click→set_current), overflow pill `已选 {n} 张 · 仅显示前 4 张` top-right.

Splitter: horizontal handle (8px, pill grip) between preview and editor;
drag adjusts editor height within EDITOR_H_RANGE; persists `editor_h`.
Use a QSplitter with custom-styled handle, or a manual drag handle — visual parity wins.

Editor area (`editor_panel.py`):
- Mode tabs 胶囊/分句/文本 (segmented buttons per design) + right-aligned mono
  `{n} 字符 · {m} 段` for current.
- Blocks: one per `controller.editor_keys()`; in multi mode each block gets a header row
  (mono name, dirty dot, `编辑中` badge when current — click header to set_current, stat
  `n 字符 · m 段`); border accent when current in multi mode.
- Floating segment toolbar (overlay top-center of editor area, animated show/hide):
  label (`第 n 段` / `插入新段 · 第 n 位`, prefixed `name · ` in multi), buttons:
  分段 / 插入 / | / 翻译 / 重译 / | / 删除 (danger). Visible while a segment edit or
  insert is active. Uses mouse-down (not click) so it fires before editor blur.
  - 分段: split active segment at the inline-editor cursor into two segments
    (strip trailing/leading commas+space around the split like the prototype); warn toast
    `把光标放在要打断的位置` when cursor at edge; `先完成插入内容` if inserting.
  - 插入: commit current edit, open insert editor after it.
  - 翻译 / 重译: translate the inline editor's text via `translate_bridge`
    (async; replaces editor text when done; 重译 requests an alternative — bypass cache).
    Unconfigured → warn toast `未配置翻译 API — 打开 工具 ▸ 设置`.
  - 删除: delete active segment (or cancel insert).
- Chips mode (`chips_editor.py` + `flow_layout.py`): chips = `controller.segments(key)`;
  pill chip (mono 12.5px, hover accent border, × delete on the right), click text → inline
  QLineEdit styled as active chip (auto-width by content), Enter commit / Esc cancel /
  blur commits; commit with commas splits into multiple chips (core edit_chip). Drag
  reorder with drop indicator (left accent bar on hover target); drop on container end →
  move to end. `+ 标签` dashed pill opens insert-at-end editor. Empty state text
  `暂无内容,点击「+ 标签」添加`. All commits via `commit_segments` with labels
  matching the prototype (`编辑分段「x」`,`删除分段「x」`,`插入片段「x」`,`拖拽排序`,`分段`)
  where `「x」` truncates to 12 chars + `…`.
- Sents mode (`sents_editor.py`): same segments, one row per segment: drag handle
  (6-dot), number badge `01`, text box (bg var(--bg), border, radius 9, 1.6 line-height,
  click→multiline inline edit with autosize), delete button. `+ 添加分段` at bottom.
  NOTE: rows are the SAME comma segments as chips (design: `按英文逗号分段`) — not the
  core sentence splitter. Drag reorder with top accent indicator.
- Text mode (`text_editor.py`): full QPlainTextEdit (or QTextEdit) with caption text;
  edits → `set_caption(key, text, '自由编辑')` throttled: push history label on
  focus-out only (live edits update core, history entry on blur — prototype behavior).
  Selection ≥1 char shows action bar above (accent-soft bg): `已选中 n 字符`,
  `替换为…` input, 替换 (accent), 删除 (danger outline), 润色 (outline; local tidy —
  split selection on commas, trim, join ', '). Buttons apply to the selection range with
  labels `替换选中片段`/`删除选中片段`(collapse `,\s*,`→`,`, strip leading comma)/`润色选中片段`.
- Bottom hint line (11px text3), exact texts:
  chips: `拖动胶囊排序 · 点击编辑,悬浮工具栏可分段 / 插入 / 翻译 / 重译 · × 删除 · 按英文逗号分段`
  sents: `按英文逗号分段 · 拖动手柄排序 · 点击编辑,悬浮工具栏可分段 / 插入 / 翻译 / 重译`
  text:  `拖选文字后可替换 / 删除 / 润色选中片段 · 失焦时自动记录历史`

## Right panel (tools_panel.py + sections/ + collapsible.py + toast.py + translate_bridge.py + settings_dialog.py) — [agent C]

Panel header: `修改工具` bold + subtitle `对当前文件或批量范围应用修改`. Four
`CollapsibleSection`s (surface card, 11px radius, header icon+title+rotating ▾, animated
expand/collapse; open state persisted in UISettings.sections).

Shared scope segmented control (`当前` / `选中 {n}` / `全部 {N}`) — component in
tools_panel.py; live counts from controller signals.

1. **查找替换** (`fr`): find input (mono, placeholder `查找内容,如: long hair`), replace
   input (placeholder `替换为 (留空则删除)`), toggle chips `Aa 区分大小写` + `W 整词`
   (tooltip explains whole-word: `hair` won't hit `hairband`), live info on its own
   line right-aligned (`输入查找内容以预览匹配` → `匹配 {m} 处 · {k} 个文件` accent → `无匹配`),
   scope, accent button `全部替换` → `controller.replace_all` (async, snapshot; toasts
   per contract). Info updates on text/scope/case/word/selection/caption changes.
   Matching is substring-level over the WHOLE caption (words inside sentences match
   too — not per comma segment); the 整词 chip maps to core `whole_word` (`\b` bounds).
2. **前缀 / 后缀** (`ps`): input (placeholder `如: aoba, masterpiece`), position
   segmented `加到开头`/`加到结尾`, toggle chip `作为独立标签 (自动加逗号)` (default ON),
   scope, accent button `应用` → `controller.apply_prefix_suffix`.
3. **翻译对照** (`tr`): section title `翻译对照` (NO 模拟 badge — real LLM). Rows for
   current file's segments: src (12px) over dst (mono 11px; accent when ok, text3 + note
   when pending/unavailable), swap button (tooltip `用译文替换原文`) enabled only when a
   translation is ready → replaces that segment (`commit_segments`, label `翻译替换`,
   toast `已替换为译文`). Bottom button `中文全部转为英文` → translate every CJK segment
   of the current file and replace successes (label `中文全部转为英文`); toasts
   `当前文件没有中文内容` / result summary. Unconfigured state: rows show
   `(未配置翻译 API)`, buttons disabled, hint link `打开设置` → settings dialog.
   `translate_bridge.py`:
   ```python
   class TranslateBridge(QObject):
       segment_ready = Signal(str, str, str, bool)   # key, source_text, result, ok
       all_done = Signal(str, int, int)              # key, succeeded, failed
       def __init__(self, controller: AppController) -> None
       def configured(self) -> bool
       def request(self, key: str, text: str, *, fresh: bool = False) -> None  # async single segment
       def translate_all_cjk(self, key: str, texts: Sequence[str]) -> None
   ```
   Direction per segment via core `detect_direction`; cache per (source text) via core
   TranslationCache keyed `key::index-free` (use text-hash key so identical tags share).
   `fresh=True` bypasses cache (重译).
4. **历史记录** (`hist`): header count `{n} 条`; rows (max-height scroll): mono time +
   `{len} 字符`, label line, `当前` accent pill for [0], `回退` button for others (→
   revert flow in history_model contract); accent-tinted border/bg for the current row.
   Bottom `清空历史` (danger hover) → `clear_keep_current` + toast `已清空历史 (保留当前状态)`.

`toast.py`: `ToastOverlay(parent)` bottom-center stack, pill (surface bg, border,
colored 8px dot: ok/warn/danger/accent-for-info), auto-dismiss 2600 ms, slide/fade in;
`show_toast(text, kind)`; connected to `controller.toast_requested`.

`settings_dialog.py` (addition — required for translation): minimal modal `设置`,
fields: 接口类型 (openai/anthropic/ollama), Base URL, API Key (masked QLineEdit
Password), 文本模型, 视觉模型(可选), plus 测试连接 button (async `client.test_connection`,
result toast) and 保存 → writes core `AppConfig` (single profile named `default`,
active) via `save_config(app_data_dir()/"config.json", ...)`; controller reloads
translator. Uses tokens/QSS only.

## Title bar / status bar / main window — [integrator]

- Frameless window (`Qt.FramelessWindowHint`), MIN_WINDOW 1360×760. Title bar 40px:
  logo (paint the design's rounded-square + lines mark with QPainter, accent bg),
  `NLapt` bold + mono `自然语言标注工具 v0.9` (version from `nlapt.__version__` — render
  as `自然语言标注工具 v{__version__}`), menu labels 文件/编辑/视图/工具/帮助 as real
  QMenus: 文件[打开文件夹…, 刷新, 保存 Ctrl+S, 全部保存, 退出], 编辑[撤销 Ctrl+Z, 复制标注],
  视图[视图模式 list/mid/big, 主题▸5], 工具[设置…], 帮助[关于 (version dialog)].
  Right: theme button (`{themeName}` + palette icon + ▾) opening the theme popup
  (230px card, 5 rows with 3 color swatch dots (bg/surface/accent) + name + check for
  active), divider, min/max/close buttons (44px wide, close hovers #E81123 — allowed
  as WINDOWS_CLOSE_HOVER token constant, not hardcoded inline).
- Drag on title bar empty space → `windowHandle().startSystemMove()`; double-click →
  max/restore; edge resize via 6px margin hit test + `startSystemResize`.
- Status bar 26px: left mono `{path} · UTF-8 · {n} 个标注文件`; right hints:
  `Ctrl+S 保存` `Alt+↑/↓ 切换` `Shift+点击 范围选` `Alt+点击 取消选` `模式 · {模式名}` `主题 · {主题名}`.
- Shortcuts (QShortcut, window-level): Ctrl+S save_current; Ctrl+Shift+S save_all;
  Ctrl+Z undo_current only when focus is NOT in a text input (match prototype);
  Alt+Up/Alt+Down nav(-1/+1); Esc closes popups/cancels inline edit (handled by editors).
- `__main__.py`: `main()` — QApplication, AA_* DPI already default in Qt6, app name/org,
  `configure_logging(app_data_dir()/'logs')`, `install_crash_handler`, load UISettings +
  core AppConfig, build controller, ThemeManager.apply, MainWindow, open `last_root` if
  it still exists. `if __name__ == '__main__': raise SystemExit(main())`.
- On window close: `controller.close()` (session-persist drafts — crash-safe; txt files
  are written only by explicit 保存/全部保存, matching the design).

## Packaging reserve — [integrator]

- `pyproject.toml`: add `[project.gui-scripts] nlapt-gui = "nlapt_gui.__main__:main"`,
  optional dep group `gui = ["PySide6"]`; keep everything else.
- `packaging/nlapt.spec`: PyInstaller spec — onedir, windowed (`console=False`), name
  `NLapt`, entry `nlapt_gui/__main__.py`, excludes tests/docs, note where to add
  `icon='packaging/icon.ico'` when an icon exists. Must import cleanly (spec is Python).
- `packaging/build.ps1`: check pyinstaller installed (friendly message if not:
  `pip install pyinstaller`), run `python -m PyInstaller packaging/nlapt.spec --noconfirm`,
  print output path.
- `packaging/README.md`: 中文 build guide (install, build, output layout, icon swap,
  version bump single-sourced from `nlapt.__version__`).
- Runtime code must already be packaging-safe: all paths via `resources.py`, settings/
  logs under `app_data_dir()`, no `__file__`-relative data files.

## Deliberate deviations from the prototype (document in README, do not "fix")

1. Real datasets from disk (core scanner: jpg/jpeg/png/webp/bmp, kohya subdirs) replace
   the 12 hardcoded samples; folder groups = relative directories (root files → `根目录`).
2. Segments split on half- AND full-width commas (core `split_chips`) — prototype was
   ASCII-only.
3. 翻译对照/悬浮翻译 use the configured LLM (core Translator + cache); prototype used a
   mock dictionary. Unconfigured → guided empty state; `重译` = fresh request.
4. 前缀/后缀 in 独立标签 mode uses core skip-if-present boundary logic (spec 7.2) so
   trigger words are not duplicated; toast reports actually-changed file count.
5. 查找替换批量 runs through the core batch engine → automatic zip snapshot + operation
   log (data safety per spec 2.4/9); plain-text (regex-escaped) matching like the prototype.
6. Explicit-save model kept; additionally unsaved drafts survive crashes via core session
   restore, and closing the app persists the session.
7. Minimal 设置 dialog added (LLM profile) — required to make translation functional.
8. Window controls are real; frameless drag/resize via Qt system move/resize.
9. Meta pill shows the real resolution/format/size instead of hardcoded `480 × 640 · PNG`.
10. Menus are functional (see title bar spec) instead of toast placeholders.

## v1.1 fix round — confirmed review findings

The multi-lens review + adversarial verification surfaced 16 confirmed findings
(deduped to 19 fixes across 11 files); all are fixed with regression tests in
`tests/gui/test_gui_fixes.py`. Highlights:

- **Data safety (critical)**: `AppController.refresh()` persists the session first and
  `_finish_open` preserves the labeled history across a same-root rescan — unsaved edits
  and history survive 刷新 instead of being silently reverted.
- **Async I/O**: `save_current` / `save_all` now run the atomic disk writes on the pool
  (no GUI-thread freeze); `busy_changed` is consumed by `MainWindow._on_busy_changed` to
  disable 保存/全部保存/全部替换/应用 while a batch or save is in flight, and `_run_batch`
  rejects overlapping batches (`TOAST_BUSY`).
- **Rescan safety**: controller reads route through the guarded `record()` and mutators
  no-op while `_rescanning`, so a keystroke during an async reload can't raise `KeyError`.
- **Undo**: `undo_current` drops the newest history entry (like the prototype's
  `history.slice(1)`) instead of appending an 撤销 row — `FileHistory.drop_newest`.
- **Editor**: `EditorPanel._rebuild(force=True)` on `dataset_opened` drops stale segment
  widgets; the floating-toolbar translate reply is targeted by
  (editor, edit_index, insert_pos) so a late reply never lands in a different segment;
  `SegmentEditorBase.start_edit` re-resolves the clicked index after a commit shifts it.
- **Translation**: the 翻译对照 section is inert until its card is expanded
  (`set_live`, driven by `ToolsPanel`), caption edits are debounced (400 ms), and a
  `中文全部转为英文` batch commits to its own key even after the user navigates away.
- **Config seam (N)**: added public `NLaptApp.reload_config` + `AppController.reload_config`;
  `SettingsDialog` no longer pokes `app._config`, and it refuses to overwrite a config it
  could not read (typed errors are not masked).
- **Fidelity**: preview zoom resets to 适应 on image change (and steps from 100%); view
  tooltips are 详细列表/中图网格/大图网格; toggle chips use the design's 6px radius; window
  controls consume their press and the title bar only drags on empty space.

## v1.2 fix round — issues found in real use

Six issues reported after running the app were fixed (tests under
`tests/gui/test_gui_fix_*.py`; full suite 1427 green, 94% coverage):

1. **Editor multi-select highlight**: the current block's accent border was an
   *unscoped* stylesheet that Qt cascaded onto every chip. `_BLOCK_CURRENT_QSS` is now a
   `EditorBlock#editorBlock { ... }` type#id rule — only the block frame gets the border;
   chips keep their own, and the `编辑中` badge is the multi-mode "current" cue.
2. **Right panel** (revised in v1.3.1): `CollapsibleSection` rebuilt around two invariants.
   (a) The header has a FIXED height (`HEADER_HEIGHT` = 40) — a QPushButton hosting a child
   layout reports a size hint from its own empty text, so a stylesheet change had squashed
   the header and clipped every title; a fixed height is immune to that. (b) The body lives
   in a section-owned frameless `QScrollArea` whose height is the user value clamped to
   `[120, 720]` — deliberately independent of the content's minimum size, so dragging the
   bottom `ResizeGrip` UP shrinks into a scrollbar and DOWN reveals more (inner row lists
   get an Expanding policy). The old clamp against `content.minimumSizeHint()` made
   dragging a no-op for form sections. Natural height (no user value) follows the content
   hint via a `LayoutRequest` event filter. Heights persist per-section in
   `app_data_dir()/section_heights.json`.
3. **Preview**: mouse-wheel over the single image zooms (`_SingleView.wheelEvent`, ignored
   in compare mode); double-clicking a compare-grid cell opens it as the standalone large
   single view (`set_current` + `clear_selection`). Also added a shiboken lifetime guard on
   the async preview-load reply so a torn-down panel can't touch a deleted view.
   *(v1.3.1)* `ZOOM_RANGE` widened from the prototype's 40–260 to the spec-5 **10–800**,
   and `_clamp_zoom`'s floor additionally drops to `fit_percent()` — zooming out always
   reaches a whole-image view even when the image's fit scale is below 10%.
4. **Help menu**: `帮助` now has `快捷键` (real shortcut list), `使用说明` (three-column
   workflow + save/snapshot note), and a richer `关于`.
5. **Translation providers** (`nlapt/llm/web_translate.py` + `nlapt_gui/translate_config.py`):
   besides the LLM translator (default), built-in **Google 翻译** (免费, unofficial keyless
   endpoint, best-effort/may rate-limit), **百度翻译** and **DeepL** (both need a key). The
   settings dialog's `翻译服务` area shows a Chinese registration note + URL per provider
   (百度: https://fanyi-api.baidu.com ; DeepL: https://www.deepl.com/pro-api ). Provider +
   keys persist in `app_data_dir()/translate.json`; `TranslateBridge` routes to the selected
   provider (network on the pool, text-hash cache + `fresh=True` preserved).
6. **Title bar**: app name + version are one baseline-aligned group vertically centered in
   the 40px bar (fixes the top-left misalignment).

## v1.3 fix round — polish from real use

Five more issues fixed (tests under `tests/gui/test_gui_fix_*`; full suite 1507 green, 94%):

1. **Window resize** (revised in v1.3.1): the first `WM_NCHITTEST` approach broke clicks —
   on a HiDPI display the physical hit-test coords fed to a logical `mapFromGlobal` marked
   the right/bottom thirds of the window as a non-client resize border, so whole panels
   became unclickable and (lacking `WS_THICKFRAME`) still didn't resize. Replaced with Qt's
   cross-platform `startSystemResize`: the app event filter shows a resize cursor when
   hovering within `EDGE_MARGIN_PX` (8px) of an edge and starts the OS resize loop on an
   edge press. It never touches `WM_NCHITTEST`, so interior pixels can never become
   non-client — clicks are always delivered.
4. **Draggable columns**: the body is a horizontal `QSplitter` (`_BodySplitter`) with two
   draggable dividers; panel width locks are released so the user can resize left/right,
   and sizes persist to `app_data_dir()/body_splitter.json`. The vertical editor-height
   handle still works inside the middle column.
2. **Preview comfort**: cursor-anchored wheel zoom, drag-to-pan when zoomed
   (`single_pannable`), double-click toggles 适应↔100% (`toggle_actual_size`), smoother
   stepping, and a skip-safe cross-fade on image change (the shiboken async-reply guard
   stays).
3. **Settings + logo**: the 翻译服务 provider dropdown uses `QFormLayout.setRowVisible` so
   switching providers fully collapses the other rows (no stray gaps/lines); the bare HLine
   became a themed 1px divider. New `nlapt_gui/theme/logo.py` provides a crisp theme-aware
   vector mark — `LogoWidget` in the title bar and `make_app_icon()` as the window/taskbar
   icon (refreshed on theme change).
5. **Aesthetics + animation**: global `qss.py` polish (themed `QComboBox` + its popup, slim
   rounded scrollbars, `QMenu`/`QToolTip`/`QMessageBox`, focus rings, hover/pressed states —
   cascades to every widget, in all 5 themes), an animated theme transition in
   `ThemeManager`, and a reusable `nlapt_gui/anim.py` (fade/pop helpers gated by
   `set_animations_enabled`, off in tests). No hardcoded colors; the 5 theme token sets are
   unchanged.

## v1.3.2 — resize-crash purge + real status indicators

- **Resize crash (native)**: resizing the window segfaulted with no Python traceback. Root
  cause: `QGraphicsOpacityEffect` on live widgets — effect buffers re-render during native
  resizes and hard-crash Qt. All persistent/long-lived effects were purged: the window
  fade-in now animates `windowOpacity`, the preview image fade is painter-level
  (`_fade_alpha` + `painter.setOpacity`), the floating segment toolbar shows/hides
  instantly, and toast pills appear without an opacity fade. Drag-time effects in the
  chip/sentence editors remain (the mouse is captured during drags — no resize possible).
  A parametrized test asserts no `QGraphicsOpacityEffect(` construction in any
  resize-sensitive widget module.
- **Real save state (spec 2.3)**: `AppController.save_state_changed` ("saving"/"saved"/
  "failed") + the StatusBar renders 已保存 / 保存中… / 保存失败 / 未保存 n (colored by
  tokens); the file-panel footer 未保存 legend is now a live indicator (hidden when clean,
  `未保存 {n}` when dirty). No prototype placeholders remain (menus/help/status all real).
- **Bottom-left clock**: live `yyyy-MM-dd HH:mm:ss` label (QTimer, 1s) at the far left of
  the status bar.

## v1.4 — history window, translate preview, scoped translation

- **历史记录 (cursor model)**: `FileHistory` is cursor-based — 回退/跳转 moves a per-file
  cursor without deleting anything, so the panel can jump BOTH ways; the window is bounded
  to `HISTORY_KEEP_NEWER`/`HISTORY_KEEP_OLDER` = **5 above + 5 below** the cursor (user
  requirement 保留上下 5 条内). A new edit from the past cuts the newer branch first
  (classic undo semantics). `Ctrl+Z` = `step_older` (cursor move, no deletion); the 当前
  pill follows the cursor and every other row gets 回退.
- **工作区翻译先预览**: the floating-toolbar 翻译/重译 no longer writes into the inline
  editor. The result appears in a `TranslationPreview` overlay (below the toolbar, text
  selectable) with **替换** (apply into the still-open editor) and **关闭** (view-only).
  Replies still validate their exact request target; failures toast `翻译失败: {msg}`.
- **指定范围翻译**: the 翻译对照 section gained a `ScopeSelector` (当前 / 选中 n / 全部 N).
  `中文全部转为英文` now processes every CJK-containing file in the scope one by one
  (button shows `翻译中 i/n`, re-entry locked; the bridge's text-hash cache dedupes
  repeated tags across files), commits per file as its batch completes, and reports an
  aggregate toast.

## v1.5 — caption workspace, select-then-edit, tabbed settings, UI polish

Four user-requested modules (tests: `test_gui_caption_bar / _vision_bridge /
_prompt_store / _prompt_editor / _dialogs / _color_dialog / _settings_tabs /
_selection_toolbar`, `test_llm_translate_to / _model_list`; full suite 1664 green, 94%):

1. **标注工作区 (module 1)** — the editor tab row hosts a `CaptionBar`
   (`widgets/caption_bar.py`) with caption-level actions on the current file:
   - **翻译 ▾**: whole-caption translation into 中文 / English / 日本語. Core
     gains target-language support: `translate.py` `TARGET_LANGS/LANG_LABELS` +
     `Translator.translate_to`, `translate_to_(zh|en|ja)` templates, and
     `translate_to(text, lang)` on all three web providers (auto source).
     `TranslateBridge.request_to` + `target_ready` carry it async with a
     per-(text, lang) cache; the provider follows 设置 ▸ 翻译服务 (LLM or API).
   - **重译**: re-infers the image through the configured **视觉模型**
     (`vision_bridge.py` over `AppController.make_vision_captioner_or_none`,
     which prepares the image and sends the 设置 ▸ 提示词 system/user prompts).
   - **删除**: clears the caption after a centered confirm (undoable, labeled).
   Results NEVER write silently — they open the shared `TranslationPreview`
   (moved to `caption_bar.py`) with 替换 / 关闭.
2. **Select-then-edit + anchored toolbar (module 2)** — `SegmentEditorBase`
   gains a selection state: the first click on a chip/sentence row SELECTS it
   (accent highlight via `chipSelected` / `selected` QSS states), a second
   click opens the inline editor; segment presses are consumed so background
   clicks (which clear the selection) never misfire. The floating toolbar now
   anchors directly ABOVE the active segment (`active_target_widget` +
   `EditorPanel._position_toolbar`, flipping below near the top edge,
   clamped, scroll-tracked) instead of the fixed top-center (kept as the
   fallback); with a selection its actions work without an open editor
   (删除/插入 immediately, 分段 enters edit + cursor hint, 翻译/重译 preview
   then replace the segment via `apply_translation`, label 翻译替换).
3. **Tabbed 设置 (module 3)** — `SettingsDialog` became a 4-tab
   `CenteredDialog`: **翻译服务** (unchanged provider/keys/测试翻译),
   **LLM 设置** (接口/URL/Key + 统一文本/视觉模型 checkbox with multimodal
   hint, split 文本模型/视觉模型 otherwise, 并发请求数 spin persisted into
   `RequestControl.concurrency`, async **获取模型** via
   `nlapt/llm/model_list.list_models` + `ModelPickerDialog` with a 视觉
   name-heuristic tag, 测试连接), **提示词** (`PromptsTab`: template dropdown
   with 新建自定义/保存模板/删除, empty built-in 默认, system + 用户提示词
   editors, 导出当前 (.txt) / 导出全部自定义 (.json); persisted to
   `vision_prompts.json` via `prompt_store.py`; consumed by 重译), and
   **本地推理** (placeholder for the future in-app local model module).
   帮助 ▸ 快捷键 was removed (the status bar shows live hints).
4. **UI & motion (module 4)** — `widgets/dialogs.py` provides `CenteredDialog`
   / `FadingMessageBox` / `show_message` / `ask_confirm`: every sub-window
   opens CENTERED on the app and fades in/out via `windowOpacity` (never
   QGraphicsOpacityEffect — resize-crash rule), skip-safe when animations are
   disabled (now globally off in tests via a conftest autouse fixture).
   色彩设置 (`widgets/color_dialog.py`, opened from 视图 ▸ 色彩设置… or the
   theme popup's 自定义色彩… row) adds free accent customization with a
   QColorDialog on top of the presets, applied live through ThemeManager.
   The main window fades in on startup (240 ms) and fades out on close; the
   title bar's left inset aligns with the panel headers (16px grid). The app
   icon is `packaging/icon.ico` (generated by `packaging/make_icon.py`,
   swappable), wired into the PyInstaller spec, runtime
   (`theme/logo.load_app_icon` falls back to the painted mark) and a Windows
   `AppUserModelID` for correct taskbar identity.

## v1.6 — 本地推理 tab becomes real (local inference module)

The v1.5 本地推理 placeholder was replaced by a working module (core side:
`nlapt/local`, see ARCHITECTURE v1.6; tests `tests/local/*`,
`test_gui_local_bridge.py`, `test_gui_local_tab.py`):

- `nlapt_gui/local_bridge.py` — `LocalBridge(QObject)`: owns the persisted
  `LocalSettings`, hardware detection, downloads and server lifecycle on the
  worker pool; signals `hardware_ready` / `download_progress` (throttled to
  8 MB steps) / `download_finished` / `server_changed`. Every async reply is
  guarded with a shiboken validity check. The `LocalServerManager` is a
  process-wide singleton (`get_server_manager()`, stopped via `atexit`): a
  started llama-server survives dialog re-opens and dies with the app.
- `nlapt_gui/widgets/local_tab.py` — `LocalTab`: hardware summary line
  (lazy first-show detection + 重新检测), catalog `QTreeWidget`
  大系列 → 小系列 → 量化档 with 体积 / 热度 / 兼容性 columns — verdicts are
  text markers (✓ ◐ ▢ ✗ ?) with explanation tooltips, deliberately
  colorless (theme-safe, keeps the no-hex rule trivial); a detail line with
  the memory breakdown + budgets + verdict sentence; the action row
  下载(断点续传 / 取消 + progress bar)/ 打开模型页 / 启动本地服务(停止)/
  设为当前模型; and the runtime form 模型目录 / llama-server 路径 /
  上下文长度 / GPU 层数(-1=自动)/ 线程(0=自动)/ 并发请求数 / 端口
  (ranges come from `nlapt.local.settings`). Downloads write into
  per-family subdirectories of the models dir.
- **设为当前模型** registers/updates the `local` profile (api_type
  `openai`, base_url `http://127.0.0.1:{port}/v1`, model = family id,
  `vision_model` only for mmproj-carrying families), sets it active, wires
  `request.concurrency = parallel` and calls `controller.reload_config` —
  the settings dialog's sanctioned config-IO exception explicitly extends
  to this tab.
- `SettingsDialog.save()` additionally calls `local_tab.persist()`
  (self-toasting on failure); the placeholder constants and
  `test_local_tab_is_placeholder` were removed
  (`test_local_tab_is_real` + a local-settings save round-trip replace it).

## v1.6.1 — feedback round (window states, landscape 本地推理, menus)

Four user-reported issues fixed (suite 1838 green):

1. **Maximize/restore**: `MainWindow.toggle_max_restore` checks the RAW
   window state (`WindowMaximized | WindowFullScreen`) — the old
   `isMaximized()`-only branch could never leave a FULLSCREEN state
   (全屏后无法恢复小窗). Restore runs `_ensure_normal_fits_screen` (on
   small logical screens the fixed minimum equalled the screen, making
   restore a visual no-op: minimum is lowered and an 86%-of-screen window
   is centered). Both direction changes run through a short
   `windowOpacity` dip (`_animate_state_switch`, skip-safe). The title
   bar delegates to the window's toggle; dragging starts the system move
   only after a real drag distance (a bare press used to enter the OS
   move loop and swallow double-clicks), and dragging a zoomed window
   un-zooms it first.
2. **本地推理 tab is landscape** (tree left / detail + actions + runtime
   settings right, tab min 840×520 so the dialog opens wide and 模型列
   no longer truncates). 兼容性 became 「能否运行」 with the five-level
   Chinese `RunGrade` (轻松运行 / 流畅运行 / 可以运行 / 勉强能跑 /
   跑不动 — headroom factor 1.3 splits PERFECT/SMOOTH and OK/BARELY);
   series/family names and grade cells are bold, the detail line leads
   with the bold grade word. 下载目录 defaults INSIDE the app
   (`default_models_dir()` → repo root or exe dir `models/`, git-ignored)
   and `LocalSettings.extra_dirs` adds reuse directories: files are FOUND
   across primary+extras (`find_model_file`/`find_mmproj_file`),
   downloads skip anything a reuse dir already provides, and the server
   spec uses the found paths.
3. **提示词共用**: the 提示词 tab states explicitly that one prompt set
   backs every LLM (online AND local) — behavior was already global via
   `vision_prompts.json` + the active profile.
4. **Menus**: 设置 is a TOP-LEVEL title-bar entry (`settings_button`,
   opens the dialog directly); the 工具 menu stays as a future home with
   a disabled 「更多工具(规划中)」 placeholder. `action_settings`
   remains for programmatic callers.

## v1.7 — 推标(LLM / 本地)、文件夹多选、内置 llama.cpp 运行时

User feature round (5 requests) — all additive:

1. **文件夹右键推标 + ALL 行 + 文件夹多选** (`file_panel.py`): every
   `_FolderGroup` header carries a tri-state `QCheckBox` (click selects /
   clears the whole folder via `AppController.set_folder_selected`;
   coverage from `folder_selection_state` → all/some/none) and a
   `CustomContextMenu`; an `_AllRow` ("ALL" + total count + checkbox)
   sits permanently above the groups (`all_row`, groups insert after it).
   Right-click (folder header or ALL row) → 推标 menu built by
   `infer_menu_actions(folder)` (the test seam): 此文件夹 / 已选 / 全部 ×
   LLM / 本地模型, with `取消当前推标` replacing everything while
   `controller.batch_running()`. Choosing an action runs `ask_confirm`
   (captions get overwritten; snapshot + 历史回滚 mentioned) and emits
   `infer_requested(keys_tuple, engine)`; MainWindow routes it to
   `vision_bridge.request_batch`.
2. **Caption bar 两个推理按钮** (`caption_bar.py`): 重译 became
   `LLM 推理` (`reinfer_btn`) + `本地推理` (`local_infer_btn`), both
   through `_on_infer(engine)`; pending state is `(key, engine)`, busy
   text `推理中…` on the active engine's button only, per-engine
   unconfigured toasts (`TOAST_VISION_UNCONFIGURED` /
   `TOAST_LOCAL_UNCONFIGURED`), history labels `推理(LLM)` / `推理(本地)`,
   preview title `{name} · 推理结果`. `configured(engine)` is duck-called
   with a no-arg fallback for legacy bridges.
3. **VisionBridge engines** (`vision_bridge.py`): `request(key,
   engine=ENGINE_LLM)` and `configured(engine)`; `ENGINE_LLM`/`ENGINE_LOCAL`
   live in `prompt_store` (re-exported). The local captioner comes from
   `local_bridge.make_local_vision_captioner` (test seam:
   `local_captioner_factory` ctor arg; typed `NLaptError` → emitted as a
   failed `caption_ready`). `request_batch(keys, engine)` resolves the
   engine captioner + per-engine prompts + concurrency (LLM →
   `config.request.concurrency`, local → `LocalSettings.parallel`), then
   calls `AppController.run_caption_batch`; for a local batch it records
   whether the server was already running and stops it AFTER the whole
   batch only if the batch started it (加载一次、全部推完再卸载).
4. **AppController batch surface** (`controller.py`):
   `run_caption_batch(keys, caption_fn, *, description, history_label,
   engine, concurrency, on_finished) -> bool` (busy-guarded like
   `_run_batch`; pushes per-file labeled history for changed keys; toasts
   开始/完成/取消; emits new `batch_progress(description, done, total)`
   from the worker thread — Qt queues delivery), `cancel_batch()`
   (cooperative `BatchController.cancel`), `batch_running()`, plus folder
   selection helpers `folder_keys` / `folder_selection_state` /
   `set_folder_selected`.
5. **统一提示词管线 + 本地覆写** (`prompt_store.py`, `prompt_editor.py`):
   `VisionPrompts` gains `local_unified=True`, `local_system`,
   `local_user_prompt` and engine-aware `system_text_for(engine)` /
   `user_prompt_for(engine)` (local fallback to `DEFAULT_USER_PROMPT`
   when its user prompt is empty). The 提示词 tab adds a
   `本地推理共用上方提示词(统一管线)` checkbox; unchecking reveals the
   local system/user editors (`local_system_edit`, `local_user_edit`).
6. **就绪即可用 runtime** (`local_bridge.py`, `local_tab.py`): settings
   path helpers went module-level (`local_settings_path`,
   `models_dir_for`, `search_dirs`, `find_model_file_in`,
   `find_mmproj_file_in`, `is_downloaded_in`, `build_spec_for`) so
   inference works outside the settings dialog; `runtime_base_dir()` =
   `app_data_dir()/runtime`; `resolve_server_path` = manual setting →
   extracted runtime → "". `pending_runtime_asset` folds the pinned
   llama.cpp archive into `start_download` (progress totals include it),
   and `start_server` auto-provisions via `ensure_runtime` on the pool.
   `resolve_local_target(settings=None, require_vision=True)` validates
   selection/download/vision/runtime with actionable Chinese
   `LocalInferenceError`s; `make_local_vision_captioner(image_max_edge=)`
   returns the blocking `(image_path, system, user) -> caption` callable
   (ensure runtime → `manager.ensure(spec)` → OpenAI-compatible request,
   `LOCAL_VISION_TIMEOUT_SECONDS` 300s). The 本地推理 tab shows an
   auto placeholder for llama-server, only demands a manual path when the
   platform has no pinned runtime, and its hint explains 就绪即可用.
7. **StatusBar** shows `batch_label` (`推标(...) done/total`) driven by
   `batch_progress`, hidden on `batch_finished`.
8. **Auto GPU layers fix** (user report: llama-server 退出码 1 / OOM):
   `local_bridge.prepare_launch_spec(spec, family, quant)` is the single
   blocking pre-launch step (worker thread only) — provisions the runtime
   when `server_path` is empty AND resolves 自动 (-1) GPU layers via
   `detect_hardware` + `gguf.read_block_count` + `advisor.auto_gpu_layers`.
   Resolved counts are cached per (model_path, context_length) under a
   lock so every caller (设置 ▸ 启动本地服务, single 推理, batch workers)
   launches an IDENTICAL spec and `manager.ensure()` keeps reusing the
   running server (fluctuating free-VRAM probes must not restart it
   mid-batch). Manual `gpu_layers >= 0` passes through untouched, no
   probe. Tests that reach `start_server` set explicit gpu_layers (or
   patch `local_bridge.detect_hardware`) to stay hermetic.
9. **Per-slot context + no-thinking launch** (user report: 推理返回空文本):
   `_auto_gpu_layers_for` budgets KV with `context_length × parallel`
   (cache key includes the total), matching `build_server_args`' new
   `-c` semantics; the 本地推理 tab's 「能否运行」/明细 estimates use
   `_estimate_context()` (ctx × parallel, recomputed when either spin
   changes) and the catalog hint says so.
10. **Preview clamp fix** (same round, user report 推理结果无法应用):
   `TranslationPreview` hosts its body in a `QScrollArea` (`body_scroll`)
   and gains `fit_within(max_w, max_h)` — deterministic manual sizing
   (label `heightForWidth` at the real inner width + chrome) so the
   title and 替换/关闭 buttons ALWAYS stay inside the host while a long
   inference result scrolls inside the card. Both positioners clamp
   before anchoring: `CaptionBar.reposition_overlay` (host minus
   `overlay_top`/margins) and `EditorPanel._position_preview` (panel
   minus toolbar gaps).

## v1.7.1 — 推标进度窗口(实时进度 / 速度 / 预计剩余)

User request: 推标(LLM / 本地)开始时弹出独立窗口,实时显示进度条、张数、
速度与预计剩余时间,风格与全局主题一致。Additive:

1. **AppController**: new `batch_started(description, total)` signal,
   emitted on the GUI thread by `run_caption_batch` right after the busy
   flip / started toast — the progress window's show trigger
   (`batch_progress` alone could arrive only after the first slow item).
   Text-op batches (`_run_batch`) do NOT emit it.
2. **BatchProgressDialog** (`widgets/batch_progress_dialog.py`): non-modal
   `CenteredDialog` owned by MainWindow (`batch_progress_dialog`),
   self-wired to controller signals. `batch_started` resets state and
   shows (each batch re-opens centered with the standard fade via the new
   `_FadeMixin.prepare_reshow()`); `batch_progress` updates bar / count /
   percent and recomputes speed + ETA; an `_active` flag makes text-op
   `batch_finished` and late queued progress events no-ops. Header = batch
   description + right-aligned mono percent; slim themed `QProgressBar`
   (`BAR_HEIGHT_PX` 8, text hidden); stat grid 进度 / 速度(`{rate:.1f}
   张/分`, overall average done÷elapsed) / 已用时间 / 预计剩余(both
   `format_duration`: `mm:ss`, `h:mm:ss` from 1h; placeholder `—` until
   the first item lands). Buttons: 后台运行 (`close()` — the batch keeps
   running, the status bar still shows progress; the window returns on the
   next batch) and 取消推标 (`controller.cancel_batch()`, flips to
   disabled 正在取消…, re-armed on the next `batch_started`). Elapsed/ETA
   tick from a 1s QTimer; wall time comes from an injectable monotonic
   `clock` ctor arg and `refresh_stats()` is public (the test seam — tests
   advance the fake clock and call it directly).
3. **Theme QSS**: global `QProgressBar` styling (surface2 trough, 1px bd
   border, radius 4, accent chunk radius 3, centered text2 10.5px text) —
   the 本地推理 download bar inherits it unchanged.

## v1.8 — Florence-2 PromptGen 打标模型(指令模式设置)

User request: 收录 Florence-2 PromptGen v2.0 并为其特有的指令提示
(`<GENERATE_TAGS>` 等)提供一个设置。Additive:

1. **LocalTab**: new 指令模式 form row — `florence_task_combo`
   (`QComboBox`, entries from `FLORENCE_TASK_LABELS`, token in userData) —
   the FIRST form row, visible only while the selected family's engine is
   `ENGINE_FLORENCE` (`QFormLayout.setRowVisible`). `persist()` writes
   `florence_task=combo.currentData()`; prefill restores it. For Florence
   selections the server button is enabled only to STOP an already-running
   llama server (tooltip `TIP_FLORENCE_NO_SERVER`; `_on_server_clicked`
   guard toasts it) and 设为当前模型 is disabled
   (`TIP_FLORENCE_NO_APPLY` — the model cannot serve translate/rewrite).
   The 体积 column shows quant + extra_files total; `SERVER_HINT` gained a
   Florence exception sentence.
2. **local_bridge**: `is_downloaded_in` / `start_download` cover
   `family.extra_files` (same resumable download + SHA256 verify; the
   llama.cpp runtime is never provisioned for Florence families);
   `find_model_file_in` doubles for extra files (basename in family dir,
   reuse dirs honored). `resolve_local_target` skips the llama-runtime
   platform check for Florence. `make_local_vision_captioner` branches:
   Florence → process-wide `get_florence_engine(florence_files_for(...))`
   singleton (replaced when the resolved file set changes) captioning
   in-process with the persisted `settings.florence_task`; the
   system/user prompts from prompt_store do NOT apply. llama path
   unchanged. VisionBridge / caption batches need no changes — the
   captioner factory hides the engine; the post-batch server stop is a
   no-op for Florence (sessions stay warm until the process exits or the
   file set changes).

## v1.9 — 推标右键菜单按目标分作用域 + 未标注推理

User request: 右键菜单只显示与点击目标相关的作用域;多选图片时提供
推理全部;文件夹菜单新增"推理未标注"。Behavior change in
`file_panel.py` + one `AppController` helper:

1. **AppController**: new `unlabeled_keys(keys) -> tuple[str, ...]` —
   subset of `keys` whose `record(key).state is CaptionState.UNLABELED`
   (empty body text), input order preserved.
2. **FilePanel menu seam**: `infer_menu_actions(folder, image=None)` /
   `show_infer_menu(folder, widget, pos, image=None)`. The menu is scoped
   to the right-click target (取消当前推标 still replaces everything while
   `batch_running()`):
   - **image cell** (`image=key`; cells get `CustomContextMenu` via
     `_attach_cell_menu` in `_build_group_content`): if the key is part of
     a multi-selection (`len(selected) > 1` and key selected) → 已选 ×2 +
     全部 ×2; otherwise 这张图片 ×2 (`MENU_INFER_IMAGE_*`, that key only).
   - **folder header** (non-root): 此文件夹 ×2 + 此文件夹未标注 ×2
     (`MENU_INFER_FOLDER_UNLABELED_*`, only when the folder has unlabeled
     files) + 全部 ×2.
   - **ALL row and the 根目录 group** (folder `FOLDER_ROOT_LABEL` maps to
     the same menu): 全部 ×2 + 全部未标注 ×2 (`MENU_INFER_ALL_UNLABELED_*`,
     only when present). 已选 / 此文件夹 never appear here — 已选 entries
     exist only on multi-selected image cells.
   Separator grouping strips spaces from the scope key so the LLM/本地模型
   variants of one scope stay in the same group.

## Testing rules

- `tests/gui/conftest.py`: offscreen env; `qapp` from pytest-qt; fixture
  `demo_dataset(tmp_path)` building a small real dataset (Pillow PNGs + txt files, incl.
  a kohya-style subfolder); fixture `controller(qapp, demo_dataset)` with dataset opened
  synchronously (call internal sync open or wait with qtbot.waitSignal).
- Focus tests on behavior: controller (selection/anchor/nav/scopes/replace/prefix,
  filter, pos_label, editor_keys), history model, theme tokens/QSS (no hardcoded colors:
  grep own widget modules for `#[0-9A-Fa-f]{6}` outside theme/ — a real test),
  chips/sents editing logic, find-replace live counts, translate bridge with a
  registered mock client, toast queueing, settings round-trip.
- Widget smoke tests: instantiate each panel with a real controller, drive one primary
  interaction via qtbot (click chip → inline editor appears; click 全部替换 with scope
  current, waitSignal batch_finished; etc.). No screenshots, no real waits > 2s.
- LLM in tests: `nlapt.llm.base.register_client` with MockLLMClient; unique api_type per
  test module.
```

## v1.10 — 拦截防护、下载进度重连、翻译重试、官方预设提示词

User request: LLM 推标偶发被安全策略拦截,产物风格明显异常;设置页关闭
重开后看不到下载进度只弹「已有下载任务正在进行」;翻译偶发超时;
JoyCaption / ToriiGate 这类模型自带官方预设,需要全部补上。

1. **推标请求走 spec-8 管控 + 拦截防护**:
   `AppController.make_vision_captioner_or_none(*, retry_sleep=time.sleep)`
   现在通过 `_paced_complete`(nlapt.llm.rewrite)执行请求 —
   `config.request` 的超时/指数退避重试/最小间隔全部生效,一个
   captioner 实例共享一个 `MinIntervalLimiter`(整个批次同一节奏)。
   清洗后的结果再过 `ensure_not_refusal`:疑似拒绝话术抛
   `LLMOutputError` → 单张推理在编辑区显示错误、批量里计入失败并保留
   原标注,不再把"道歉文"写进数据集。本地 llama captioner 同样加了
   refusal 防护(不加重试 — 本地 300 s 超时重试只会翻倍等待)。
2. **下载任务进程级化(重开对话框可重连)**:`local_bridge` 新增
   `_DownloadHub`(进程级 QObject,`get_download_hub()`)与单一活动任务
   快照 `active_download() -> (family_id, quant_label, done, total) | None`
   (模块函数 + `LocalBridge.active_download()`)。`LocalBridge.__init__`
   把 hub 的 `progress`/`finished` 转接到自己的同名信号(Qt 在 bridge
   销毁时自动断开);worker 只向 hub 发信号,原先对已销毁 bridge 的
   `_alive`/RuntimeError 兜底不再需要。`is_downloading()` 现在是
   进程级判断,`cancel_download()` 可从任何 bridge 取消当前任务,
   `start_download` 在已有任务时直接返回 False(每次一个任务;
   原 `_ACTIVE_DOWNLOAD_PATHS` 集合被单任务快照取代)。
   **LocalTab**:`_prefill_from_settings` 末尾检查 `active_download()`,
   命中则选中对应量化档、显示进度条并回放快照(按钮经由
   `is_downloading()` 自动进入「取消下载」态)。
3. **翻译协调修复**:`TranslateBridge(..., retry_sleep=time.sleep)`;
   web 服务(Google/百度/DeepL)的 `translate`/`translate_to` 调用统一
   包上 `with_retry(RetryPolicy(max_retries=config.request.max_retries))`
   (LLM 路径本就在 Translator 内部重试,不重复包装);配合核心侧
   `WEB_TRANSLATE_TIMEOUT_SECONDS = 15.0`,卡住的请求 15 秒即进入重试
   而不是干等 60 秒后直接失败。
4. **提示词预设(官方指令)**:`LocalTab` 新增 `提示词预设` 表单行 —
   `preset_combo`(QComboBox,userData 为 preset_id),仅当所选
   family 在 `nlapt.local.presets` 有官方预设时显示(JoyCaption 12 项 /
   ToriiGate 10 项),末项固定为「自定义(使用推理提示词)」
   (`PRESET_CUSTOM`)。按 family 惰性填充(`_populate_presets`,
   family 未变不重建),恢复持久化选择,未知/空值落到该 family 的
   默认(第一个)预设;`persist()` 写 `prompt_preset`(combo 未填充时
   保留存量值)。`make_local_vision_captioner` 的 llama 分支按
   `resolve_preset(family_id, settings.prompt_preset)` 用预设的
   system/user 覆盖传入提示词;`PRESET_CUSTOM` 沿用 设置 ▸ 提示词。
   云端 LLM 引擎与 Florence 不受影响。

## v1.11 — 空闲 30 秒再卸载、缩略图磁盘缓存、整段对照、目录树扁平化

User request: 本地推理经常连续跑好几轮,推理一结束就卸载模型导致每轮都
重新加载数 GB 权重 — 改为 30 秒内无新请求才卸载;左侧预览图加载很慢;
翻译对照按逗号切段翻译长句效果差,需要整段对照;目录树里单模型也要
展开两层;右栏底部说明文字无用;测试连接对思考型模型必失败,高并发
推标容易拿到空回复。

1. **空闲自动卸载(先不卸载,30 秒后再卸载)**:`local_bridge` 新增
   `IdleServerStopper`(`IDLE_STOP_DELAY_SECONDS = 30.0`,进程级单例
   `get_idle_stopper()`,绑定共享 server manager)。
   `note_request()`/`note_finished()` 成对包住每一次本地推理(单张与
   整个批次各算一次):请求开始先取消挂起的停止计时器,并在服务未运行
   时记下「本次是推理拉起的」;全部在飞请求结束且服务确为推理拉起时,
   armed 一个 `threading.Timer`(daemon,`timer_factory` 可注入),
   到点才 `manager.stop()`。`note_user_control()`(用户点 启动/停止
   本地服务 时由 `LocalBridge.start_server`/`stop_server` 调用)取消
   计时并清除 auto 标记 — 用户手动启动的服务永不被自动卸载。
   `vision_bridge` 不再在批量结束时立即 `manager.stop()`(原
   `server_was_running` 判定删除),单张本地推理同样参与空闲窗口。
2. **缩略图磁盘缓存(预览加载提速)**:`ThumbnailLoader` 在内存 LRU 外
   增加持久缓存 `app_data_dir()/thumbs`(`THUMB_DISK_DIR_NAME`,构造
   参数 `disk_cache_dir` 可注入)。键 = 源路径 + mtime_ns + size +
   bucket 高度的 SHA1,值为缩好的 PNG:大 PNG 缩到 64px 仍要整幅解码,
   有了磁盘缓存后 refresh 与后续会话直接读小图。`clear()` 只清内存
   (磁盘缓存保留,刷新数据集秒开);替换过的图片因 mtime/size 变化
   自然换新键,旧条目不清理(体积极小)。回调补了
   `shiboken6.isValid(self)` 守卫 — 加载器销毁后迟到的解码结果不再
   触发 "Signal source has been deleted"。
3. **翻译对照 整段模式**:`TranslateSection` 顶部新增 对照方式
   分段/整段 `SegmentedBar`(`MODE_SEGMENTS`/`MODE_WHOLE`,属性
   `whole_mode`)。整段模式下整篇标注是唯一一行(`_sources_for`),
   ⇄ 替换整篇(`set_caption`,标签仍为 翻译替换);中文全部转为英文
   在整段模式按整篇翻译、每文件提交一次(`_commit_results` 按模式
   分流);批次进行中 mode_bar 禁用,防止中途切换用错单元形状。
   翻译缓存仍按文本哈希共享,整篇文本只是另一个键。
4. **目录树扁平化(单模型不分层)**:`LocalTab._populate_tree` 折叠
   单子链 — 只有一个家族的系列不再出系列行(家族顶格),只有一个
   量化档的家族自身就是可选量化行(合并行带体积/热度/评级,同时携带
   ROLE_FAMILY + ROLE_QUANT;单档不再显示 ★推荐)。Florence-2 从三层
   变一行,ToriiGate/JoyCaption 变两层。`_quant_items()` 改为任意深度
   递归收集(判据:携带 ROLE_QUANT),`_select_quant_item` 与
   `current_selection()` 不变。
5. **移除右栏底部说明**:`SERVER_HINT` 常量与 `server_hint` QLabel
   删除(内容与实际操作重复,占空间)。
6. **测试连接测速 + 空回复修复(GUI 侧)**:设置页 测试连接 成功 toast
   变为 `连接成功({seconds:.1f} 秒)`(`probe` 返回 monotonic 耗时)。
   配合核心 v1.10 addendum(空回复抛可重试错误、
   `CONNECTION_TEST_MAX_TOKENS` 1024),思考型模型的测试不再必失败,
   高并发下的空回复走指数退避重试。`make_local_vision_captioner` 的
   llama 分支现在也包 `with_retry(RetryPolicy())` — 本地并发槽位繁忙
   或空回复同样重试(v1.10 曾因 300 s 超时顾虑不加;空回复是即时
   失败,重试代价低)。

## v1.12 — Florence LoRA 选择(设置 ▸ 本地推理)

1. **LoRA 行(仅 Florence 家族显示)**:`LocalTab` 在 指令模式 下新增
   「LoRA」行 — `florence_lora_combo`(首项固定为 `LORA_NONE_LABEL`
   「不使用 LoRA」,data=""; 其余项 data=safetensors 绝对路径,显示
   文件名 stem)+ 添加 / 移除按钮(`_add_lora` 打开
   `QFileDialog.getOpenFileName`,过滤 `*.safetensors`,去重后选中;
   `_remove_lora` 只移除非首项)。行可见性与 `florence_task_combo`
   同步(`florence_lora_holder`,选中非 Florence 或无选择时隐藏)。
   `persist()` 写入 `florence_lora`(当前选中,"" = 不使用)与
   `florence_loras`(全部已注册路径);预填时未注册但被选中的路径会
   自动补进下拉框。
2. **桥接**:`get_florence_engine(files, lora_path=None)` — 缓存键
   加入 LoRA 标识(路径 + mtime_ns + size,重新训练同名文件会自动
   重建引擎);`make_local_vision_captioner` 的 Florence 分支读取
   `settings.florence_lora`,非空则先做存在性检查(缺失 →
   `LocalInferenceError`,复用 `nlapt.local.lora.MSG_LORA_FILE_MISSING`
   的可操作文案),再把 `Path` 传给引擎。合并/兼容性错误(架构不匹配、
   零命中等)由核心层在加载时抛出,经既有推理错误链路展示给用户。
3. **约束**:LoRA 必须与当前 ONNX 模型同架构(现有目录中的
   Florence-2 PromptGen v2.0 实为 base 0.23B,隐藏维度 768);基于
   large(1024 维)训练的 LoRA(如 ModelScope 的 BAI_JSON LoRA)会
   得到明确的架构不匹配报错 — large 架构的 PromptGen ONNX 导出在
   快照时间点尚不存在,无法入目录。

## v1.13 — 官方 Florence-2 家族 + 内置 LoRA 下载

1. **内置 LoRA 展示与下载**:`_rebuild_lora_combo(family)` 组装 LoRA 下拉:
   固定首项「不使用 LoRA」+ 所选家族兼容的内置 LoRA(`loras_for_family`;
   data=主目录内 `loras/<id>/adapter_model.safetensors` 绝对路径,
   `ROLE_LORA_ID` 角色存 lora_id,ToolTip 显示 notes,未下载时名称追加
   `LORA_MISSING_SUFFIX`「(未下载)」)+ 用户自选文件(`self._user_loras`,
   持久化到 `florence_loras`;内置项不写入该字段)。选中未下载的内置项时
   显示「下载」按钮(`lora_download_button`)→
   `LocalBridge.start_lora_download(lora_id)`;完成后 toast
   `TOAST_LORA_OK` 并刷新(去掉未下载后缀)。内置项不可移除
   (`_remove_lora` 对带 `ROLE_LORA_ID` 的项不动作)。选中带 `task` 的
   内置 LoRA 会自动把 指令模式 切到该 指令(BAI_JSON → `<BAI_JSON>`)。
2. **指令按家族过滤**:`_rebuild_task_combo(family)` 只列出
   `family.florence_tasks`(官方家族看不到 PromptGen 专属指令;PromptGen
   看不到 `<BAI_JSON>`),切家族时保留仍合法的当前选择,否则回退第一项;
   `make_local_vision_captioner` 在推理侧同样钳制(设置文件被手改也安全)。
3. **桥接下载管道复用**:`LocalBridge._launch_jobs(family_id, quant_label,
   jobs, runtime_asset)` 是从 `start_download` 提取的共用执行器(单任务
   槽位 + 进度节流 + hub 广播不变);`start_lora_download` 用哨兵
   family_id `LORA_FAMILY_PREFIX + lora_id`(`"lora:"` 前缀)与空 quant
   走同一管道,LoRA 文件按 pinned 大小 + SHA256 校验,已存在即刻发
   OK。新增 `lora_adapter_file(entry)` / `is_lora_downloaded(entry)`
   (module 级 `is_lora_downloaded_in`,只查主目录 — LoRA 体积小,不进
   复用目录机制)。`_on_download_finished` 对 `lora:` 前缀走专属 toast,
   不再误入 `find_family`。
4. **目录树**:Florence 系列现有 5 个家族,系列节点恢复(此前单家族被
   扁平化到顶层);相关结构测试同步更新。

## v1.14 — 原型骨架接到现有功能(轨 + 两栏 + 工具浮层)

`nlapt-prototype/` 只当视觉规格,不进运行时、不进 PyInstaller。继续走
`AppController` + 现有 bridge;不嵌 WebEngine。

1. **主窗装配**:`QHBox(ToolbarRail 44px, body_splitter)`。splitter 两栏
   (文件 | 预览+编辑)。标题栏与状态栏不再挂到 `MainWindow`;
   `TitleBar` / `StatusBar` 模块保留,单测仍可直接实例化。
   `body_splitter.json` 从 3 个数迁到 2 个数(旧档取 left,丢掉 right)。
2. **窗控搬家**:`window_chrome.py` 抽出 `WindowButton` / `ThemePopup` /
   `WindowDragHelper`。预览 46px 信息栏承担拖窗、双击最大化、─ □ ×。
   轨空白也可 `startSystemMove`。边缘缩放仍在 `MainWindow`。
3. **ToolbarRail**:打开 / 保存 / 全部保存 / 导出 / 撤销 / 重做 /
   修改工具 / 主题 / 设置;Logo 菜单(使用说明 / 关于 / 退出)。
   主题弹层仍是五套 + 色彩设置,不是亮暗对切。
4. **ToolsPanel 浮层**:轨「修改工具」开关,右侧 340px 覆盖中栏,默认隐;
   Esc 收起。四张卡片不重写。动画只用几何 / `windowOpacity`。
5. **信息栏**:文件名 + 格式胶囊 + `宽 × 高` / 大小 / `修改于 yyyy-mm-dd`
   + 窗控。撤销 / 复制 / 保存从预览顶栏撤走(复制改快捷键 + 文件右键)。
   多选对比、CaptionBar、文件夹推标、未保存黄点全部保留。
6. **重做**:`FileHistory.step_newer` + `AppController.redo_current`
   (`Ctrl+Y` / `Ctrl+Shift+Z`)。走 UI 历史光标,不是 `NLaptApp.redo`。
7. **导出**:`MainWindow.export_dataset` → 未打开 toast;有脏文件则确认
   「先全部保存再打包」; `QFileDialog` 选目标; worker 调
   `AppController.export_dataset(dest, save_first=...)`。
8. **文件栏默认宽 260 / 最小 200**。最小窗仍 1360×760。

## v1.15 — 修改工具抽屉滑入/滑出

轨「修改工具」开关仍是右侧 340px 覆盖层 + Esc 收起。打开/关闭改为
`nlapt_gui.anim.slide_geometry` 纯 geometry 滑动(屏外 → 右缘 / 右缘 → 屏外),
不用 `QGraphicsOpacityEffect`。`animations_enabled()` 为假时同步落到终态
(测试契约不变)。窗口 resize 时把进行中的滑动跳到终态再贴新右缘;关闭滑动
的 `finished` 才 `hide()`,且只在该动画仍是当前抽屉动画时生效,避免连点
互打误藏。

## v1.16 — 工具抽屉避开预览顶栏窗控

抽屉不再铺满窗高:停靠矩形从 `HEADER_H`(46px 预览信息栏)下沿起,高度
`window.height - HEADER_H`。预览顶栏的 ─ □ × 始终露在抽屉上方,可点。
屏外起点仍是该矩形右移 340px。

## v1.17 — 胶囊 FLIP / 预览去闪 / 折叠门控 / 本地推理卡片

1. **胶囊拖拽**:`ChipsEditor._rebuild` 做 FLIP(`anim.animate_reflow` /
   `flip_reflow`),按段文本+出现序号对齐旧几何;被拖放的胶囊 `pop_in`。
   测试禁用动画时同步落终态。
2. **预览切图**:未命中缓存时保留上一张,解码完成再淡入;数据集切换仍清空。
   `_start_fade` 经 `animations_enabled()` 门控。
3. **合集折叠**:`_FolderGroup.set_open` / `_Arrow.animate_to` 经同一门控,
   重入先 stop 旧动画。
4. **布局**:提示词页系统/用户框 3:2 stretch,导出并入模板行,本地区块分隔;
   文件列表头行间距统一;`SettingsDialog` 表单 label 最小宽与间距统一。
5. **本地推理右栏**:四张 `surfaceCard`(模型状态 / 模型设置 / 运行参数 / 目录),
   构建抽出 `local_tab_sections.py`;属性名与中文文案不变。

## v1.18 — 分层推标(人物卡 + 画面描述)

Two-stage caption workflow. Final txt = locked English character card + blank
line + per-image pose/scene paragraph. Core `nlapt/` unchanged; batch write
reuses `AppController.run_caption_batch` (v1.7 snapshot / oplog / progress).

1. **Skills** (`nlapt_gui/layered_prompts.py`): `CHARACTER_CARD_PROMPT` /
   `POSE_SCENE_PROMPT`; `build_card_prompt(name, series, *, variant)`,
   `build_scene_prompt(name)`, `assemble_caption(card, scene)`. Scene system
   prompt is the user's existing 提示词 template (`VisionPrompts.system_text_for`);
   the scene skill is the user message. History label `分层推标`.
2. **Memory** (`nlapt_gui/layered_store.py`): `LayeredMemory` at
   `app_data_dir()/layered_infer.json` (name / series / last card).
3. **VisionBridge**: `custom_ready(request_id, text, ok)`;
   `request_custom(request_id, key, engine, *, system, user_prompt)`;
   `request_layered_batch(keys, engine, *, card_text, scene_system, scene_user)`;
   `local_engine_is_florence()` (no download required). Florence local engine
   cannot run free-form skills — the wizard disables 本地模型.
4. **Wizard** (`widgets/layered_infer_dialog.py` + `layered_infer_cards.py`):
   `CenteredDialog` / `QStackedWidget` — role (name, series, engine, clickable
   `RefPickerGrid` strip + `PreviewPane` that scales with the dialog) → three
   bilingual candidates (editable EN, review-only ZH via
   `TranslateBridge.request_to`, live `约 N tokens · M 词` stats) → confirm
   (same stats on the locked card and the scene-prompt preview). Confirm calls
   `request_layered_batch` and closes; `BatchProgressDialog` takes over.
   Token counts are a `len//4` heuristic (`format_card_stats`); no tokenizer.
5. **FilePanel**: `layered_infer_requested(keys)` plus scoped menu entries
   (`分层推标此文件夹/已选/全部/这张图片` and unlabeled variants). Labels and
   scope wiring live in `widgets/file_panel_infer.py`. No `ask_confirm` on
   this path (wizard page 3 confirms). `MainWindow` opens
   `LayeredInferDialog`. Batch-running menus still only offer `取消当前推标`.

## v1.19 — 底部常驻译文 + token/词统计

1. **CaptionBar**: `删除` is followed by `译文`. That button translates the
   whole current caption to Chinese via `TranslateBridge.request_to(..., "zh")`
   and emits `inline_translation_ready(key, result)` — no overlay, no replace.
   The existing `翻译 ▾` menu (floating `TranslationPreview` + 替换) is unchanged.
   `InlineTranslationPanel` is the read-only docked card (title `译文(中文)`,
   selectable text, 关闭).
2. **EditorPanel**: the panel sits under the editor blocks and above the mode
   hint. `current_changed` / 关闭 clears it. Switching files never leaves a
   stale translation visible.
3. **AppController.char_seg_info**:
   `{n} 字符 · {m} 段 · 约 {t} tokens · {w} 词`
   using `layered_prompts.estimate_tokens` / `count_words` (`len//4` heuristic).
   Tab-row `_char_info` and per-block `_stat` follow automatically.

## v1.20 — 人物卡 skill 外貌优先

`CHARACTER_CARD_PROMPT` is assembled from two named skills:
`CARD_APPEARANCE_SKILL` (hair / eyes / pupils / fixed face facts) then
`CARD_CLOTHING_SKILL` (full outfit). The card paragraph must finish
appearance before the first garment. Variant hints no longer ask the
model to lead with outerwear or footwear.

## v1.21 — 人物卡对照串行 + 调试模式

1. **LayeredInferDialog** Chinese gloss: English cards enqueue `request_to`
   and the wizard sends one Google/web translate at a time (`_zh_queue` /
   `_zh_inflight`). Status stays `正在翻译对照…` while the queue is
   nonempty. Failure writes `翻译失败: {message}` on the card — never the
   `中文对照将显示在这里` placeholder.
2. **`UISettings.debug`** defaults to `False` and is persisted in
   `ui_settings.json`. The 翻译服务 tab has a 调试模式 checkbox. Save
   calls `controller.update_settings(debug=…)` and immediately
   `configure_logging(..., console=debug)` plus `workers.set_debug`.
3. **`configure_logging(..., *, console=False)`**: file handler unchanged;
   a console `StreamHandler` is added only when `console=True`.
   `__main__` loads UI settings first and passes `console=settings.debug`.
   `FunctionWorker.run` logs a one-line warning without traceback when
   debug is off; `exception` (full stack) when on.

## v1.22 — 自定义翻译 API

The 翻译服务 dropdown adds **自定义 API** (`provider=custom`): an
OpenAI-compatible `/chat/completions` endpoint (Base URL + 模型 + optional
API Key). Provider selection still lives in `app_data_dir()/translate.json`.
Custom credentials are written only to
`Documents/NLapt/translate_api.json` (`NLAPT_DOCUMENTS_DIR` overrides the
Documents root in tests). `CustomOpenAIProvider` in `web_translate.py`
implements `translate` / `translate_to`.

## v1.23 — DeepLX 免费接口

The 翻译服务 dropdown adds **DeepLX (免费)** (`provider=deeplx`):
`POST {url}/translate` with `{text, source_lang: auto, target_lang}` per
https://deeplx.owo.network/endpoints/free.html . URL is required; optional
access token is sent as `Authorization: Bearer`. Settings persist
`deeplx_url` / `deeplx_token` in `translate.json`.

## v1.24 — DeepLX 密钥隔离 + 本地 Hy-MT2

DeepLX URL/token move to `Documents/NLapt/translate_api.json` (same file as
the custom OpenAI key). AppData `translate.json` keeps only provider +
Baidu/DeepL + `local_mt_tier`. Community instances are rate-limited in
`DeepLXProvider` (shared 1 s gap + 429 backoff). Error toasts never echo
a key embedded in the URL path.

The dropdown adds **本地模型 (Hy-MT2)** (`provider=local_mt`): a tier
combo (快速 / 均衡 / 高质量) and **管理模型**, which opens
`MTModelsDialog` (download / cancel / delete). `configured()` is true only
when the selected tier's GGUF is present. TranslateBridge treats
`local_mt` like `llm` (GUI-owned server), not `create_provider`.

## v1.25 — Dedicated worker pools

Translation, model download, and image decode no longer share
`QThreadPool.globalInstance()`. `TranslateBridge` uses a 3-thread pool
and drops queued work on `dataset_opened` (`invalidate_pending`).
Downloads run on a 1-thread pool in `download_hub`. Thumbnails and the
preview panel decode on a 4-thread pool. The global pool remains for
dataset scan and local inference. `FunctionWorker` swallows
`RuntimeError` from `emit` when the signal QObject is already gone.

## v1.26 — 分层推标 does not restore last card

`LayeredMemory` still persists name / series / last card. The wizard
prefills only name and series. Candidate cards start empty; a generate
or 重新生成 clears the slot before the new reply arrives.

## v1.27 — DeepLX 418 对用户可读

分层推标 / 译文对照 on DeepLX no longer surface `I'm a teapot` or raw
JSON. Busy replies (418 / 429 / 503) retry, then show
`DeepLX 请求过于频繁，请稍后再试或换用自建实例`.

## v1.28 — 翻译备选顺序

`TranslationConfig.fallback_order` is a tuple of provider ids persisted
in AppData `translate.json`. `provider_chain` is
`(provider, *fallback_order)` after dropping the primary, unknowns, and
duplicates. `TranslateBridge` builds one callable per usable id and
runs `run_fallback_chain` for both `translate` and `translate_to`.
`configured()` is true when any id in the chain can be built.

设置 ▸ 翻译服务 shows a 备选顺序 list (`FallbackOrderEditor`): 上移 /
下移 / 移除 / 添加 / 填入推荐 (Google → 百度 → 本地 Hy-MT2). Changing
the 首选 dropdown removes that id from the list. Unconfigured fallbacks
are skipped at runtime and do not block save.

## v1.29 — Hy-MT2 picker always listed + completion cap

The 翻译服务 tab always shows `MtTierPicker` (three Hy-MT2 rows +
管理模型), not only when the primary provider is `local_mt`.
`LocalMTProvider` posts `/v1/completions` with the official prompt and
`max_tokens` (512–1024). `ensure_mt_server` uses `MT_CONTEXT_LENGTH`
(4096), not the captioner's context setting.

## v1.30 — Hy-MT2 chat protocol and complete translations

`LocalMTProvider` posts `/v1/chat/completions` with a single user message,
no system message, and `stream=False`, letting llama.cpp apply the GGUF's
chat template. The model name is the GGUF filename. The official prompt
separates instructions from source text with two newlines.

Sampling uses `temperature=0.7`, `top_p=0.6`, `top_k=20`, and llama.cpp's
`repeat_penalty=1.05`; `min_p=0` disables its additional default filter.
The output budget is 4096 tokens and the dedicated MT context is 8192,
leaving room for source tokens without inheriting caption-model settings.
Only `choices[0].message.content` with `finish_reason="stop"` is accepted;
missing, empty, reasoning-only, or truncated replies raise `LLMRequestError`.

The provider constructor requires keyword-only `retry_sleep: Callable[[float], None]`.
HTTP/transport failures log warnings and retry twice (1 s / 2 s); the final
error propagates. Response-validation errors are not retried. The idle
stopper brackets the whole operation, including retries, and is released
on both success and failure. Production callers pass their sleep callable;
tests inject a nonblocking callable.

## v1.31 — 分组裁切动画与渐进式图片预览

Folder groups animate an outer clipping viewport from the current visible
height to the natural content height (or zero). The child list/grid keeps its
natural row geometry throughout the animation. Folder and arrow animations
are reused on re-entry, and `folder_open` persistence and existing click
semantics remain unchanged.

Thumbnail decoding keeps the existing dedicated four-thread pool but submits
at most three thumbnail workers at once. `ThumbnailLoader.clear()` advances
the active generation, invalidates old callbacks, preserves the disk cache,
and suppresses repeated failures until the next clear. Thumbnail painters use
source cropping directly rather than allocating a scaled pixmap per repaint.

The main preview uses one decode slot and presents a cached or newly decoded
coarse frame (maximum edge 512px) before loading the native frame. Original
image dimensions drive fit and zoom geometry, so native-frame promotion does
not move the image or change the scroll anchor. Dataset generations prevent
late frames from replacing the current dataset; native preview cache limits
remain eight entries and 64M pixels.

## v1.32 - Stable folder collapse geometry

Folder-group layouts stay top-aligned while animated content heights and
parent layout requests settle on different frames. The clipping viewport's
minimum height hint is zero, preserving header space on the first expanding
frame while its body retains its natural height. The file panel reserves
its vertical scrollbar width even at zero scroll range, so collapse/expand
never shifts the header downward or resizes visible thumbnail rows mid-frame.

## v1.33 - Wrapped chips and filtered selection

Caption chips render plain text through QTextDocument using
WrapAtWordBoundaryOrAnywhere, preserving explicit line breaks and wrapping
unbroken tokens. FlowLayout bounds each item to its available width before
evaluating heightForWidth. Short chips retain their natural width; delete
buttons stay at the top right. InlineChipField remains a single-line editor
with a content-based size hint, bounded by the same layout, and native cursor
scrolling. Enter, Escape, focus-out commits, and caption serialization retain
their existing meanings. ChipWidget and InlineChipField remain importable from
chips_editor; their implementation now lives in chip_widgets.

AppController.select_filtered() replaces selection with filtered_keys(),
including matching items outside the viewport. Top-level selection coverage
uses these matching keys, and the checkbox is disabled for zero matches.
set_folder_selected() and folder_selection_state() likewise use matching keys
inside that folder. folder_keys(), select_all(), the ALL row, and explicit
whole-folder context-menu operations retain their full-dataset scopes. ALL
tooltips state that hidden results are included. Clearing selection is global.

set_filter() intersects existing selection with the new results before emitting
filter_changed, then emits selection_changed only when membership changed.
It preserves the current image. In single-image mode, pos_label() reports an
index within the filtered results or `未匹配 / N` for a non-matching current
image (`- / N` when there is no current image). nav(1)/nav(-1) enter the first/
last result when current is unmatched; no results means no navigation, with no
fallback to all images. can_navigate() reports whether this navigation order
is nonempty. Multi-selection navigation continues to use the selected keys,
including an explicit ALL selection. Preview navigation buttons follow this
availability without changing the displayed image or its zoom on filtering.

## v1.34 — Deferred chip/sentence drop commit

`ChipsEditor` / `SentsEditor` drag-reorder must not commit or rebuild inside
`QDrag.exec()`. `dropEvent` only records `_pending_drop = (from, to)`;
`begin_drag` calls `_finish_drag()` after `exec` returns, which then
`reorder`s (history label `拖拽排序`) and rebuilds. `refresh()` while
`_drag_index` is set sets `_refresh_pending` and returns; `_finish_drag`
applies that refresh and discards the drop (indices may be stale).

`anim.pop_in` is geometry-only and never installs `QGraphicsOpacityEffect`.
`flip_reflow` puts the slide and the pop in one group so
`anim.finish_animation` lands both. `ChipsEditor.begin_drag` finishes any
running reflow and clears leftover effects before starting a drag;
`_rebuild` skips FLIP while a drag is active.

## v1.35 — Chip rebuild must show before layout / drop FLIP

`QWidgetItem.setGeometry` is a no-op on hidden children. `ChipsEditor._rebuild`
must `show()` every flow item, then `invalidate()` + `activate()`, before
`flip_reflow` snapshots end rects. Otherwise a wrapping chip still has the
default `(0, 0, 100, 30)` and the drop animation flies it to the top-left.

`SegmentEditorBase.reorder(from, to) -> bool` returns whether the caption was
committed. `_finish_drag` rebuilds only when the drop did not commit: a
successful `reorder` already rebuilt via synchronous `caption_changed` →
`refresh()`, and a second `_rebuild` would `finish_animation` the drop FLIP.

`anim.pop_in` keeps width fixed (`final.adjusted(0, dy, 0, -dy)`). A width
tween reflows `_ChipText` every frame.

## v1.36 — CHA标注（组合分层推标）改名 + 独立设置

User-facing name: daily short form **CHA标注**; full name **组合分层推标**
(English: Combined Hierarchical Annotation) appears only on the wizard
title and the settings-tab hint. Code identifiers stay `layered_*`.

- Wizard title `CHA标注 · 组合分层推标 (Combined Hierarchical Annotation)`;
  confirm button `开始CHA标注`; file-panel menus `CHA标注此文件夹/已选/全部/这张图片`
  (+ unlabeled variants); history / progress `CHA标注`.
- `CHASettings` (`nlapt_gui/cha_config.py`) persists
  `app_data_dir()/cha_annotation.json`; `api_key` lives in
  `Documents/NLapt/cha_api.json`. Fields: `api_mode` (`sync`|`own`),
  `api_type`, `base_url`, `api_key`, `card_models` (3 slots), `batch_model`.
  Empty model strings inherit the resolved base profile's `vision_model`.
- Resolvers: `resolve_base_profile`, `resolve_card_profile(settings, active, index)`,
  `resolve_batch_profile`. Return `None` when no usable `base_url` /
  `vision_model`.
- `AppController.active_profile()`; `make_vision_captioner_or_none(*, profile=None)`.
  `VisionBridge.configured` / `_make_captioner` / `request_custom` /
  `request_layered_batch` accept `profile=`; `ENGINE_LOCAL` ignores it.
- `LayeredInferDialog(..., cha_settings=None)` defaults to `load_cha_settings()`.
  Each candidate card shows `模型: {name}`; confirm page shows
  `整批画面模型: {name}` (`本地模型` when the local engine is selected).
- Settings dialog fifth tab `CHA标注` (`CHATab`): checkbox
  `同步 LLM 设置中的接口` (default on) hides the own-API rows;
  four editable model combos (候选 1/2/3 + 整批画面段);
  `获取模型` / `测试连接`. `save()` writes CHA settings; own mode
  requires a Base URL. A CHA persist error toasts and does not
  roll back the other tabs.

## v1.37 — Built-in vision system-prompt schemes

`VisionPrompts.names()` is `默认`, then `BUILTIN_PROMPT_ORDER`
(`结构化视觉编译`, `客观视觉报告`), then custom names. Bodies live in
`nlapt_gui/builtin_prompts.py` and are resolved by
`VisionPrompts.system_text_of`; they are not written to
`vision_prompts.json`. `load_vision_prompts` accepts a shipped name as
`active` even when `prompts` is empty. A custom entry with the same name
overrides the shipped body.

`is_locked_template(name)` is true for 默认 and the shipped schemes.
`PromptsTab` treats locked templates as read-only (save/delete disabled);
新建自定义 forks the visible text. `current_prompts()` keeps a shipped
`active` without copying the body into `prompts`.

## v1.38 — Unified API config (`Documents/NLapt/api.json`)

All interface settings (type / URL / key / model names) live in one file
`user_documents_dir()/NLapt/api.json`. AppData files keep non-interface
options only. `NLAPT_DOCUMENTS_DIR` isolates the Documents root in tests.

```python
# nlapt_gui/api_config.py
API_FILE_NAME = "api.json"; API_FORMAT_VERSION = 1

@dataclass(frozen=True)
class TranslateCredentials:
    baidu_appid: str = ""
    baidu_key: str = ""
    deepl_key: str = ""
    deeplx_url: str = ""
    deeplx_token: str = ""
    custom_base_url: str = ""
    custom_api_key: str = ""
    custom_model: str = ""

@dataclass(frozen=True)
class CHAApi:
    api_type: str = "openai"
    base_url: str = ""
    api_key: str = ""

@dataclass(frozen=True)
class ApiConfig:
    profiles: tuple[LLMProfile, ...] = ()
    active_profile: str = ""
    translate: TranslateCredentials = TranslateCredentials()
    cha: CHAApi = CHAApi()
    def with_changes(**changes) -> ApiConfig

def api_config_path() -> Path            # does not create the directory
def load_api_config() -> ApiConfig       # missing -> legacy fallback; corrupt existing file -> defaults
def save_api_config(config: ApiConfig) -> None
def update_api_config(*, profiles=None, active_profile=None, translate=None, cha=None) -> ApiConfig
def load_app_config() -> AppConfig       # AppData non-interface + api.json profiles
def save_app_config(config: AppConfig) -> None  # profiles -> api.json; rest -> config.json (profiles emptied)
def migrate_legacy_api_files() -> bool   # startup; best-effort; never raises
```

Field ownership:

| File | Fields |
|---|---|
| `Documents/NLapt/api.json` | `llm.active_profile` / `llm.profiles`; translate credentials; CHA `api_type` / `base_url` / `api_key` |
| `%APPDATA%/NLapt/config.json` | `request`, `snapshot_retention`, and other non-interface `AppConfig` fields (no `profiles` / `active_profile`) |
| `%APPDATA%/NLapt/translate.json` | `provider` / `fallback_order` / `local_mt_tier` |
| `%APPDATA%/NLapt/cha_annotation.json` | `api_mode` / `card_models` / `batch_model` |

Startup (`nlapt_gui/__main__._load_app_config`) calls
`migrate_legacy_api_files()` then `load_app_config()`. Migration runs only
when `api.json` is absent and a legacy location has content: it writes
`api.json`, strips secrets from the three AppData files, and deletes
`Documents/NLapt/translate_api.json` and `cha_api.json`. A second call is
a no-op. Settings / 本地推理 persist LLM profiles through
`load_app_config` / `save_app_config`. `translate_config` and `cha_config`
keep their public dataclasses; credentials go through `load_api_config` /
`update_api_config`. This module must not import those two (legacy file
names are duplicated strings to avoid a cycle).
