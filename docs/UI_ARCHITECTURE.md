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
