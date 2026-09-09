# NLapt 打包指南 (PyInstaller)

本目录为打包预留:`nlapt.spec`(PyInstaller 规格文件)与 `build.ps1`(Windows 构建脚本)。

## 最快方式:双击根目录的批处理

- `Launch-NLapt.bat` — 双击直接运行图形界面(缺依赖会自动安装 PySide6 / Pillow / httpx,
  并尽力安装 Florence-2 所需的 numpy / onnxruntime)。
- `Build-NLapt.bat` — 双击一键打包(缺 PyInstaller 或任一运行依赖会自动安装),完成后自动打开
  `dist\NLapt` 输出目录。等价于下文的 `build.ps1` 流程。

> `httpx` 承担全部 LLM / 翻译 / 本机 llama-server 请求。它是懒加载的隐式依赖,
> 打包机上若未安装,PyInstaller 只会给出 warning 并照常产出 exe,用户一做推理就会看到
> 「The 'httpx' package is required for LLM HTTP clients」。因此 `nlapt.spec` 在
> `Analysis` 之前会先检查 PySide6 / PIL / httpx / numpy / onnxruntime,缺任一个直接报错终止。

> 两个 `.bat` 为纯 ASCII 英文脚本(不含中文、不使用 `chcp`),以避免在 GBK 默认
> 代码页的中文 Windows 上出现乱码/解析错误。

如需手动或在 CI 中构建,继续参考以下步骤。

## 环境准备

```powershell
# 在仓库根目录,安装全部运行依赖与打包依赖
pip install -e .[gui,images,llm,local]
pip install pyinstaller
```

## 构建

```powershell
# 方式一:构建脚本(推荐)
powershell -ExecutionPolicy Bypass -File packaging/build.ps1

# 方式二:直接调用 PyInstaller
python -m PyInstaller packaging/nlapt.spec --noconfirm
```

## 输出结构

onedir(目录式)+ windowed(无控制台)构建:

```
dist/
  NLapt/
    NLapt.exe          # 主程序
    _internal/         # Python 运行时、PySide6、nlapt 代码等
```

运行时数据(设置、日志、LLM 配置)不写入安装目录,统一位于
`%APPDATA%\NLapt`(通过 `nlapt_gui.resources.app_data_dir()` 解析,
可用环境变量 `NLAPT_DATA_DIR` 覆盖),因此打包目录可整体拷贝分发。

## 替换图标

`packaging/icon.ico` 已内置(由 `python packaging/make_icon.py` 从程序
logo 生成),`nlapt.spec` 的 `EXE(icon=...)` 与运行时窗口/任务栏图标
(`nlapt_gui.theme.logo.load_app_icon`)都指向它。想换图标时,直接用
新的 `.ico` 覆盖 `packaging/icon.ico` 即可,无需改代码;删除该文件则
回退为主题色矢量 logo。

## 版本号

版本号单一来源于 `nlapt.__version__`(`nlapt/__init__.py`):
标题栏 `自然语言标注工具 v{version}`、关于对话框与
`QApplication.applicationVersion` 均从它读取。升级版本时只需修改
`nlapt/__init__.py` 与 `pyproject.toml` 中的 `version` 字段。

## 注意事项

- `.spec` 中已排除 `tests` / `docs` / `pytest` / `PyQt5` / `PyQt6` / `tkinter`。
- 资源解析已通过 `nlapt_gui.resources.resource_path()` 兼容
  `sys._MEIPASS`;`.spec` 的 `datas` 仅随包附带 `packaging/icon.ico`
  (供运行时窗口图标使用),除此之外无需额外 datas 配置。
- 首次构建后建议在干净机器(或删除 `%APPDATA%\NLapt` 后)冒烟验证:
  打开文件夹、编辑标注、保存、切换主题。
