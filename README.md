<div align="center">

# NLapt

**自然语言标注工具** · Natural Language Annotation Processing Tools

面向图像生成与 LoRA 训练的桌面标注编辑器：处理「图片 + 同名 `.txt`」，单张精修，批量可回滚。

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Version](https://img.shields.io/badge/version-0.2.2-informational)
[![SLIM](https://img.shields.io/badge/Best%20Practices%20from-SLIM-blue)](https://nasa-ammos.github.io/slim/)

</div>

界面为中文。Windows 优先，其它平台可用 Python 3.11+ / PySide6。数据格式兼容 kohya 子目录；danbooru 标签与自然语言可混排。标注只在显式保存时写入磁盘。

[更新说明](docs/更新日志.md) · [Releases](https://github.com/storyAura/NLapt/releases) · [Issue](https://github.com/storyAura/NLapt/issues)

## 界面预览

<table>
<tr>
<td><img src="docs/screenshots/main-window.jpg" alt="主界面"><br><em>主界面：左栏文件、中栏编辑、右栏工具</em></td>
<td><img src="docs/screenshots/multi-compare.jpg" alt="多选对比"><br><em>最多 4 张并排对比</em></td>
</tr>
<tr>
<td><img src="docs/screenshots/caption-translate.jpg" alt="整篇翻译"><br><em>整篇中 / 英 / 日翻译对照</em></td>
<td><img src="docs/screenshots/settings-llm.jpg" alt="设置 · LLM"><br><em>设置 · LLM 档案与模型池</em></td>
</tr>
<tr>
<td><img src="docs/screenshots/settings-translate.jpg" alt="设置 · 翻译服务"><br><em>设置 · 翻译通道与备选</em></td>
<td><img src="docs/screenshots/settings-local.jpg" alt="设置 · 本地推理"><br><em>设置 · 本地模型目录</em></td>
</tr>
</table>

## Features

- **三种编辑模式**：胶囊 / 分句 / 文本；悬浮工具栏可分段、插入、翻译、重译、删除
- **整篇工作**：中 / 英 / 日翻译；LLM 或本地模型重新推标；最多 4 张多选对比
- **多 API 与模型池**：同时保存多个 API 档案，逐个模型开关；从池里分别选当前文本模型（翻译 / 改写）和当前视觉模型（推标），可来自不同供应商
- **CHA标注**（组合分层推标）：一次最多锁定 8 张角色卡（多角色，或同一角色多套服装）。先写人物卡，再整批只写姿态与场景，并按图核对服装。可选本地角色识别；人物卡 / 画面段提示词可分别自定义
- **多对比推标**：同一批图同时交给多个模型，并排核对后再写入已选结果
- **图像工具**：替换透明底（固定或随机纯色）、哈希检索雷同图；范围含当前文件夹及子文件夹
- **翻译通道**：大模型、Google、百度、DeepL、DeepLX、本机 Hy-MT2；首选失败按备选顺序继续
- **本地推理**：GGUF 目录、五级「能否运行」、应用内下载与 SHA256 校验；Florence-2 可挂 LoRA
- **数据安全**：批量前自动 zip 快照并可整批回滚；未保存草稿崩溃后可恢复

## Contents

* [界面预览](#界面预览)
* [Features](#features)
* [Quick Start](#quick-start)
* [Configuration](#configuration)
* [Changelog](#changelog)
* [FAQ](#frequently-asked-questions-faq)
* [Contributing](#contributing)
* [License](#license)
* [Support](#support)

## Quick Start

### Requirements

- Python 3.11+
- Windows 推荐；其它平台需自行安装 PySide6 等依赖
- 在线功能需要 LLM / 翻译接口；本地推标需要磁盘空间以下载 GGUF 或 Florence 权重

### Setup Instructions

**方式一（Windows）**

1. 双击 [`Launch-NLapt.bat`](Launch-NLapt.bat)
2. 脚本会检查 Python 与全部运行依赖（PySide6 / Pillow / httpx，以及本地 Florence 用的 numpy / onnxruntime），缺失时自动安装

**方式二（pip）**

```powershell
pip install -e .[gui,images,llm,local]
```

可选依赖：`gui` = PySide6，`images` = Pillow，`llm` = httpx，`local` = onnxruntime / numpy。开发再加 `,dev`。

### Run Instructions

```powershell
python -m nlapt_gui
```

或安装后使用 `nlapt-gui`。启动后点左侧图标轨的「打开」，选择数据集根目录；下次启动会重开上次目录。

### Usage Examples

- 左栏浏览与多选，中栏编辑 caption；轨上「修改工具」抽出查找替换 / 前缀后缀 / 翻译对照 / 历史。
- 轨上 Logo 打开「关于」或退出。轨上「工具」弹层可切换当前文本 / 视觉模型，并启动 CHA / LLM / 本地 / 多对比推标、替换透明底、查找雷同图片。
- 「设置」配置翻译服务、LLM 档案与模型池、提示词、本地推理、CHA标注、多对比推标。
- 左栏右键可对这张 / 已选 / 此文件夹 / 未标注 / 全部做 LLM、本地、CHA 或多对比推标。
- CHA标注：向导里为每张角色卡填角色名 / 作品名、选参考图，可「识别角色」；生成三套人物卡后选定，再整批写入「人物卡 + 空行 + 画面段」。人物卡与画面段提示词在设置 ▸ CHA标注 ▸ 编辑提示词 里分别自定义。

常用快捷键：`Ctrl+S` 保存当前，`Ctrl+Shift+S` 全部保存，`Alt+↑` / `Alt+↓` 换图，`Shift+点击` 范围选。

核心库也可单独调用：

```python
from pathlib import Path

from nlapt import NLaptApp

app = NLaptApp(log_dir=Path("logs"))
scan = app.open_dataset(Path(r"D:\dataset"))
app.edit(scan.images[0].key, "a girl, standing")
app.close()
```

### Build Instructions

```powershell
Build-NLapt.bat
```

产出 `dist/NLapt/NLapt.exe`（onedir、无控制台）。应用图标为 `packaging/icon.ico`。详见 [packaging/README.md](packaging/README.md)。

### Test Instructions

```powershell
python -m pytest -q
python -m pytest tests/gui -q
python -m pytest --cov=nlapt --cov=nlapt_gui -q
python -m ruff check nlapt nlapt_gui tests
```

GUI 测试离屏运行，不需要显示器。套件不访问真实网络、不使用真实 sleep。

## Configuration

| 位置 | 内容 |
|---|---|
| `%APPDATA%\NLapt`（`NLAPT_DATA_DIR` 可覆盖） | 主题、通道选择、提示词、本地模型设置、缩略图缓存、日志 |
| 文档\NLapt\api.json（`NLAPT_DOCUMENTS_DIR` 可覆盖「文档」根） | LLM 档案（类型 / 地址 / 密钥 / 模型开关）、当前文本 / 视觉模型、翻译通道、CHA 独立接口 |

第一次启动若还没有 `api.json`，会把旧版散落的密钥迁过去并清掉残留。密钥不进数据集目录。

## Changelog

近期面向使用者的改动见 [docs/更新日志.md](docs/更新日志.md)。当前版本 **0.2.2**。

发布记录见 [Releases](https://github.com/storyAura/NLapt/releases)。

## Frequently Asked Questions (FAQ)

1. **CHA标注和普通推标有什么区别？**  
   普通推标直接为每张图生成整段标注。CHA标注先锁定一套或多套人物卡，再为每张图只写姿态与场景，最终拼成「人物卡 + 空行 + 画面段」。一次最多 8 张卡。
2. **多对比推标怎么用？**  
   先在设置 ▸ 多对比推标勾选至少两个已启用的模型，再从工具弹层或左栏右键选范围。对比窗口里点选或「全部采用某模型」，最后「写入已选」才落盘。
3. **Florence-2 能做 CHA标注吗？**  
   不能。Florence-2 只走指令模式，不支持自定义提示词，请改用 LLM。
4. **人物卡 / 画面段提示词可以改吗？**  
   可以。设置 ▸ CHA标注 ▸ 编辑提示词，左右两栏分别改。与内置默认相同则仍走默认。
5. **配置和密钥存在哪里？**  
   应用数据在 `%APPDATA%\NLapt`。全部接口配置在「文档\NLapt\api.json」。详见 [Configuration](#configuration)。
6. **批量操作会弄坏数据集吗？**  
   执行前会把全部 txt 打成 zip 快照（默认保留 20 份），可在历史记录整批回滚。快照写在 `%APPDATA%\NLapt\datasets\`，不放进数据集目录。

## Contributing

1. 在 [Issues](https://github.com/storyAura/NLapt/issues) 说明要改什么
2. [Fork](https://github.com/storyAura/NLapt/fork) 本仓库
3. 在自己的 fork 里修改，并补上对应测试
4. 提交 Pull Request，约定使用 Conventional Commits（`feat:` / `fix:` / …）

首次参与可参考：[How to Contribute to an Open Source Project on GitHub](https://kcd.im/pull-request)

本地请保持 `python -m pytest -q` 与 `python -m ruff check nlapt nlapt_gui tests` 通过。代码与注释用英文，用户可见文案用中文。

## License

[MIT](LICENSE)。Copyright (c) 2026 storyAura。

## Support

维护者：[storyAura](https://github.com/storyAura)

问题与功能请求请开 [Issue](https://github.com/storyAura/NLapt/issues)。
