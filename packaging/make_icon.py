"""Generate ``packaging/icon.ico`` from the NLapt vector logo.

Renders the theme-default logo mark at every standard icon size and writes a
multi-resolution .ico via Pillow. Re-run after logo changes; replace the
generated ``icon.ico`` with any hand-made file to swap the app icon (the
runtime and the PyInstaller spec both point at ``packaging/icon.ico``).

Usage (from the repository root)::

    python packaging/make_icon.py
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ICON_PATH = Path(__file__).resolve().parent / "icon.ico"
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PIL import Image
    from PySide6.QtCore import QBuffer
    from PySide6.QtWidgets import QApplication

    from nlapt_gui.theme.logo import make_app_icon
    from nlapt_gui.theme.tokens import DEFAULT_THEME, THEMES

    app = QApplication.instance() or QApplication([])
    tokens = THEMES[DEFAULT_THEME]
    frames: list[Image.Image] = []
    for size in ICO_SIZES:
        pixmap = make_app_icon(tokens, size).pixmap(size, size)
        buffer = QBuffer()
        buffer.open(QBuffer.OpenModeFlag.ReadWrite)
        pixmap.save(buffer, "PNG")
        frames.append(Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGBA"))
        buffer.close()
    largest = frames[-1]
    largest.save(
        ICON_PATH,
        format="ICO",
        sizes=[(size, size) for size in ICO_SIZES],
        append_images=frames[:-1],
    )
    print(f"wrote {ICON_PATH} ({ICON_PATH.stat().st_size} bytes)")
    del app
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
