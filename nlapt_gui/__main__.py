"""GUI entry point: ``python -m nlapt_gui`` / the ``nlapt-gui`` script.

Bootstraps logging + crash handling under ``app_data_dir()/logs``, loads the
persisted UI settings and the core LLM config, builds the controller /
theme / main window and reopens the last dataset root when it still exists.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from nlapt.app import NLaptApp
from nlapt.core.config import AppConfig, load_config
from nlapt.core.errors import NLaptError
from nlapt.diagnostics import configure_logging, get_logger
from nlapt.diagnostics.crash import install_crash_handler

from nlapt_gui import __version__
from nlapt_gui.resources import app_data_dir

APP_NAME = "NLapt"
ORG_NAME = "NLapt"
LOG_DIR_NAME = "logs"
# Windows taskbar identity: without an explicit AppUserModelID, python.exe
# groups the window under the Python icon instead of the NLapt one.
APP_USER_MODEL_ID = "NLapt.NLapt.GUI"

_LOGGER = get_logger(__name__)


def _set_windows_app_id() -> None:
    """Give the process its own taskbar identity (Windows only, best-effort)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(  # type: ignore[attr-defined]
            APP_USER_MODEL_ID
        )
    except Exception:  # cosmetic only — never block startup
        _LOGGER.exception("could not set AppUserModelID")


def _load_app_config() -> AppConfig:
    """The persisted core config (LLM profiles); defaults when absent/corrupt."""
    # Imported lazily so config_path stays single-sourced with the dialog.
    from nlapt_gui.widgets.settings_dialog import config_path

    path = config_path()
    if not path.exists():
        return AppConfig()
    try:
        return load_config(path)
    except NLaptError:
        _LOGGER.exception("corrupt app config %s; starting with defaults", path)
        return AppConfig()


def main(argv: Sequence[str] | None = None) -> int:
    """Build and run the NLapt GUI application."""
    log_dir = app_data_dir() / LOG_DIR_NAME
    configure_logging(log_dir)
    install_crash_handler(log_dir)
    _LOGGER.info("NLapt GUI %s starting", __version__)

    # Qt imports happen after logging so import-time failures are recorded.
    from PySide6.QtWidgets import QApplication

    from nlapt_gui.controller import AppController
    from nlapt_gui.settings import load_ui_settings
    from nlapt_gui.theme.logo import load_app_icon
    from nlapt_gui.theme.manager import ThemeManager
    from nlapt_gui.widgets.main_window import MainWindow

    _set_windows_app_id()
    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationVersion(__version__)

    ui_settings = load_ui_settings()
    controller = AppController(NLaptApp(config=_load_app_config()), settings=ui_settings)
    theme_manager = ThemeManager(app)
    theme_manager.apply(ui_settings.theme, ui_settings.accent or None)

    # Application (taskbar) icon: packaging/icon.ico when present, else the
    # theme-aware painted logo; the window keeps its own copy that follows
    # theme changes (see MainWindow).
    app.setWindowIcon(load_app_icon(theme_manager.tokens))

    window = MainWindow(controller, theme_manager)
    window.show()

    if ui_settings.last_root:
        last_root = Path(ui_settings.last_root)
        if last_root.is_dir():
            controller.open_dataset(last_root)
        else:
            _LOGGER.info("last root %s no longer exists; skipping reopen", last_root)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
