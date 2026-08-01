"""Centered, fade-in/fade-out dialogs (spec module 4: 子窗口居中 + 过渡动画).

Every NLapt sub-window opens centered on the main window and fades in/out
instead of popping (设计要求:出现在软件中央,不闪烁). The fade animates
``windowOpacity`` — deliberately NOT ``QGraphicsOpacityEffect``, which
hard-crashes Qt when attached to top-level windows during native resizes.

Skip-safe: when :mod:`nlapt_gui.anim` animations are disabled (tests), the
dialogs open/close synchronously at full opacity; only the centering runs.

Use :class:`CenteredDialog` as the base class for custom dialogs, and
:func:`show_message` / :func:`ask_confirm` instead of the static
``QMessageBox`` helpers.
"""

from __future__ import annotations

from PySide6.QtCore import QAbstractAnimation, QEasingCurve, QPropertyAnimation
from PySide6.QtGui import QGuiApplication, QShowEvent
from PySide6.QtWidgets import QDialog, QMessageBox, QWidget

from nlapt.diagnostics import get_logger

from nlapt_gui import anim

_LOGGER = get_logger(__name__)

# Fade durations (ms) — quick enough to feel instant, long enough to read as
# a transition instead of a flash.
DIALOG_FADE_IN_MS = 160
DIALOG_FADE_OUT_MS = 120

# Default confirm-dialog button texts (UI stays Chinese).
CONFIRM_OK_TEXT = "确定"
CONFIRM_CANCEL_TEXT = "取消"


def center_on_host(widget: QWidget) -> None:
    """Move ``widget`` so it is centered on its parent window (or screen)."""
    parent = widget.parentWidget()
    host = parent.window() if parent is not None else None
    frame = widget.frameGeometry()
    if host is not None and host.isVisible():
        center = host.frameGeometry().center()
    else:
        screen = widget.screen() or QGuiApplication.primaryScreen()
        if screen is None:  # headless safety
            return
        center = screen.availableGeometry().center()
    frame.moveCenter(center)
    widget.move(frame.topLeft())


class _FadeMixin:
    """Shared open/close fade behavior for QDialog subclasses.

    First show: center on the parent window, then fade ``windowOpacity``
    0 → 1. ``done()``: fade 1 → 0, then really close. Both steps collapse to
    synchronous no-ops when animations are disabled.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._fade_shown_once = False
        self._fade_closing = False
        self._fade_anim: QPropertyAnimation | None = None

    def _start_fade(self, start: float, end: float, ms: int, on_done) -> None:
        if self._fade_anim is not None:
            self._fade_anim.stop()
        animation = QPropertyAnimation(self, b"windowOpacity", self)
        animation.setDuration(ms)
        animation.setStartValue(start)
        animation.setEndValue(end)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.finished.connect(on_done)
        self._fade_anim = animation
        animation.start(QAbstractAnimation.DeletionPolicy.KeepWhenStopped)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)  # type: ignore[misc]
        if self._fade_shown_once:
            return
        self._fade_shown_once = True
        center_on_host(self)  # type: ignore[arg-type]
        if not anim.animations_enabled():
            return
        self.setWindowOpacity(0.0)  # type: ignore[attr-defined]
        self._start_fade(
            0.0, 1.0, DIALOG_FADE_IN_MS, lambda: self.setWindowOpacity(1.0)  # type: ignore[attr-defined]
        )

    def prepare_reshow(self) -> None:
        """Arm centering + the open fade again on a reused dialog instance.

        A still-running close fade is stopped first (Qt emits ``finished``
        on stop, so the pending hide completes synchronously) and both
        one-shot flags reset — the next ``show()`` behaves like a fresh
        first open.
        """
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim = None
        self._fade_shown_once = False
        self._fade_closing = False

    def done(self, result: int) -> None:  # noqa: A003 - Qt override
        if (
            self._fade_closing
            or not anim.animations_enabled()
            or not self.isVisible()  # type: ignore[attr-defined]
        ):
            super().done(result)  # type: ignore[misc]
            return
        self._fade_closing = True
        self._start_fade(
            float(self.windowOpacity()),  # type: ignore[attr-defined]
            0.0,
            DIALOG_FADE_OUT_MS,
            lambda: super(_FadeMixin, self).done(result),  # type: ignore[misc]
        )


class CenteredDialog(_FadeMixin, QDialog):
    """Base class for NLapt dialogs: centered on the app, fade in/out."""


class FadingMessageBox(_FadeMixin, QMessageBox):
    """QMessageBox variant with the same centered fade open/close."""


def show_message(
    parent: QWidget | None,
    title: str,
    text: str,
    *,
    icon: QMessageBox.Icon = QMessageBox.Icon.Information,
) -> None:
    """Centered, fading information dialog (replaces QMessageBox.information)."""
    box = FadingMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()


def ask_confirm(
    parent: QWidget | None,
    title: str,
    text: str,
    *,
    confirm_text: str = CONFIRM_OK_TEXT,
    cancel_text: str = CONFIRM_CANCEL_TEXT,
) -> bool:
    """Centered, fading confirm dialog; True when the user confirms."""
    box = FadingMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    confirm = box.addButton(confirm_text, QMessageBox.ButtonRole.AcceptRole)
    box.addButton(cancel_text, QMessageBox.ButtonRole.RejectRole)
    box.exec()
    return box.clickedButton() is confirm
