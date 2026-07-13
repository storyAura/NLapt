"""Regression: a preview load reply must not crash after the panel is gone.

Under the full suite an async ``_read_full`` reply could be queued while its
:class:`PreviewPanel` was still alive but its C++ widgets were torn down during
app/window teardown. When the queued slot then fired it touched a deleted
``_SingleView``, raising ``RuntimeError`` and poisoning the *next* test's setup.
The panel now guards the worker callbacks with ``shiboken6.isValid(self)``.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QImage
from shiboken6 import delete, isValid

from nlapt_gui.widgets import preview_panel as preview_module
from nlapt_gui.widgets.preview_panel import PreviewPanel


def test_late_preview_reply_is_ignored_after_delete(controller, monkeypatch) -> None:
    """The captured ``on_done`` no-ops (no RuntimeError) once the panel is gone."""
    captured: list[Callable[[object], None]] = []

    def fake_run_async(pool, fn, *args, on_done=None, on_error=None):  # noqa: ANN001
        # Capture the reply callback instead of running the worker.
        if on_done is not None:
            captured.append(on_done)

    monkeypatch.setattr(preview_module, "run_async", fake_run_async)

    # NOTE: not registered with qtbot.addWidget — this test deletes the C++
    # object itself, so qtbot must not try to close it during teardown.
    panel = PreviewPanel(controller)
    key = controller.keys()[0]
    controller.set_current(key)
    panel.pixmap_for(key)  # triggers _ensure_image -> captures the done callback
    assert captured, "expected a preview load to be scheduled"
    reply = captured[0]

    # Tear the panel's C++ object down the way window/app teardown would.
    delete(panel)
    assert not isValid(panel)

    # The queued reply now fires: it must return quietly, not raise.
    reply(QImage(4, 3, QImage.Format.Format_RGB32))
