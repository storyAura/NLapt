"""Reusable, skip-safe UI animation helpers.

Every helper here is *purely visual polish*. Correctness must never depend on
an animation actually running: when animations are disabled (or a helper is
handed a widget it cannot animate) the helper puts the widget into its final
visual state synchronously and returns ``None``. When animations are enabled it
returns the running :class:`~PySide6.QtCore.QAbstractAnimation` so callers may
keep a reference, connect to ``finished``, or jump it to the end in tests via
``anim.setCurrentTime(anim.duration())``.

Design intent (mirrors the CaptionForge prototype's ``popIn`` / ``jellyIn``
keyframes) is expressed with an overshoot easing curve; the module stays
dependency-light (only ``QtCore`` / ``QtWidgets`` graphics-effect primitives).

Tests disable animations with :func:`set_animations_enabled` so no wall-clock
wait is ever required.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QRect,
)
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget
from shiboken6 import isValid

# Default durations (milliseconds) tuned to feel like the prototype's springs.
FADE_MS = 180
POP_MS = 260
SLIDE_MS = 200

# Module-level master switch. Tests flip this to ``False`` so helpers no-op
# instantly and assert end-state without sleeping.
ANIMATIONS_ENABLED = True


def set_animations_enabled(enabled: bool) -> None:
    """Enable/disable every animation helper process-wide (tests set ``False``)."""
    global ANIMATIONS_ENABLED
    ANIMATIONS_ENABLED = bool(enabled)


def animations_enabled() -> bool:
    """Return whether animation helpers currently produce running animations."""
    return ANIMATIONS_ENABLED


def _opacity_effect(widget: QWidget) -> QGraphicsOpacityEffect:
    """Return the widget's opacity effect, installing one if needed."""
    effect = widget.graphicsEffect()
    if isinstance(effect, QGraphicsOpacityEffect):
        return effect
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    return effect


def _spring() -> QEasingCurve:
    """Overshoot easing approximating the prototype's cubic-bezier springs."""
    curve = QEasingCurve(QEasingCurve.Type.OutBack)
    curve.setOvershoot(1.1)
    return curve


def _fade(widget: QWidget, start: float, end: float, ms: int) -> QPropertyAnimation | None:
    """Shared opacity-fade driver used by :func:`fade_in` / :func:`fade_out`."""
    effect = _opacity_effect(widget)
    if not ANIMATIONS_ENABLED:
        # Jump straight to the final state; no animation object.
        effect.setOpacity(end)
        return None
    effect.setOpacity(start)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(max(0, ms))
    anim.setStartValue(start)
    anim.setEndValue(end)
    anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
    anim.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
    return anim


def fade_in(widget: QWidget, ms: int = FADE_MS) -> QPropertyAnimation | None:
    """Fade ``widget`` from transparent to opaque. Returns ``None`` when disabled."""
    if widget is not None and not widget.isVisible():
        widget.show()
    return _fade(widget, 0.0, 1.0, ms)


def fade_out(widget: QWidget, ms: int = FADE_MS) -> QPropertyAnimation | None:
    """Fade ``widget`` from opaque to transparent. Returns ``None`` when disabled."""
    return _fade(widget, 1.0, 0.0, ms)


def pop_in(widget: QWidget, ms: int = POP_MS) -> QAbstractAnimation | None:
    """Scale-and-fade a widget in (the design's ``popIn``).

    Fades opacity 0 -> 1 in parallel with a subtle geometry pop from a slightly
    shrunk rectangle to the widget's laid-out geometry. Falls back to a plain
    fade when the widget has no usable geometry yet. Returns ``None`` when
    animations are disabled (widget left fully visible at its final geometry).
    """
    if widget is not None and not widget.isVisible():
        widget.show()
    effect = _opacity_effect(widget)
    final = widget.geometry() if widget is not None else QRect()
    if not ANIMATIONS_ENABLED:
        effect.setOpacity(1.0)
        return None
    fade = QPropertyAnimation(effect, b"opacity", widget)
    fade.setDuration(max(0, ms))
    fade.setStartValue(0.0)
    fade.setEndValue(1.0)
    fade.setEasingCurve(QEasingCurve.Type.InOutQuad)
    effect.setOpacity(0.0)
    if final.isEmpty():
        # No geometry to scale against — a pure fade is the graceful fallback.
        fade.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
        return fade
    # Shrink toward the widget's center for the scale-up illusion.
    dx = round(final.width() * 0.06)
    dy = round(final.height() * 0.06)
    start_rect = final.adjusted(dx, dy, -dx, -dy)
    scale = QPropertyAnimation(widget, b"geometry", widget)
    scale.setDuration(max(0, ms))
    scale.setStartValue(start_rect)
    scale.setEndValue(final)
    scale.setEasingCurve(_spring())
    group = QParallelAnimationGroup(widget)
    group.addAnimation(fade)
    group.addAnimation(scale)
    group.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
    return group


def slide_geometry(
    widget: QWidget,
    start: QRect,
    end: QRect,
    ms: int = SLIDE_MS,
) -> QPropertyAnimation | None:
    """Slide ``widget`` from ``start`` to ``end`` geometry.

    Skip-safe: when animations are disabled the widget is moved to ``end``
    synchronously and this returns ``None``. Callers that keep a reference
    may jump a running animation to the end via ``setCurrentTime(duration)``.
    """
    if widget is None:
        return None
    if not ANIMATIONS_ENABLED:
        widget.setGeometry(end)
        return None
    widget.setGeometry(start)
    animation = QPropertyAnimation(widget, b"geometry", widget)
    animation.setDuration(max(0, ms))
    animation.setStartValue(QRect(start))
    animation.setEndValue(QRect(end))
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    animation.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
    return animation


def snapshot_named_rects(
    items: Sequence[tuple[str, QWidget]],
) -> dict[str, QRect]:
    """Record each named widget's current geometry (copied ``QRect``s)."""
    result: dict[str, QRect] = {}
    for key, widget in items:
        if widget is not None:
            result[key] = QRect(widget.geometry())
    return result


def animate_reflow(
    moves: Sequence[tuple[QWidget, QRect, QRect]],
    ms: int = SLIDE_MS,
) -> QParallelAnimationGroup | None:
    """Slide many widgets from start to end rects in parallel.

    Skip-safe: when animations are disabled every widget is moved to its
    end rect synchronously and this returns ``None``.
    """
    if not moves:
        return None
    if not ANIMATIONS_ENABLED:
        for widget, _start, end in moves:
            if widget is not None:
                widget.setGeometry(end)
        return None
    group = QParallelAnimationGroup()
    for widget, start, end in moves:
        if widget is None or start == end:
            continue
        widget.setGeometry(start)
        slide = QPropertyAnimation(widget, b"geometry", widget)
        slide.setDuration(max(0, ms))
        slide.setStartValue(QRect(start))
        slide.setEndValue(QRect(end))
        slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        group.addAnimation(slide)
    if group.animationCount() == 0:
        return None
    group.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
    return group


def flip_reflow(
    items: Sequence[tuple[str, QWidget]],
    previous: Mapping[str, QRect],
    *,
    pop: QWidget | None = None,
    ms: int = SLIDE_MS,
) -> QAbstractAnimation | None:
    """FLIP unmatched-by-identity widgets from ``previous`` rects to now.

    ``pop`` is excluded from the slide and given :func:`pop_in` instead.
    """
    moves: list[tuple[QWidget, QRect, QRect]] = []
    for key, widget in items:
        start = previous.get(key)
        if widget is None or start is None or widget is pop:
            continue
        end = widget.geometry()
        if start != end and not start.isEmpty() and not end.isEmpty():
            moves.append((widget, start, end))
    group = animate_reflow(moves, ms)
    if pop is not None:
        pop_in(pop)
    return group


def finish_animation(animation: QAbstractAnimation | None) -> None:
    """Jump a running animation to its end state (no-op when ``None``)."""
    if animation is None or not isValid(animation):
        return
    animation.setCurrentTime(animation.duration())


def stop_animation(animation: QAbstractAnimation | None) -> None:
    """Stop a held animation; no-op when missing or already deleted."""
    if animation is None or not isValid(animation):
        return
    animation.stop()
