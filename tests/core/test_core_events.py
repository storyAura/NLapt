"""Tests for nlapt.core.events (EventBus + Event)."""

from __future__ import annotations

import logging

import pytest

from nlapt.core.errors import ValidationError
from nlapt.core.events import (
    EVT_BATCH_FINISHED,
    EVT_BATCH_PROGRESS,
    EVT_BATCH_STARTED,
    EVT_CAPTION_CHANGED,
    EVT_FILE_SAVED,
    EVT_OPLOG_APPENDED,
    EVT_PENDING_CLEARED,
    EVT_PENDING_SET,
    EVT_SAVE_FAILED,
    EVT_SNAPSHOT_CREATED,
    EVT_STATE_CHANGED,
    Event,
    EventBus,
)


@pytest.fixture(autouse=True)
def _propagate_nlapt_logs():
    """Ensure caplog can see nlapt logs even if configure_logging ran earlier."""
    logger = logging.getLogger("nlapt")
    previous = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = previous


def test_well_known_event_names_are_unique_strings() -> None:
    names = [
        EVT_CAPTION_CHANGED, EVT_STATE_CHANGED, EVT_PENDING_SET, EVT_PENDING_CLEARED,
        EVT_FILE_SAVED, EVT_SAVE_FAILED, EVT_BATCH_STARTED, EVT_BATCH_PROGRESS,
        EVT_BATCH_FINISHED, EVT_SNAPSHOT_CREATED, EVT_OPLOG_APPENDED,
    ]
    assert all(isinstance(n, str) and n for n in names)
    assert len(set(names)) == len(names)


def test_event_payload_is_immutable_copy() -> None:
    source = {"key": "img.jpg", "n": 1}
    event = Event(name="x", payload=source)
    source["n"] = 99
    assert event.payload["n"] == 1
    with pytest.raises(TypeError):
        event.payload["n"] = 2  # type: ignore[index]


def test_event_requires_non_empty_name() -> None:
    with pytest.raises(ValidationError):
        Event(name="", payload={})


def test_subscribe_and_publish_specific_name() -> None:
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe("saved", received.append)
    bus.publish("saved", key="a.txt")
    bus.publish("other", key="b.txt")
    assert [e.name for e in received] == ["saved"]
    assert received[0].payload["key"] == "a.txt"


def test_wildcard_subscription_receives_all() -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(None, lambda e: seen.append(e.name))
    bus.publish("a")
    bus.publish("b", x=1)
    assert seen == ["a", "b"]


def test_unsubscribe_stops_delivery_and_is_idempotent() -> None:
    bus = EventBus()
    seen: list[str] = []
    unsubscribe = bus.subscribe("evt", lambda e: seen.append(e.name))
    bus.publish("evt")
    unsubscribe()
    unsubscribe()  # second call must not raise
    bus.publish("evt")
    assert seen == ["evt"]


def test_handler_exception_is_isolated_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    bus = EventBus()
    calls: list[str] = []

    def broken(event: Event) -> None:
        raise RuntimeError("handler exploded")

    bus.subscribe("evt", broken)
    bus.subscribe("evt", lambda e: calls.append("second"))
    bus.subscribe(None, lambda e: calls.append("wildcard"))

    with caplog.at_level(logging.ERROR, logger="nlapt.core.events"):
        bus.publish("evt", key="k")  # must not raise

    assert calls == ["second", "wildcard"]
    assert any("handler" in rec.message for rec in caplog.records)


def test_publish_rejects_empty_name() -> None:
    bus = EventBus()
    with pytest.raises(ValidationError):
        bus.publish("")


def test_subscribe_validates_inputs() -> None:
    bus = EventBus()
    with pytest.raises(ValidationError):
        bus.subscribe("", lambda e: None)
    with pytest.raises(ValidationError):
        bus.subscribe("evt", "not-callable")  # type: ignore[arg-type]
