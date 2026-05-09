"""Pure-unit tests for billing.yookassa_client — webhook parser + enabled flag."""

from __future__ import annotations

import pytest

from billing import yookassa_client


def test_parse_webhook_event_succeeded():
    payload = {
        "event": "payment.succeeded",
        "object": {
            "id": "abc-123",
            "status": "succeeded",
            "metadata": {"user_id": "u1", "plan_code": "monthly"},
        },
    }
    event = yookassa_client.parse_webhook_event(payload)
    assert event.event == "payment.succeeded"
    assert event.payment_id == "abc-123"
    assert event.status == "succeeded"
    assert event.metadata == {"user_id": "u1", "plan_code": "monthly"}


def test_parse_webhook_event_canceled_no_metadata():
    payload = {
        "event": "payment.canceled",
        "object": {"id": "x", "status": "canceled"},
    }
    event = yookassa_client.parse_webhook_event(payload)
    assert event.event == "payment.canceled"
    assert event.metadata == {}


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "string",
        {},
        {"event": "payment.succeeded"},
        {"event": "payment.succeeded", "object": "not-a-dict"},
        {"event": "payment.succeeded", "object": {"status": "succeeded"}},  # missing id
        {"event": "payment.succeeded", "object": {"id": "x"}},  # missing status
    ],
)
def test_parse_webhook_event_rejects_bad_payload(payload):
    with pytest.raises(ValueError):
        yookassa_client.parse_webhook_event(payload)


def test_is_enabled_false_with_empty_creds():
    # Default test env has empty creds (see tests/conftest.py).
    assert yookassa_client.is_enabled() is False


def test_is_enabled_true_when_both_set(monkeypatch):
    # Settings is a frozen dataclass — bypass the frozen check via __setattr__
    # the same way config._validate does when generating a dev session secret.
    from config import settings

    object.__setattr__(settings, "yookassa_shop_id", "shop")
    object.__setattr__(settings, "yookassa_secret_key", "secret")
    try:
        assert yookassa_client.is_enabled() is True
    finally:
        object.__setattr__(settings, "yookassa_shop_id", "")
        object.__setattr__(settings, "yookassa_secret_key", "")
