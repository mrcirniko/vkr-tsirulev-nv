"""Thin wrapper over the official `yookassa` SDK.

Indirection layer: keeps SDK imports out of route handlers, makes the test
suite easy (one fixture monkey-patches `create_payment` / `parse_webhook_event`
on this module), and centralises the "is billing actually configured?" check
so callers don't repeat the same `if not shop_id ...` bail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from config import settings

LOGGER = logging.getLogger("app.billing.yookassa")

_configured = False


def is_enabled() -> bool:
    return bool(settings.yookassa_shop_id and settings.yookassa_secret_key)


def _ensure_configured() -> None:
    """Lazy one-shot SDK auth bootstrap. Idempotent."""
    global _configured
    if _configured:
        return
    if not is_enabled():
        raise RuntimeError("YooKassa is not configured (empty shop_id or secret_key)")
    from yookassa import Configuration

    Configuration.account_id = settings.yookassa_shop_id
    Configuration.secret_key = settings.yookassa_secret_key
    _configured = True
    LOGGER.info("YooKassa configured shop_id=%s test_mode=%s", settings.yookassa_shop_id, settings.yookassa_test_mode)


@dataclass(frozen=True)
class CreatedPayment:
    id: str
    confirmation_url: str
    status: str


def create_payment(
    *,
    amount_rub: Decimal,
    description: str,
    return_url: str,
    idempotence_key: str,
    metadata: dict,
) -> CreatedPayment:
    """Create a redirect-style Payment in YooKassa.

    `metadata` is opaque to YooKassa but is echoed back on the webhook, so we
    stash {user_id, plan_code, intent_id} there for cross-referencing.
    """
    _ensure_configured()
    from yookassa import Payment

    body = {
        "amount": {"value": f"{Decimal(amount_rub):.2f}", "currency": "RUB"},
        "capture": True,
        "confirmation": {"type": "redirect", "return_url": return_url},
        "description": description,
        "metadata": metadata,
    }
    LOGGER.info("Creating YooKassa payment amount=%s desc=%r meta=%s", body["amount"], description, metadata)
    payment = Payment.create(body, idempotence_key)
    confirmation_url = ""
    confirmation = getattr(payment, "confirmation", None)
    if confirmation is not None:
        confirmation_url = getattr(confirmation, "confirmation_url", "") or ""
    return CreatedPayment(
        id=str(payment.id),
        confirmation_url=confirmation_url,
        status=str(getattr(payment, "status", "") or ""),
    )


@dataclass(frozen=True)
class WebhookEvent:
    event: str  # e.g. "payment.succeeded" / "payment.canceled"
    payment_id: str
    status: str  # "succeeded" / "canceled" / "pending" / etc.
    metadata: dict


def parse_webhook_event(payload: dict) -> WebhookEvent:
    """Validate the shape of a YooKassa notification and project it to a small DTO.

    Raises ValueError for malformed payloads; callers translate to HTTP 400.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    event = payload.get("event")
    obj = payload.get("object") or {}
    if not isinstance(event, str) or not isinstance(obj, dict):
        raise ValueError("missing 'event' or 'object' fields")
    payment_id = obj.get("id")
    status = obj.get("status")
    metadata = obj.get("metadata") or {}
    if not isinstance(payment_id, str) or not payment_id:
        raise ValueError("missing object.id")
    if not isinstance(status, str):
        raise ValueError("missing object.status")
    if not isinstance(metadata, dict):
        metadata = {}
    return WebhookEvent(event=event, payment_id=payment_id, status=status, metadata=metadata)
