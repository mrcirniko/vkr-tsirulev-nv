"""YooKassa webhook receiver.

Mounted at /api/billing/yookassa/webhook. Public endpoint — no auth — but
trusted via the IP allowlist documented at https://yookassa.ru/developers/.
In test mode (YOOKASSA_TEST_MODE=true) we skip the IP check so localhost
curl-tests work; production should keep test_mode off.
"""

from __future__ import annotations

import ipaddress
import logging

from config import settings
from fastapi import APIRouter, HTTPException, Request

from billing import service, yookassa_client
from billing.yookassa_client import WebhookEvent
from db.models import PaymentStatus

LOGGER = logging.getLogger("app.billing.webhook")

router = APIRouter(prefix="/api/billing", tags=["billing"])

# Documented YooKassa notification IPs. See:
# https://yookassa.ru/developers/using-api/webhooks#ip
_YOOKASSA_NETS = tuple(
    ipaddress.ip_network(net)
    for net in (
        "185.71.76.0/27",
        "185.71.77.0/27",
        "77.75.153.0/25",
        "77.75.156.11/32",
        "77.75.156.35/32",
        "77.75.154.128/25",
        "2a02:5180::/32",
    )
)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client is None:
        return ""
    return request.client.host or ""


def _ip_is_allowed(ip_str: str) -> bool:
    if not ip_str:
        return False
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(ip in net for net in _YOOKASSA_NETS)


@router.post("/yookassa/webhook", status_code=200)
async def yookassa_webhook(request: Request) -> dict:
    if not settings.billing_enabled:
        raise HTTPException(status_code=503, detail="Billing not configured")

    if not settings.yookassa_test_mode:
        ip = _client_ip(request)
        if not _ip_is_allowed(ip):
            LOGGER.warning("Rejecting YooKassa webhook from untrusted IP=%s", ip)
            raise HTTPException(status_code=403, detail="Forbidden")

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc

    try:
        event: WebhookEvent = yookassa_client.parse_webhook_event(payload)
    except ValueError as exc:
        LOGGER.warning("Malformed YooKassa webhook: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    LOGGER.info("YooKassa webhook event=%s payment=%s status=%s", event.event, event.payment_id, event.status)

    if event.event == "payment.succeeded":
        user_id, plan_code, already = service.mark_payment_status(event.payment_id, PaymentStatus.SUCCEEDED)
        if user_id is None or plan_code is None:
            LOGGER.warning("Webhook for unknown payment_id=%s; ignoring", event.payment_id)
            return {"status": "ignored"}
        if already:
            return {"status": "duplicate"}
        try:
            sub = service.apply_purchase(user_id, plan_code, event.payment_id)
        except Exception:
            LOGGER.exception("apply_purchase failed for payment=%s", event.payment_id)
            raise HTTPException(status_code=500, detail="Internal error") from None
        # WS push so open tabs update without F5. Imported lazily to dodge
        # circular imports between billing/* and realtime via server.py.
        try:
            from realtime import emit_to_user

            await emit_to_user(
                str(user_id),
                {
                    "type": "subscription_updated",
                    "plan_code": str(plan_code),
                    "expires_at": sub.expires_at.isoformat(),
                },
            )
        except Exception:
            LOGGER.exception("Failed to emit subscription_updated WS event user_id=%s", user_id)
        return {"status": "ok"}

    if event.event == "payment.canceled":
        service.mark_payment_status(event.payment_id, PaymentStatus.CANCELED)
        return {"status": "ok"}

    LOGGER.info("Ignoring YooKassa event=%s", event.event)
    return {"status": "ignored"}
