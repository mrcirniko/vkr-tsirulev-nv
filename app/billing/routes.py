"""User-facing billing API: /api/billing/{me,plans,checkout}.

These endpoints sit behind require_user. The webhook receiver is in
billing/webhooks.py — it must NOT require_user (YooKassa is not a logged-in
session).
"""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal

from config import settings
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from auth.session import require_user
from billing import service, yookassa_client
from db.crud import session_scope
from db.models import FREE_PLAN_CODE, PaymentIntent, PaymentStatus, SubscriptionPlan, User

LOGGER = logging.getLogger("app.billing.routes")

router = APIRouter(prefix="/api/billing", tags=["billing"])


class PlanDTO(BaseModel):
    code: str
    title: str
    price_rub: float
    duration_days: int | None
    monthly_generation_limit: int | None
    allow_edit: bool
    description_md: str
    display_order: int
    is_active: bool


class MeDTO(BaseModel):
    plan: dict
    generations_used_this_month: int
    monthly_generation_limit: int | None
    expires_at: str | None
    subscription_status: str | None
    billing_enabled: bool


class CheckoutRequest(BaseModel):
    plan_code: str = Field(..., description="monthly | yearly | biennial")


class CheckoutResponse(BaseModel):
    payment_id: str
    confirmation_url: str


@router.get("/me", response_model=MeDTO)
def get_me(user: User = Depends(require_user)) -> MeDTO:
    summary = service.usage_summary(user.id)
    return MeDTO(**summary, billing_enabled=settings.billing_enabled)


@router.get("/plans", response_model=list[PlanDTO])
def get_plans(_: User = Depends(require_user)) -> list[PlanDTO]:
    return [
        PlanDTO(
            code=plan.code,
            title=plan.title,
            price_rub=plan.price_rub,
            duration_days=plan.duration_days,
            monthly_generation_limit=plan.monthly_generation_limit,
            allow_edit=plan.allow_edit,
            description_md=plan.description_md,
            display_order=plan.display_order,
            is_active=plan.is_active,
        )
        for plan in service.list_active_plans()
    ]


@router.post("/checkout", response_model=CheckoutResponse, status_code=201)
async def post_checkout(payload: CheckoutRequest, user: User = Depends(require_user)) -> CheckoutResponse:
    if not settings.billing_enabled:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Billing not configured")

    code = (payload.plan_code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="plan_code is required")
    if code == FREE_PLAN_CODE:
        raise HTTPException(status_code=400, detail="Free plan is not purchasable")

    with session_scope() as session:
        plan = session.scalar(select(SubscriptionPlan).where(SubscriptionPlan.code == code))
        if plan is None or not plan.is_active or plan.duration_days is None:
            raise HTTPException(status_code=400, detail="Plan not available for purchase")
        amount: Decimal = Decimal(plan.price_rub)
        plan_title = plan.title

        idempotence_key = uuid.uuid4().hex
        intent = PaymentIntent(
            user_id=user.id,
            plan_code=code,
            amount_rub=amount,
            idempotence_key=idempotence_key,
            status=PaymentStatus.PENDING,
        )
        session.add(intent)
        session.flush()
        intent_id = str(intent.id)

    try:
        created = yookassa_client.create_payment(
            amount_rub=amount,
            description=f"PactumAI {plan_title} ({code})",
            return_url=settings.yookassa_return_url,
            idempotence_key=idempotence_key,
            metadata={
                "user_id": str(user.id),
                "plan_code": code,
                "intent_id": intent_id,
            },
        )
    except Exception as exc:
        LOGGER.exception("YooKassa payment creation failed user_id=%s plan=%s", user.id, code)
        with session_scope() as session:
            failing = session.get(PaymentIntent, uuid.UUID(intent_id))
            if failing is not None:
                failing.status = PaymentStatus.CANCELED
        raise HTTPException(status_code=502, detail="Payment provider error") from exc

    with session_scope() as session:
        persisted = session.get(PaymentIntent, uuid.UUID(intent_id))
        if persisted is not None:
            persisted.yookassa_payment_id = created.id

    LOGGER.info(
        "Created checkout intent=%s payment=%s user_id=%s plan=%s",
        intent_id,
        created.id,
        user.id,
        code,
    )

    if settings.yookassa_auto_confirm:
        # Dev shortcut: skip the YooKassa webhook entirely — apply the purchase
        # immediately so the user comes back from the redirect to an already-
        # active subscription.
        try:
            user_id_uuid, plan_code_str, already = service.mark_payment_status(created.id, PaymentStatus.SUCCEEDED)
            if user_id_uuid is not None and plan_code_str is not None and not already:
                sub = service.apply_purchase(user_id_uuid, plan_code_str, created.id)
                from realtime import emit_to_user

                await emit_to_user(
                    str(user_id_uuid),
                    {
                        "type": "subscription_updated",
                        "plan_code": str(plan_code_str),
                        "expires_at": sub.expires_at.isoformat(),
                    },
                )
                LOGGER.info("Auto-confirmed payment=%s user_id=%s plan=%s", created.id, user.id, code)
        except Exception:
            LOGGER.exception("Auto-confirm failed for payment=%s", created.id)

    return CheckoutResponse(payment_id=created.id, confirmation_url=created.confirmation_url)
