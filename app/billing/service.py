"""Billing business logic: plan resolution, quota enforcement, purchase application.

All functions here are sync — they wrap their own session_scope. Callers in
async FastAPI routes wrap them with `await asyncio.to_thread(...)` if the call
is on a hot path. For low-volume admin endpoints we just call directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, select

from db.crud import session_scope
from db.models import (
    FREE_PLAN_CODE,
    Case,
    PaymentIntent,
    PaymentStatus,
    SubscriptionPlan,
    SubscriptionStatus,
    UserSubscription,
)


class BillingError(Exception):
    """Base for billing-domain errors that translate to 4xx HTTP responses."""


class QuotaExceeded(BillingError):
    """Free user hit their monthly generation cap."""


class EditNotAllowed(BillingError):
    """Free user tried to edit a contract."""


class PlanNotFound(BillingError):
    pass


@dataclass(frozen=True)
class PlanSnapshot:
    """Immutable plain-object view of a SubscriptionPlan, safe to pass around."""

    code: str
    title: str
    price_rub: float
    duration_days: int | None
    monthly_generation_limit: int | None
    allow_edit: bool
    description_md: str
    display_order: int
    is_active: bool

    @classmethod
    def from_orm(cls, plan: SubscriptionPlan) -> PlanSnapshot:
        return cls(
            code=plan.code,
            title=plan.title,
            price_rub=float(plan.price_rub),
            duration_days=plan.duration_days,
            monthly_generation_limit=plan.monthly_generation_limit,
            allow_edit=plan.allow_edit,
            description_md=plan.description_md or "",
            display_order=plan.display_order or 0,
            is_active=plan.is_active,
        )


def _ordering():
    """Common ORDER BY for plan listings: by display_order, then price as tiebreaker."""
    return (SubscriptionPlan.display_order, SubscriptionPlan.price_rub)


def list_active_plans() -> list[PlanSnapshot]:
    """Plans visible to end users. Admin endpoints use list_all_plans()."""
    with session_scope() as session:
        rows = session.scalars(
            select(SubscriptionPlan).where(SubscriptionPlan.is_active.is_(True)).order_by(*_ordering())
        ).all()
        return [PlanSnapshot.from_orm(p) for p in rows]


def list_all_plans() -> list[PlanSnapshot]:
    with session_scope() as session:
        rows = session.scalars(select(SubscriptionPlan).order_by(*_ordering())).all()
        return [PlanSnapshot.from_orm(p) for p in rows]


def get_plan(code: str) -> PlanSnapshot:
    code_str = str(code)
    with session_scope() as session:
        plan = session.get(SubscriptionPlan, code_str)
        if plan is None:
            raise PlanNotFound(f"Plan not found: {code_str}")
        return PlanSnapshot.from_orm(plan)


def get_active_subscription(user_id: str | UUID) -> UserSubscription | None:
    """Latest non-expired ACTIVE subscription, or None if user is on free."""
    user_uuid = UUID(str(user_id))
    now = datetime.now(UTC)
    with session_scope() as session:
        stmt = (
            select(UserSubscription)
            .where(
                UserSubscription.user_id == user_uuid,
                UserSubscription.status == SubscriptionStatus.ACTIVE,
                UserSubscription.expires_at > now,
            )
            .order_by(UserSubscription.expires_at.desc())
            .limit(1)
        )
        sub = session.scalars(stmt).first()
        if sub is not None:
            session.expunge(sub)
        return sub


def effective_plan(user_id: str | UUID) -> PlanSnapshot:
    """User's active plan, falling back to FREE when no paid subscription is live.

    If FREE plan was deleted from the catalog (shouldn't happen but defensive),
    fall back to a synthesised "no-op" snapshot so quota gates still work.
    """
    sub = get_active_subscription(user_id)
    code = sub.plan_code if sub is not None else FREE_PLAN_CODE
    try:
        return get_plan(code)
    except PlanNotFound:
        if code == FREE_PLAN_CODE:
            return PlanSnapshot(
                code=FREE_PLAN_CODE,
                title="Бесплатный",
                price_rub=0.0,
                duration_days=None,
                monthly_generation_limit=10,
                allow_edit=False,
                description_md="",
                display_order=0,
                is_active=True,
            )
        raise


def _month_start_utc(now: datetime | None = None) -> datetime:
    moment = now or datetime.now(UTC)
    return moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def count_generations_this_month(user_id: str | UUID, *, now: datetime | None = None) -> int:
    """Cases created by this user since the start of the current calendar month (UTC).

    We treat "1 chat = 1 generation" — that is, a Free user can start at most
    `monthly_generation_limit` cases per month. The created_at column on
    `cases` is naive in this codebase, so we strip tz before comparing.
    """
    user_uuid = UUID(str(user_id))
    month_start = _month_start_utc(now).replace(tzinfo=None)
    with session_scope() as session:
        return int(
            session.scalar(
                select(func.count(Case.id)).where(
                    Case.user_id == user_uuid,
                    Case.created_at >= month_start,
                )
            )
            or 0
        )


def usage_summary(user_id: str | UUID) -> dict:
    """Compose the payload `/api/billing/me` returns to the user."""
    plan = effective_plan(user_id)
    sub = get_active_subscription(user_id)
    used = count_generations_this_month(user_id) if plan.monthly_generation_limit is not None else 0
    return {
        "plan": {
            "code": plan.code,
            "title": plan.title,
            "monthly_generation_limit": plan.monthly_generation_limit,
            "allow_edit": plan.allow_edit,
            "description_md": plan.description_md,
        },
        "generations_used_this_month": used,
        "monthly_generation_limit": plan.monthly_generation_limit,
        "expires_at": sub.expires_at.isoformat() if sub else None,
        "subscription_status": sub.status.value if sub else None,
    }


def enforce_create_case(user_id: str | UUID) -> None:
    """Raise QuotaExceeded if a Free user has already used their monthly cap.

    Called from POST /api/chat before starting a new case. Paid users with
    `monthly_generation_limit=None` skip the count entirely.
    """
    plan = effective_plan(user_id)
    if plan.monthly_generation_limit is None:
        return
    used = count_generations_this_month(user_id)
    if used >= plan.monthly_generation_limit:
        raise QuotaExceeded(
            f"Free plan limit reached: {used}/{plan.monthly_generation_limit} generations used this month"
        )


def apply_purchase(
    user_id: str | UUID,
    plan_code: str,
    yookassa_payment_id: str,
    *,
    now: datetime | None = None,
) -> UserSubscription:
    """Activate or extend a paid subscription.

    Rules:
      - If the user has no active paid subscription: create a fresh row,
        expires_at = now + plan.duration_days.
      - If the user has an active paid subscription:
          * Same plan_code: extend the existing row's expires_at by duration.
          * Different plan_code: create a NEW row whose started_at is the old
            row's expires_at and switch the user to the new plan from there.
            (The old row stays ACTIVE until it naturally expires; effective_plan
            returns the row with the latest expires_at, so the user gets the
            longest-living one.)

    The free plan has duration_days=NULL — calling apply_purchase("free") is a
    programming error and raises ValueError.
    """
    code_str = str(plan_code)
    if code_str == FREE_PLAN_CODE:
        raise ValueError("Cannot purchase the free plan")

    user_uuid = UUID(str(user_id))
    moment = now or datetime.now(UTC)

    with session_scope() as session:
        plan = session.get(SubscriptionPlan, code_str)
        if plan is None or plan.duration_days is None:
            raise PlanNotFound(f"Plan not purchasable: {code_str}")
        duration = timedelta(days=int(plan.duration_days))

        active_stmt = (
            select(UserSubscription)
            .where(
                UserSubscription.user_id == user_uuid,
                UserSubscription.status == SubscriptionStatus.ACTIVE,
                UserSubscription.expires_at > moment,
            )
            .order_by(UserSubscription.expires_at.desc())
            .limit(1)
        )
        active = session.scalars(active_stmt).first()

        if active is None:
            sub = UserSubscription(
                user_id=user_uuid,
                plan_code=code_str,
                started_at=moment,
                expires_at=moment + duration,
                status=SubscriptionStatus.ACTIVE,
                yookassa_payment_id=yookassa_payment_id,
            )
            session.add(sub)
            session.flush()
            session.refresh(sub)
            session.expunge(sub)
            return sub

        if active.plan_code == code_str:
            active.expires_at = active.expires_at + duration
            active.yookassa_payment_id = yookassa_payment_id
            session.flush()
            session.refresh(active)
            session.expunge(active)
            return active

        # Plan upgrade/downgrade: stack the new plan after the existing one.
        sub = UserSubscription(
            user_id=user_uuid,
            plan_code=code_str,
            started_at=active.expires_at,
            expires_at=active.expires_at + duration,
            status=SubscriptionStatus.ACTIVE,
            yookassa_payment_id=yookassa_payment_id,
        )
        session.add(sub)
        session.flush()
        session.refresh(sub)
        session.expunge(sub)
        return sub


def mark_payment_status(
    yookassa_payment_id: str,
    status: PaymentStatus,
) -> tuple[UUID | None, str | None, bool]:
    """Idempotently set payment_intents.status from a webhook event.

    Returns (user_id, plan_code, was_already_in_terminal_state). Webhooks can
    arrive multiple times — caller checks the boolean to skip duplicate
    apply_purchase calls.
    """
    with session_scope() as session:
        intent = session.scalar(select(PaymentIntent).where(PaymentIntent.yookassa_payment_id == yookassa_payment_id))
        if intent is None:
            return None, None, False
        already_terminal = intent.status in {PaymentStatus.SUCCEEDED, PaymentStatus.CANCELED}
        if already_terminal:
            return intent.user_id, intent.plan_code, True
        intent.status = status
        session.flush()
        return intent.user_id, intent.plan_code, False
