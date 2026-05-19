"""ORM helpers for the admin NPA workflow and admin-user management.

Kept in app/admin/ to avoid bloating db/crud.py with admin-only queries.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import FREE_PLAN_CODE, AdminUser, NpaSource, NpaStatus, SubscriptionPlan
from db.session import SessionLocal
from rag.source_registry import clear_source_registry_cache


@contextmanager
def _scope() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---- Admin user management ----


def list_admin_users() -> list[AdminUser]:
    with SessionLocal() as session:
        stmt = select(AdminUser).order_by(AdminUser.created_at.asc())
        return list(session.scalars(stmt).all())


def get_admin_user(admin_id: str | UUID) -> AdminUser | None:
    with SessionLocal() as session:
        return session.get(AdminUser, UUID(str(admin_id)))


def create_admin_user(username: str, password_hash: str) -> AdminUser:
    """Insert a new admin_users row. Raises ValueError if username taken."""
    with _scope() as session:
        existing = session.scalars(select(AdminUser).where(AdminUser.username == username)).first()
        if existing is not None:
            raise ValueError(f"Username {username!r} is already taken")
        admin = AdminUser(username=username, password_hash=password_hash)
        session.add(admin)
        session.flush()
        session.refresh(admin)
        return admin


def delete_admin_user(admin_id: str | UUID) -> bool:
    """Delete an admin_users row. Returns True if deleted, False if not found."""
    with _scope() as session:
        admin = session.get(AdminUser, UUID(str(admin_id)))
        if admin is None:
            return False
        session.delete(admin)
        return True


# ---- NPA sources ----


def create_npa_source(
    *,
    created_by: str | UUID | None,
    original_filename: str,
    source_name: str,
    raw_format: str,
    conversion_options: dict,
    raw_txt_s3_key: str,
) -> NpaSource:
    with _scope() as session:
        npa = NpaSource(
            created_by=UUID(str(created_by)) if created_by else None,
            original_filename=original_filename,
            source_name=source_name,
            raw_format=raw_format,
            conversion_options=conversion_options,
            raw_txt_s3_key=raw_txt_s3_key,
            status=NpaStatus.UPLOADED,
        )
        session.add(npa)
        session.flush()
        session.refresh(npa)
        return npa


def get_npa_source(npa_id: str | UUID) -> NpaSource | None:
    with SessionLocal() as session:
        return session.get(NpaSource, UUID(str(npa_id)))


def list_npa_sources() -> list[NpaSource]:
    with SessionLocal() as session:
        stmt = select(NpaSource).order_by(NpaSource.created_at.desc())
        return list(session.scalars(stmt).all())


def update_npa_source(
    npa_id: str | UUID,
    *,
    status: NpaStatus | str | None = None,
    chunks_json_s3_key: str | None = None,
    chunks_count: int | None = None,
    last_indexed_collection: str | None = None,
    last_indexed_with_refs: bool | None = None,
    error_text: str | None = None,
    clear_error: bool = False,
) -> NpaSource:
    with _scope() as session:
        npa = session.get(NpaSource, UUID(str(npa_id)))
        if npa is None:
            raise ValueError(f"NpaSource {npa_id} not found")

        if status is not None:
            npa.status = NpaStatus(status) if isinstance(status, str) else status
        if chunks_json_s3_key is not None:
            npa.chunks_json_s3_key = chunks_json_s3_key
        if chunks_count is not None:
            npa.chunks_count = chunks_count
        if last_indexed_collection is not None:
            npa.last_indexed_collection = last_indexed_collection
        if last_indexed_with_refs is not None:
            npa.last_indexed_with_refs = last_indexed_with_refs
            npa.last_indexed_at = _utcnow()
        if clear_error:
            npa.error_text = None
        elif error_text is not None:
            npa.error_text = error_text
        npa.updated_at = _utcnow()
        session.flush()
        session.refresh(npa)
        # Invalidate registry so chunker/retriever see the new state without an app restart.
        if last_indexed_collection is not None or status is not None:
            clear_source_registry_cache()
        return npa


def delete_npa_source(npa_id: str | UUID) -> NpaSource | None:
    with _scope() as session:
        npa = session.get(NpaSource, UUID(str(npa_id)))
        if npa is None:
            return None
        session.delete(npa)
        clear_source_registry_cache()
        return npa


def get_active_npa_jobs() -> list[NpaSource]:
    """Rows currently in chunking/indexing — used to deny duplicate jobs."""
    with SessionLocal() as session:
        stmt = select(NpaSource).where(NpaSource.status.in_((NpaStatus.CHUNKING, NpaStatus.INDEXING)))
        return list(session.scalars(stmt).all())


def list_plans() -> list[SubscriptionPlan]:
    with SessionLocal() as session:
        stmt = select(SubscriptionPlan).order_by(SubscriptionPlan.display_order, SubscriptionPlan.price_rub)
        return list(session.scalars(stmt).all())


def get_plan(code: str) -> SubscriptionPlan | None:
    with SessionLocal() as session:
        return session.get(SubscriptionPlan, str(code))


def create_plan(
    *,
    code: str,
    title: str,
    price_rub: Decimal | float | str = 0,
    duration_days: int | None = None,
    monthly_generation_limit: int | None = None,
    allow_edit: bool = True,
    description_md: str = "",
    is_active: bool = True,
) -> SubscriptionPlan:
    """Create a new plan. `code` must be a unique slug; `free` is reserved."""
    code_str = (code or "").strip().lower()
    if not code_str:
        raise ValueError("code is required")
    if not code_str.replace("_", "").replace("-", "").isalnum():
        raise ValueError("code must be alphanumeric (with optional - or _)")
    with _scope() as session:
        existing = session.get(SubscriptionPlan, code_str)
        if existing is not None:
            raise ValueError(f"Plan {code_str!r} already exists")
        # New plans land at the bottom of the order; admin can shuffle later.
        max_order = session.scalar(select(_max_display_order())) or 0
        plan = SubscriptionPlan(
            code=code_str,
            title=title,
            price_rub=Decimal(str(price_rub)),
            duration_days=duration_days,
            monthly_generation_limit=monthly_generation_limit,
            allow_edit=allow_edit,
            description_md=description_md,
            display_order=int(max_order) + 1,
            is_active=is_active,
        )
        session.add(plan)
        session.flush()
        session.refresh(plan)
        return plan


def _max_display_order():
    from sqlalchemy import func as sa_func

    return sa_func.max(SubscriptionPlan.display_order)


def update_plan(
    code: str,
    *,
    title: str | None = None,
    price_rub: Decimal | float | str | None = None,
    duration_days: int | None = None,
    monthly_generation_limit: int | None = None,
    allow_edit: bool | None = None,
    description_md: str | None = None,
    is_active: bool | None = None,
    clear_limit: bool = False,
) -> SubscriptionPlan:
    """Mutate the admin-editable fields of a plan. `code` is immutable."""
    code_str = str(code)
    with _scope() as session:
        plan = session.get(SubscriptionPlan, code_str)
        if plan is None:
            raise ValueError(f"Plan {code_str} not found")
        if title is not None:
            plan.title = title
        if price_rub is not None:
            plan.price_rub = Decimal(str(price_rub))
        if duration_days is not None:
            plan.duration_days = int(duration_days) if duration_days > 0 else None
        if clear_limit:
            plan.monthly_generation_limit = None
        elif monthly_generation_limit is not None:
            plan.monthly_generation_limit = int(monthly_generation_limit)
        if allow_edit is not None:
            plan.allow_edit = bool(allow_edit)
        if description_md is not None:
            plan.description_md = str(description_md)
        if is_active is not None:
            plan.is_active = bool(is_active)
        plan.updated_at = _utcnow()
        session.flush()
        session.refresh(plan)
        return plan


def delete_plan(code: str) -> None:
    """Delete a plan. The FREE plan is reserved and cannot be removed."""
    code_str = str(code)
    if code_str == FREE_PLAN_CODE:
        raise ValueError("Free plan cannot be deleted")
    with _scope() as session:
        plan = session.get(SubscriptionPlan, code_str)
        if plan is None:
            return
        session.delete(plan)


def reorder_plans(order: list[str]) -> list[SubscriptionPlan]:
    """Bulk-update display_order to match the given list of codes (left=first).

    Codes not present in the list keep their existing position by being pushed
    after the explicit ones (preserving relative order).
    """
    with _scope() as session:
        all_plans = list(session.scalars(select(SubscriptionPlan)).all())
        by_code = {p.code: p for p in all_plans}
        next_order = 0
        seen: set[str] = set()
        for code in order:
            plan = by_code.get(str(code))
            if plan is None:
                continue
            plan.display_order = next_order
            next_order += 1
            seen.add(plan.code)
        # Append the rest in their previous relative order.
        for plan in sorted(all_plans, key=lambda p: (p.display_order or 0, p.price_rub)):
            if plan.code in seen:
                continue
            plan.display_order = next_order
            next_order += 1
        session.flush()
        for plan in all_plans:
            session.refresh(plan)
        return sorted(all_plans, key=lambda p: p.display_order)


def has_active_job_for(source_name: str, collection: str | None = None) -> bool:
    """True if any row for the same source is currently chunking/indexing."""
    with SessionLocal() as session:
        stmt = select(NpaSource).where(
            NpaSource.source_name == source_name,
            NpaSource.status.in_((NpaStatus.CHUNKING, NpaStatus.INDEXING)),
        )
        if collection is not None:
            # additional narrowing if needed; left for later use
            pass
        return session.scalars(stmt).first() is not None
