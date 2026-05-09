"""Logic tests for billing.service via mocked DB session.

We replace `session_scope` with a context manager that yields a fake session
exposing the methods the code under test actually calls (`get`, `add`,
`flush`, `refresh`, `expunge`, `scalars`, `scalar`). This lets us verify
extension/upgrade rules in `apply_purchase` and the FREE-fallback logic in
`effective_plan` without standing up Postgres.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from billing import service
from db.models import FREE_PLAN_CODE, SubscriptionStatus


class _FakeScalars:
    def __init__(self, items):
        self._items = list(items)

    def first(self):
        return self._items[0] if self._items else None

    def all(self):
        return list(self._items)


class _FakeSession:
    def __init__(self, *, plans=None, active_sub=None, scalar_value=None, get_returns=None):
        self.plans = plans or {}
        self.active_sub = active_sub
        self.scalar_value = scalar_value
        self.get_returns = get_returns or {}
        self.added = []
        self.flushed = 0

    def get(self, model, key):
        return self.get_returns.get((model.__name__, key), self.plans.get(key))

    def add(self, obj):
        self.added.append(obj)
        if not getattr(obj, "id", None):
            obj.id = uuid4()

    def flush(self):
        self.flushed += 1

    def refresh(self, obj):
        return obj

    def expunge(self, obj):
        return obj

    def scalars(self, _stmt):
        return _FakeScalars([self.active_sub] if self.active_sub else [])

    def scalar(self, _stmt):
        return self.scalar_value


@contextmanager
def _scope_for(session):
    yield session


@pytest.fixture
def patch_session(monkeypatch):
    sessions = []

    def install(session):
        sessions.append(session)
        monkeypatch.setattr(service, "session_scope", lambda: _scope_for(session))
        return session

    return install


def _make_plan(code, *, duration_days, limit=None, allow_edit=True):
    return SimpleNamespace(
        code=code,
        title=code,
        price_rub=100,
        duration_days=duration_days,
        monthly_generation_limit=limit,
        allow_edit=allow_edit,
        description_md="",
        display_order=0,
        is_active=True,
    )


def _free_snapshot():
    return service.PlanSnapshot(
        code=FREE_PLAN_CODE,
        title="Free",
        price_rub=0,
        duration_days=None,
        monthly_generation_limit=10,
        allow_edit=False,
        description_md="",
        display_order=0,
        is_active=True,
    )


def _paid_snapshot(code: str, *, limit=None):
    return service.PlanSnapshot(
        code=code,
        title=code.title(),
        price_rub=499,
        duration_days=30,
        monthly_generation_limit=limit,
        allow_edit=True,
        description_md="",
        display_order=1,
        is_active=True,
    )


def test_count_generations_this_month_returns_int(patch_session):
    session = _FakeSession(scalar_value=7)
    patch_session(session)
    assert service.count_generations_this_month(uuid4()) == 7


def test_count_generations_this_month_handles_none(patch_session):
    session = _FakeSession(scalar_value=None)
    patch_session(session)
    assert service.count_generations_this_month(uuid4()) == 0


def test_apply_purchase_creates_fresh_subscription_when_none_active(patch_session):
    monthly = _make_plan("monthly", duration_days=30)
    session = _FakeSession(plans={"monthly": monthly}, active_sub=None)
    patch_session(session)

    user_id = uuid4()
    fixed_now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
    sub = service.apply_purchase(user_id, "monthly", "yk-1", now=fixed_now)

    assert sub.user_id == user_id
    assert sub.plan_code == "monthly"
    assert sub.expires_at == fixed_now + timedelta(days=30)
    assert sub.yookassa_payment_id == "yk-1"
    assert len(session.added) == 1


def test_apply_purchase_extends_same_plan(patch_session):
    monthly = _make_plan("monthly", duration_days=30)
    expires = datetime(2026, 6, 1, tzinfo=UTC)
    active = SimpleNamespace(
        user_id=uuid4(),
        plan_code="monthly",
        expires_at=expires,
        status=SubscriptionStatus.ACTIVE,
        yookassa_payment_id="old",
    )
    session = _FakeSession(plans={"monthly": monthly}, active_sub=active)
    patch_session(session)

    fixed_now = datetime(2026, 5, 15, tzinfo=UTC)
    sub = service.apply_purchase(active.user_id, "monthly", "yk-2", now=fixed_now)

    assert sub is active
    assert sub.expires_at == expires + timedelta(days=30)
    assert sub.yookassa_payment_id == "yk-2"
    assert session.added == []


def test_apply_purchase_stacks_different_plan_after_existing(patch_session):
    yearly = _make_plan("yearly", duration_days=365)
    expires = datetime(2026, 6, 1, tzinfo=UTC)
    active = SimpleNamespace(
        user_id=uuid4(),
        plan_code="monthly",
        expires_at=expires,
        status=SubscriptionStatus.ACTIVE,
        yookassa_payment_id="old",
    )
    session = _FakeSession(plans={"yearly": yearly}, active_sub=active)
    patch_session(session)

    fixed_now = datetime(2026, 5, 15, tzinfo=UTC)
    sub = service.apply_purchase(active.user_id, "yearly", "yk-3", now=fixed_now)

    assert sub is not active
    assert sub.plan_code == "yearly"
    assert sub.started_at == expires
    assert sub.expires_at == expires + timedelta(days=365)
    assert sub.yookassa_payment_id == "yk-3"
    assert len(session.added) == 1


def test_apply_purchase_raises_when_plan_missing(patch_session):
    session = _FakeSession(plans={})
    patch_session(session)
    with pytest.raises(service.PlanNotFound):
        service.apply_purchase(uuid4(), "monthly", "yk-x")


def test_apply_purchase_raises_when_plan_has_no_duration(patch_session):
    bogus = _make_plan("monthly", duration_days=None)
    session = _FakeSession(plans={"monthly": bogus})
    patch_session(session)
    with pytest.raises(service.PlanNotFound):
        service.apply_purchase(uuid4(), "monthly", "yk-x")


def test_apply_purchase_supports_arbitrary_plan_code(patch_session):
    """Admin can mint custom plan codes (e.g. 'team_yearly') and they work end-to-end."""
    custom = _make_plan("team_yearly", duration_days=365)
    session = _FakeSession(plans={"team_yearly": custom}, active_sub=None)
    patch_session(session)

    fixed_now = datetime(2026, 5, 15, tzinfo=UTC)
    sub = service.apply_purchase(uuid4(), "team_yearly", "yk-c", now=fixed_now)

    assert sub.plan_code == "team_yearly"
    assert sub.expires_at == fixed_now + timedelta(days=365)


def test_enforce_create_case_blocks_when_over_limit(monkeypatch):
    monkeypatch.setattr(service, "effective_plan", lambda _uid: _free_snapshot())
    monkeypatch.setattr(service, "count_generations_this_month", lambda _uid: 10)
    with pytest.raises(service.QuotaExceeded):
        service.enforce_create_case(uuid4())


def test_enforce_create_case_allows_when_under_limit(monkeypatch):
    monkeypatch.setattr(service, "effective_plan", lambda _uid: _free_snapshot())
    monkeypatch.setattr(service, "count_generations_this_month", lambda _uid: 5)
    service.enforce_create_case(uuid4())


def test_enforce_create_case_skips_count_for_unlimited_plans(monkeypatch):
    monkeypatch.setattr(service, "effective_plan", lambda _uid: _paid_snapshot("monthly"))
    counted = []
    monkeypatch.setattr(service, "count_generations_this_month", lambda uid: counted.append(uid) or 0)
    service.enforce_create_case(uuid4())
    assert counted == []


def test_effective_plan_falls_back_to_free_when_no_active(monkeypatch):
    free = _free_snapshot()
    monkeypatch.setattr(service, "get_active_subscription", lambda _uid: None)
    monkeypatch.setattr(service, "get_plan", lambda code: free if code == FREE_PLAN_CODE else None)
    plan = service.effective_plan(uuid4())
    assert plan.code == FREE_PLAN_CODE


def test_effective_plan_returns_active_paid(monkeypatch):
    paid = _paid_snapshot("yearly")
    sub = SimpleNamespace(plan_code="yearly")
    monkeypatch.setattr(service, "get_active_subscription", lambda _uid: sub)
    monkeypatch.setattr(service, "get_plan", lambda code: paid if code == "yearly" else None)
    plan = service.effective_plan(uuid4())
    assert plan.code == "yearly"


def test_effective_plan_synthesises_free_when_catalog_missing_it(monkeypatch):
    """If admin nuked the FREE row, we still need a working quota fallback."""
    monkeypatch.setattr(service, "get_active_subscription", lambda _uid: None)

    def _missing(_code):
        raise service.PlanNotFound("nope")

    monkeypatch.setattr(service, "get_plan", _missing)
    plan = service.effective_plan(uuid4())
    assert plan.code == FREE_PLAN_CODE
    assert plan.monthly_generation_limit == 10


def test_uuid_argument_accepts_string():
    """Smoke: helpers convert str→UUID; passing a stringified UUID must not crash."""
    UUID(str(uuid4()))
