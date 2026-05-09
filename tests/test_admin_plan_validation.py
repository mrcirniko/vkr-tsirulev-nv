"""Lightweight tests for admin.crud.update_plan + reorder via mocked session.

Verifies field-level validation, immutable code, markdown description,
display_order reordering rules, and FREE-plan deletion guard.
"""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
from types import SimpleNamespace

import pytest

from admin import crud as admin_crud
from db.models import FREE_PLAN_CODE


class _FakeSession:
    def __init__(self, plan=None, plans_list=None):
        self.plan = plan
        self.plans_list = list(plans_list or [])
        self.deleted = []

    def get(self, _model, key):
        if self.plan is not None and self.plan.code == key:
            return self.plan
        for plan in self.plans_list:
            if plan.code == key:
                return plan
        return None

    def flush(self):
        pass

    def refresh(self, obj):
        return obj

    def add(self, obj):
        self.plans_list.append(obj)

    def delete(self, obj):
        self.deleted.append(obj)
        self.plans_list = [p for p in self.plans_list if p.code != obj.code]

    def scalar(self, _stmt):
        if not self.plans_list:
            return 0
        return max((p.display_order or 0) for p in self.plans_list)

    def scalars(self, _stmt):
        return _Scalars(self.plans_list)


class _Scalars:
    def __init__(self, items):
        self._items = list(items)

    def first(self):
        return self._items[0] if self._items else None

    def all(self):
        return list(self._items)


@contextmanager
def _scope_for(session):
    yield session


@pytest.fixture
def with_plan(monkeypatch):
    def install(plan):
        session = _FakeSession(plan=plan)
        monkeypatch.setattr(admin_crud, "_scope", lambda: _scope_for(session))
        monkeypatch.setattr(admin_crud, "SessionLocal", lambda: session)
        return plan, session

    return install


@pytest.fixture
def with_plans(monkeypatch):
    def install(plans):
        session = _FakeSession(plans_list=plans)
        monkeypatch.setattr(admin_crud, "_scope", lambda: _scope_for(session))
        monkeypatch.setattr(admin_crud, "SessionLocal", lambda: session)
        return plans, session

    return install


def _make_plan(code: str = "monthly", *, order: int = 1):
    return SimpleNamespace(
        code=code,
        title=code.title(),
        price_rub=Decimal("499"),
        duration_days=30,
        monthly_generation_limit=None,
        allow_edit=True,
        features_json=[],
        description_md="",
        display_order=order,
        is_active=True,
        updated_at=None,
    )


def test_update_plan_changes_price(with_plan):
    with_plan(_make_plan())
    result = admin_crud.update_plan("monthly", price_rub=999)
    assert result.price_rub == Decimal("999")


def test_update_plan_clear_limit_wipes_existing(with_plan):
    plan, _session = with_plan(_make_plan())
    plan.monthly_generation_limit = 10
    result = admin_crud.update_plan("monthly", clear_limit=True, monthly_generation_limit=5)
    assert result.monthly_generation_limit is None  # clear wins


def test_update_plan_description_md(with_plan):
    plan, _session = with_plan(_make_plan())
    plan.description_md = "old"
    result = admin_crud.update_plan("monthly", description_md="**bold**")
    assert result.description_md == "**bold**"


def test_update_plan_raises_when_missing(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(admin_crud, "_scope", lambda: _scope_for(session))
    with pytest.raises(ValueError):
        admin_crud.update_plan("monthly", price_rub=1)


def test_update_plan_accepts_arbitrary_code(with_plan):
    with_plan(_make_plan(code="team_yearly"))
    result = admin_crud.update_plan("team_yearly", title="Team Yearly")
    assert result.title == "Team Yearly"


def test_delete_plan_rejects_free():
    with pytest.raises(ValueError):
        admin_crud.delete_plan(FREE_PLAN_CODE)


def test_delete_plan_removes_existing(with_plans):
    _plans, session = with_plans([_make_plan("monthly"), _make_plan("custom", order=2)])
    admin_crud.delete_plan("custom")
    assert any(p.code == "custom" for p in session.deleted)


def test_reorder_plans_moves_codes_to_front(with_plans):
    a = _make_plan("a", order=0)
    b = _make_plan("b", order=1)
    c = _make_plan("c", order=2)
    with_plans([a, b, c])
    result = admin_crud.reorder_plans(["c", "a", "b"])
    assert [p.code for p in result] == ["c", "a", "b"]
    assert [p.display_order for p in result] == [0, 1, 2]


def test_reorder_plans_appends_unlisted_codes_at_end(with_plans):
    a = _make_plan("a", order=0)
    b = _make_plan("b", order=1)
    c = _make_plan("c", order=2)
    with_plans([a, b, c])
    result = admin_crud.reorder_plans(["b"])  # only mention 'b' explicitly
    assert result[0].code == "b"
    assert {p.code for p in result[1:]} == {"a", "c"}


def test_create_plan_rejects_invalid_code(with_plans):
    with_plans([])
    with pytest.raises(ValueError):
        admin_crud.create_plan(code="bad code with spaces", title="x")


def test_create_plan_assigns_display_order_after_existing(with_plans):
    existing = _make_plan("monthly", order=5)
    with_plans([existing])
    new = admin_crud.create_plan(code="custom", title="Custom", duration_days=60)
    assert new.display_order == 6
