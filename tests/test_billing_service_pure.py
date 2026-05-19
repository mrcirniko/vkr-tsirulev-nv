"""Pure-unit tests for billing.service helpers that don't need a DB roundtrip.

`apply_purchase`, `count_generations_this_month`, `mark_payment_status` etc.
are tested separately in test_billing_service_logic.py with mocked sessions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from billing import service
from db.models import FREE_PLAN_CODE


def test_month_start_utc_zeroes_time_components():
    start = service._month_start_utc(datetime(2026, 5, 17, 13, 42, 11, tzinfo=UTC))
    assert start == datetime(2026, 5, 1, 0, 0, 0, tzinfo=UTC)


def test_month_start_utc_default_argument_uses_now():
    # Just verify it doesn't crash and returns the first of *some* month.
    result = service._month_start_utc()
    assert result.day == 1
    assert (result.hour, result.minute, result.second, result.microsecond) == (0, 0, 0, 0)


def test_apply_purchase_rejects_free_plan():
    with pytest.raises(ValueError):
        service.apply_purchase("00000000-0000-0000-0000-000000000001", FREE_PLAN_CODE, "yk-test")


def test_plan_snapshot_from_orm_copies_fields():
    fake = SimpleNamespace(
        code="monthly",
        title="Месяц",
        price_rub=499,
        duration_days=30,
        monthly_generation_limit=None,
        allow_edit=True,
        description_md="- a\n- b",
        display_order=1,
        is_active=True,
    )
    snap = service.PlanSnapshot.from_orm(fake)
    assert snap.code == "monthly"
    assert snap.title == "Месяц"
    assert snap.price_rub == 499.0
    assert snap.allow_edit is True
    assert snap.description_md == "- a\n- b"
    assert snap.display_order == 1
    assert snap.is_active is True


def test_plan_snapshot_handles_missing_optional_fields():
    fake = SimpleNamespace(
        code="custom",
        title="X",
        price_rub=0,
        duration_days=None,
        monthly_generation_limit=None,
        allow_edit=False,
        description_md=None,  # exercise the `or ""` fallback
        display_order=None,  # exercise the `or 0` fallback
        is_active=True,
    )
    snap = service.PlanSnapshot.from_orm(fake)
    assert snap.description_md == ""
    assert snap.display_order == 0
