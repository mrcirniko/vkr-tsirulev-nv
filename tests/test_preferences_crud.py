"""DB-layer tests for `db.crud.{get,upsert}_user_preferences` via mocked session."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from db import crud as db_crud


class _FakeSession:
    """Minimal Session stub. `get` returns the stored row, `add` stores it."""

    def __init__(self, row=None):
        self.row = row
        self.added = []

    def get(self, _model, _key):
        return self.row

    def add(self, obj):
        self.added.append(obj)
        self.row = obj

    def flush(self):
        pass

    def refresh(self, obj):
        return obj

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@contextmanager
def _scope_for(session):
    yield session


@pytest.fixture
def patch_session(monkeypatch):
    def install(session):
        monkeypatch.setattr(db_crud, "session_scope", lambda: _scope_for(session))
        monkeypatch.setattr(db_crud, "SessionLocal", lambda: session)
        return session

    return install


def _row(**overrides):
    base = SimpleNamespace(
        user_id=uuid4(),
        contract_generation_policy="legal_only",
        ask_personal_data=True,
        theme="dark",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_get_user_preferences_returns_defaults_for_new_user(patch_session):
    patch_session(_FakeSession(row=None))
    user_id = uuid4()
    prefs = db_crud.get_user_preferences(user_id)
    assert prefs["contract_generation_policy"] == "legal_only"
    assert prefs["ask_personal_data"] is True
    assert prefs["user_id"] == user_id


def test_get_user_preferences_returns_persisted_row(patch_session):
    row = _row(contract_generation_policy="always", ask_personal_data=False)
    patch_session(_FakeSession(row=row))
    prefs = db_crud.get_user_preferences(row.user_id)
    assert prefs["contract_generation_policy"] == "always"
    assert prefs["ask_personal_data"] is False


def test_upsert_user_preferences_creates_row_when_missing(patch_session):
    session = patch_session(_FakeSession(row=None))
    uid = uuid4()
    result = db_crud.upsert_user_preferences(
        uid,
        contract_generation_policy="always",
        ask_personal_data=False,
    )
    assert result["contract_generation_policy"] == "always"
    assert result["ask_personal_data"] is False
    assert len(session.added) == 1


def test_upsert_user_preferences_partial_patch(patch_session):
    row = _row(contract_generation_policy="legal_only", ask_personal_data=True)
    patch_session(_FakeSession(row=row))
    result = db_crud.upsert_user_preferences(row.user_id, ask_personal_data=False)
    # Policy unchanged, only the boolean flipped.
    assert result["contract_generation_policy"] == "legal_only"
    assert result["ask_personal_data"] is False


def test_upsert_user_preferences_rejects_unknown_policy(patch_session):
    patch_session(_FakeSession(row=None))
    with pytest.raises(ValueError):
        db_crud.upsert_user_preferences(uuid4(), contract_generation_policy="hack")


def test_upsert_accepts_string_user_id(patch_session):
    """API hands us str(UUID); helper must coerce to UUID without crashing."""
    patch_session(_FakeSession(row=None))
    result = db_crud.upsert_user_preferences(
        str(uuid4()),
        contract_generation_policy="always_ask",
    )
    assert result["contract_generation_policy"] == "always_ask"
