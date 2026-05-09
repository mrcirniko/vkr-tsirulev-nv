"""Unit tests for admin user management CRUD (admin/crud.py).

DB calls are mocked via monkeypatching — no real Postgres required.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

import admin.crud as admin_crud
from db.models import AdminUser

# ---- fake session infrastructure ----


class _FakeAdminSession:
    def __init__(self, admins: list | None = None):
        self._admins: list = list(admins or [])
        self.deleted: list = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def get(self, model, key):
        if model is not AdminUser:
            return None
        for a in self._admins:
            if a.id == key:
                return a
        return None

    def add(self, obj):
        self._admins.append(obj)

    def delete(self, obj):
        self.deleted.append(obj)
        self._admins = [a for a in self._admins if a.id != obj.id]

    def flush(self):
        pass

    def refresh(self, obj):
        return obj

    def scalars(self, _stmt):
        return _Scalars(self._admins)


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


def _make_admin(username: str = "testadmin") -> AdminUser:
    return AdminUser(
        id=uuid.uuid4(),
        username=username,
        password_hash="$2b$fake",
        created_at=datetime.now(UTC),
    )


# ---- fixtures ----


@pytest.fixture
def with_admins(monkeypatch):
    def install(admins):
        session = _FakeAdminSession(admins)
        monkeypatch.setattr(admin_crud, "_scope", lambda: _scope_for(session))
        monkeypatch.setattr(admin_crud, "SessionLocal", lambda: session)
        return admins, session

    return install


# ---- list_admin_users ----


def test_list_admin_users_empty(with_admins):
    with_admins([])
    result = admin_crud.list_admin_users()
    assert result == []


def test_list_admin_users_returns_all(with_admins):
    admins = [_make_admin("alice"), _make_admin("bob")]
    with_admins(admins)
    result = admin_crud.list_admin_users()
    assert len(result) == 2
    assert {a.username for a in result} == {"alice", "bob"}


# ---- create_admin_user ----


def test_create_admin_user_succeeds(with_admins):
    with_admins([])
    admin = admin_crud.create_admin_user("newadmin", "$2b$hashed")
    assert admin.username == "newadmin"
    assert admin.password_hash == "$2b$hashed"


def test_create_admin_user_raises_on_duplicate_username(with_admins):
    existing = _make_admin("dupe")
    with_admins([existing])
    with pytest.raises(ValueError, match="already taken"):
        admin_crud.create_admin_user("dupe", "$2b$hashed")


# ---- delete_admin_user ----


def test_delete_admin_user_returns_true(with_admins):
    admin = _make_admin("todelete")
    _, session = with_admins([admin])
    result = admin_crud.delete_admin_user(admin.id)
    assert result is True
    assert any(a.id == admin.id for a in session.deleted)


def test_delete_admin_user_returns_false_when_not_found(with_admins):
    with_admins([])
    result = admin_crud.delete_admin_user(uuid.uuid4())
    assert result is False


def test_delete_admin_user_removes_correct_entry(with_admins):
    a1 = _make_admin("keep")
    a2 = _make_admin("remove")
    _, session = with_admins([a1, a2])
    admin_crud.delete_admin_user(a2.id)
    assert any(a.id == a2.id for a in session.deleted)
    assert not any(a.id == a1.id for a in session.deleted)
