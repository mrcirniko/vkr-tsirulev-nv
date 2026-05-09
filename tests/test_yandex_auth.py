"""Unit tests for Yandex OAuth user upsert logic (db/crud.py).

No real DB — SessionLocal and session_scope are patched.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager

import db.crud as db_crud
from db.models import User

# ---- fake session ----


class _FakeUserSession:
    def __init__(self, users: list | None = None):
        self._users: list = list(users or [])
        self.added: list = []

    def scalars(self, stmt):
        return _Scalars(self._users)

    def add(self, obj):
        self._users.append(obj)
        self.added.append(obj)

    def flush(self):
        pass

    def refresh(self, obj):
        return obj

    def get(self, model, key):
        for u in self._users:
            if u.id == key:
                return u
        return None

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _Scalars:
    def __init__(self, items):
        self._items = list(items)

    def first(self):
        return self._items[0] if self._items else None

    def all(self):
        return list(self._items)


def _make_user(yandex_id: str | None = "y123", email: str = "test@ya.ru") -> User:
    return User(
        id=uuid.uuid4(),
        yandex_id=yandex_id,
        google_id=None,
        email=email,
        name="Test User",
        picture_url=None,
    )


# ---- upsert_user_by_yandex ----


def test_upsert_creates_new_user_when_not_found(monkeypatch):
    session = _FakeUserSession()
    monkeypatch.setattr(db_crud, "SessionLocal", lambda: session)

    @contextmanager
    def fake_scope():
        yield session

    monkeypatch.setattr(db_crud, "session_scope", fake_scope)

    def _empty_scalars(stmt):
        return _Scalars([])

    session.scalars = _empty_scalars

    user = db_crud.upsert_user_by_yandex("y_new", "new@ya.ru", name="New User")
    assert user.yandex_id == "y_new"
    assert user.email == "new@ya.ru"
    assert user.name == "New User"
    assert len(session.added) == 1


def test_upsert_links_yandex_to_existing_email_user(monkeypatch):
    existing = _make_user(yandex_id=None, email="existing@ya.ru")
    session = _FakeUserSession([existing])
    monkeypatch.setattr(db_crud, "SessionLocal", lambda: session)

    call_count = [0]

    @contextmanager
    def fake_scope():
        yield session

    monkeypatch.setattr(db_crud, "session_scope", fake_scope)

    def _scalars_by_call(stmt):
        call_count[0] += 1
        # First call: lookup by yandex_id → empty; second: lookup by email → existing
        if call_count[0] == 1:
            return _Scalars([])
        return _Scalars([existing])

    session.scalars = _scalars_by_call

    user = db_crud.upsert_user_by_yandex("y_new", "existing@ya.ru")
    assert user.yandex_id == "y_new"
    assert len(session.added) == 0  # no new row created


def test_upsert_returns_existing_user_by_yandex_id(monkeypatch):
    existing = _make_user(yandex_id="y123", email="test@ya.ru")
    session = _FakeUserSession([existing])
    monkeypatch.setattr(db_crud, "SessionLocal", lambda: session)

    @contextmanager
    def fake_scope():
        yield session

    monkeypatch.setattr(db_crud, "session_scope", fake_scope)
    session.scalars = lambda _: _Scalars([existing])

    user = db_crud.upsert_user_by_yandex("y123", "test@ya.ru", name="Updated Name")
    assert user is existing
    assert user.name == "Updated Name"
    assert len(session.added) == 0


def test_upsert_updates_name_and_picture(monkeypatch):
    existing = _make_user()
    existing.name = "Old Name"
    existing.picture_url = None
    session = _FakeUserSession([existing])
    monkeypatch.setattr(db_crud, "SessionLocal", lambda: session)

    @contextmanager
    def fake_scope():
        yield session

    monkeypatch.setattr(db_crud, "session_scope", fake_scope)
    session.scalars = lambda _: _Scalars([existing])

    user = db_crud.upsert_user_by_yandex("y123", "test@ya.ru", name="New Name", picture_url="https://img")
    assert user.name == "New Name"
    assert user.picture_url == "https://img"
