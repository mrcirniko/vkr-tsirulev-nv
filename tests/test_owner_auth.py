"""Unit tests for owner authentication logic.

All tests run without a real database or network — auth functions are patched
at the settings level or via monkeypatching.
"""

from __future__ import annotations

# ---- helpers ----


def _make_settings(owner_username: str = "owner", owner_password: str = "secret"):
    from types import SimpleNamespace

    return SimpleNamespace(
        owner_username=owner_username,
        owner_password=owner_password,
        owner_enabled=bool(owner_username and owner_password),
    )


# ---- authenticate_owner ----


def test_authenticate_owner_correct_credentials(monkeypatch):
    import admin.auth as auth_mod

    monkeypatch.setattr(auth_mod, "settings", _make_settings())
    assert auth_mod.authenticate_owner("owner", "secret") is True


def test_authenticate_owner_wrong_password(monkeypatch):
    import admin.auth as auth_mod

    monkeypatch.setattr(auth_mod, "settings", _make_settings())
    assert auth_mod.authenticate_owner("owner", "wrong") is False


def test_authenticate_owner_wrong_username(monkeypatch):
    import admin.auth as auth_mod

    monkeypatch.setattr(auth_mod, "settings", _make_settings())
    assert auth_mod.authenticate_owner("notowner", "secret") is False


def test_authenticate_owner_disabled_when_empty(monkeypatch):
    import admin.auth as auth_mod

    monkeypatch.setattr(auth_mod, "settings", _make_settings(owner_username="", owner_password=""))
    assert auth_mod.authenticate_owner("", "") is False


def test_authenticate_owner_disabled_partial_config(monkeypatch):
    import admin.auth as auth_mod

    monkeypatch.setattr(auth_mod, "settings", _make_settings(owner_username="owner", owner_password=""))
    assert auth_mod.authenticate_owner("owner", "") is False


# ---- AdminPrincipal.db_id ----


def test_admin_principal_db_id_for_regular_admin():
    import uuid

    from admin.auth import AdminPrincipal

    uid = uuid.uuid4()
    principal = AdminPrincipal(id=str(uid), username="admin", is_owner=False)
    assert principal.db_id == uid


def test_admin_principal_db_id_for_owner():
    from admin.auth import OWNER_ID, AdminPrincipal

    principal = AdminPrincipal(id=OWNER_ID, username="owner", is_owner=True)
    assert principal.db_id is None


def test_admin_principal_owner_flag():
    from admin.auth import OWNER_ID, AdminPrincipal

    owner = AdminPrincipal(id=OWNER_ID, username="owner", is_owner=True)
    assert owner.is_owner is True

    regular = AdminPrincipal(id="some-uuid", username="admin", is_owner=False)
    assert regular.is_owner is False


# ---- generate_password ----


def test_generate_password_default_length():
    from admin.auth import generate_password

    pwd = generate_password()
    # token_urlsafe(20) produces ceil(20 * 4/3) ≈ 27 base64url chars
    assert len(pwd) >= 20


def test_generate_password_is_unique():
    from admin.auth import generate_password

    passwords = {generate_password() for _ in range(20)}
    assert len(passwords) == 20


# ---- hash_password / _verify_password ----


def test_hash_and_verify_roundtrip():
    from admin.auth import _verify_password, hash_password

    plain = "my-secret-password"
    hashed = hash_password(plain)
    assert hashed != plain
    assert _verify_password(plain, hashed) is True


def test_verify_wrong_password_fails():
    from admin.auth import _verify_password, hash_password

    hashed = hash_password("correct")
    assert _verify_password("incorrect", hashed) is False


# ---- current_admin (session-based) ----


def _make_request(session: dict):
    from types import SimpleNamespace

    return SimpleNamespace(session=session)


def test_current_admin_returns_none_when_no_session():
    from admin.auth import current_admin

    req = _make_request({})
    assert current_admin(req) is None


def test_current_admin_returns_owner_principal(monkeypatch):
    import admin.auth as auth_mod

    monkeypatch.setattr(auth_mod, "settings", _make_settings())
    req = _make_request({"admin_id": auth_mod.OWNER_ID, "is_owner": True})
    principal = auth_mod.current_admin(req)
    assert principal is not None
    assert principal.is_owner is True
    assert principal.username == "owner"


def test_current_admin_rejects_owner_id_without_flag(monkeypatch):
    import admin.auth as auth_mod

    monkeypatch.setattr(auth_mod, "settings", _make_settings())
    req = _make_request({"admin_id": auth_mod.OWNER_ID})  # no is_owner key
    principal = auth_mod.current_admin(req)
    assert principal is None
    assert "admin_id" not in req.session
