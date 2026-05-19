"""Admin authentication: owner (env-var) + regular admins (DB + bcrypt).

Two tiers:
- Owner: credentials come from OWNER_USERNAME / OWNER_PASSWORD env vars.
  No DB row — auth is done in-memory. Has exclusive access to admin management.
- Admin: stored in admin_users table, password hashed with bcrypt.

Both principals are represented as AdminPrincipal and stored in the Starlette
session under admin_id (UUID string for DB admins, OWNER_ID for owner) and
is_owner flag.
"""

from __future__ import annotations

import logging
import secrets
import uuid
from dataclasses import dataclass

from config import settings
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select

from db.models import AdminUser
from db.session import SessionLocal

LOGGER = logging.getLogger("app.admin.auth")

OWNER_ID = "__owner__"


@dataclass
class AdminPrincipal:
    """Unified principal for both owner and regular admins."""

    id: str  # UUID string for DB admins, OWNER_ID for owner
    username: str
    is_owner: bool

    @property
    def db_id(self) -> uuid.UUID | None:
        """UUID for DB FK columns (e.g. created_by). None for owner."""
        if self.is_owner:
            return None
        try:
            return uuid.UUID(self.id)
        except ValueError:
            return None


# ---- Password helpers ----


def _verify_password(plain: str, hashed: str) -> bool:
    try:
        import bcrypt
    except Exception:
        LOGGER.error("bcrypt is not installed; admin login is disabled")
        return False
    try:
        pw = plain.encode("utf-8")
        if len(pw) > 72:
            pw = pw[:72]
        return bcrypt.checkpw(pw, hashed.encode("utf-8"))
    except Exception:
        return False


def hash_password(plain: str) -> str:
    import bcrypt

    pw = plain.encode("utf-8")
    if len(pw) > 72:
        pw = pw[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def generate_password(length: int = 20) -> str:
    """Generate a URL-safe random password."""
    return secrets.token_urlsafe(length)


# ---- Auth functions ----


def authenticate_owner(username: str, password: str) -> bool:
    if not settings.owner_enabled:
        return False
    return secrets.compare_digest(username, settings.owner_username) and secrets.compare_digest(
        password, settings.owner_password
    )


def authenticate_admin(username: str, password: str) -> AdminUser | None:
    if not username or not password:
        return None
    with SessionLocal() as session:
        admin = session.scalars(select(AdminUser).where(AdminUser.username == username)).first()
        if admin is None:
            return None
        if not _verify_password(password, admin.password_hash):
            return None
        return admin


def get_admin_by_id(admin_id: str) -> AdminUser | None:
    if not admin_id:
        return None
    with SessionLocal() as session:
        try:
            return session.get(AdminUser, uuid.UUID(admin_id))
        except (ValueError, TypeError):
            return None


# ---- FastAPI dependencies ----


def current_admin(request: Request) -> AdminPrincipal | None:
    admin_id = request.session.get("admin_id")
    if not admin_id:
        return None
    if admin_id == OWNER_ID:
        if not request.session.get("is_owner"):
            request.session.pop("admin_id", None)
            return None
        return AdminPrincipal(id=OWNER_ID, username=settings.owner_username, is_owner=True)
    db_admin = get_admin_by_id(str(admin_id))
    if db_admin is None:
        request.session.pop("admin_id", None)
        return None
    return AdminPrincipal(id=str(db_admin.id), username=db_admin.username, is_owner=False)


def require_admin(admin: AdminPrincipal | None = Depends(current_admin)) -> AdminPrincipal:
    if admin is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin auth required")
    return admin


def require_owner(admin: AdminPrincipal | None = Depends(current_admin)) -> AdminPrincipal:
    if admin is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin auth required")
    if not admin.is_owner:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Owner access required")
    return admin
