from __future__ import annotations

import contextlib

from fastapi import Depends, HTTPException, Request, status

from db.crud import get_user
from db.models import User


def current_user(request: Request) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    user = get_user(user_id)
    if user is None:
        request.session.pop("user_id", None)
        return None
    return user


def require_user(user: User | None = Depends(current_user)) -> User:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def optional_user(user: User | None = Depends(current_user)) -> User | None:
    return user


def optional_user_from_request(request) -> User | None:
    """Resolve the current user without going through FastAPI's Depends.

    Works for both HTTP Request and WebSocket — both expose `.session`
    populated by Starlette SessionMiddleware. Useful for WS endpoints
    where Depends-based session injection is awkward.
    """
    user_id = request.session.get("user_id") if hasattr(request, "session") else None
    if not user_id:
        return None
    user = get_user(user_id)
    if user is None:
        with contextlib.suppress(Exception):
            request.session.pop("user_id", None)
        return None
    return user
