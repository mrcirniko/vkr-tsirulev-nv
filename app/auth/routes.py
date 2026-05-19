from __future__ import annotations

import logging
import secrets
from urllib.parse import urlencode

import httpx
from config import settings
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from auth.oauth import google_configured, oauth, yandex_configured
from auth.session import current_user, require_user
from db.crud import upsert_user_by_google, upsert_user_by_yandex
from db.models import User

LOGGER = logging.getLogger("app.auth.routes")

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _frontend_home() -> str:
    if settings.frontend_origins:
        return settings.frontend_origins[0]
    return "/"


@router.get("/me")
def me(user: User | None = Depends(current_user)) -> JSONResponse:
    if user is None:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    return JSONResponse(
        {
            "id": str(user.id),
            "email": user.email,
            "name": user.name,
            "picture_url": user.picture_url,
        }
    )


# ---- Google OAuth ----


@router.get("/google/login")
async def google_login(request: Request):
    if not google_configured():
        raise HTTPException(status_code=503, detail="Google OAuth is not configured")
    return await oauth.google.authorize_redirect(request, settings.google_redirect_uri)


@router.get("/google/callback")
async def google_callback(request: Request):
    if not google_configured():
        raise HTTPException(status_code=503, detail="Google OAuth is not configured")
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        LOGGER.exception("Google OAuth callback failed")
        raise HTTPException(status_code=400, detail="OAuth callback failed") from None

    userinfo = token.get("userinfo") or {}
    if not userinfo:
        try:
            userinfo = await oauth.google.parse_id_token(request, token)
        except Exception:
            LOGGER.exception("Failed to parse Google id_token")
            raise HTTPException(status_code=400, detail="Invalid OAuth response") from None

    google_sub = userinfo.get("sub")
    email = userinfo.get("email")
    if not google_sub or not email:
        raise HTTPException(status_code=400, detail="Google account missing sub/email")

    user = upsert_user_by_google(
        google_id=google_sub,
        email=email,
        name=userinfo.get("name"),
        picture_url=userinfo.get("picture"),
    )
    request.session["user_id"] = str(user.id)
    LOGGER.info("Google login user_id=%s email=%s", user.id, user.email)
    return RedirectResponse(_frontend_home())


# ---- Yandex OAuth ----
# Hand-rolled via httpx — authlib's non-OIDC state mgmt conflicts with Starlette under some proxies,
# and Yandex uses `Authorization: OAuth` (not Bearer) for userinfo.

_YANDEX_AUTH_URL = "https://oauth.yandex.ru/authorize"
_YANDEX_TOKEN_URL = "https://oauth.yandex.ru/token"  # noqa: S105
_YANDEX_USERINFO_URL = "https://login.yandex.ru/info"


@router.get("/yandex/login")
async def yandex_login(request: Request):
    if not yandex_configured():
        raise HTTPException(status_code=503, detail="Yandex OAuth is not configured")
    state = secrets.token_urlsafe(16)
    request.session["yandex_oauth_state"] = state
    params = urlencode(
        {
            "response_type": "code",
            "client_id": settings.yandex_client_id,
            "redirect_uri": settings.yandex_redirect_uri,
            "scope": "login:email login:info login:avatar",
            "state": state,
            "force_confirm": "no",
        }
    )
    return RedirectResponse(f"{_YANDEX_AUTH_URL}?{params}")


@router.get("/yandex/callback")
async def yandex_callback(request: Request):
    if not yandex_configured():
        raise HTTPException(status_code=503, detail="Yandex OAuth is not configured")

    returned_state = request.query_params.get("state", "")
    stored_state = request.session.pop("yandex_oauth_state", None)
    if not stored_state or not secrets.compare_digest(returned_state, stored_state):
        LOGGER.warning("Yandex OAuth state mismatch: stored=%r returned=%r", stored_state, returned_state)
        raise HTTPException(status_code=400, detail="OAuth state mismatch — try logging in again")

    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing code in Yandex callback")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            token_resp = await client.post(
                _YANDEX_TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": settings.yandex_client_id,
                    "client_secret": settings.yandex_client_secret,
                    "redirect_uri": settings.yandex_redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()
    except Exception:
        LOGGER.exception("Yandex token exchange failed")
        raise HTTPException(status_code=400, detail="Yandex token exchange failed") from None

    access_token = token_data.get("access_token", "")
    if not access_token:
        raise HTTPException(status_code=400, detail="No access token in Yandex response")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            info_resp = await client.get(
                _YANDEX_USERINFO_URL,
                params={"format": "json"},
                headers={"Authorization": f"OAuth {access_token}"},
            )
            info_resp.raise_for_status()
            userinfo = info_resp.json()
    except Exception:
        LOGGER.exception("Failed to fetch Yandex userinfo")
        raise HTTPException(status_code=400, detail="Failed to fetch Yandex user info") from None

    yandex_id = str(userinfo.get("id") or "")
    email = (
        userinfo.get("default_email")
        or next(iter(userinfo.get("emails") or []), None)
        # Fallback: Yandex always has login@yandex.ru even without email scope.
        or (f"{userinfo['login']}@yandex.ru" if userinfo.get("login") else None)
    )
    if not yandex_id or not email:
        LOGGER.error(
            "Yandex userinfo missing id/email: id_val=%r email_val=%r login_val=%r",
            userinfo.get("id"),
            userinfo.get("default_email"),
            userinfo.get("login"),
        )
        raise HTTPException(status_code=400, detail="Yandex account missing id/email")

    display_name = userinfo.get("display_name") or userinfo.get("real_name")
    avatar_id = userinfo.get("default_avatar_id")
    picture_url = f"https://avatars.yandex.net/get-yapic/{avatar_id}/islands-200" if avatar_id else None

    user = upsert_user_by_yandex(
        yandex_id=yandex_id,
        email=email,
        name=display_name,
        picture_url=picture_url,
    )
    request.session["user_id"] = str(user.id)
    LOGGER.info("Yandex login user_id=%s email=%s", user.id, user.email)
    return RedirectResponse(_frontend_home())


# ---- Logout ----


@router.post("/logout")
def logout(request: Request, _: User = Depends(require_user)) -> dict:
    request.session.clear()
    return {"status": "ok"}
