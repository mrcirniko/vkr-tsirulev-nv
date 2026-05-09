from __future__ import annotations

from authlib.integrations.starlette_client import OAuth
from config import settings

oauth = OAuth()

if settings.google_client_id and settings.google_client_secret:
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


# Yandex OAuth is implemented manually in auth/routes.py (httpx) because
# authlib's non-OIDC state handling conflicts with some Starlette proxy setups.


def google_configured() -> bool:
    return bool(settings.google_client_id and settings.google_client_secret)


def yandex_configured() -> bool:
    return bool(settings.yandex_client_id and settings.yandex_client_secret)
