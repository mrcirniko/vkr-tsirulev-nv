"""FastAPI composition root.

Owns the app instance, lifespan, middleware stack and the few endpoints
that don't fit any feature domain (health, root, WebSocket, studio link).
Domain endpoints (auth, admin, billing, preferences, cases) are mounted
via include_router from their respective packages.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from config import settings
from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from realtime import WS_MANAGER
from starlette.middleware.sessions import SessionMiddleware

from admin.routes import router as admin_router
from admin.storage import ensure_admin_bucket
from agent.deal_types import seed_s3_if_missing as seed_deal_types_into_s3
from auth.routes import router as auth_router
from auth.session import optional_user_from_request, require_user
from billing.routes import router as billing_router
from billing.webhooks import router as billing_webhook_router
from cases import langgraph as cases_langgraph
from cases.routes import router as cases_router
from db.migrate import cleanup_stuck_processing_messages, reset_stuck_npa_jobs, seed_subscription_plans
from db.models import User
from db.session import engine, init_db
from preferences.routes import router as preferences_router
from rag.preload import preload_startup_models

LOGGER = logging.getLogger("app.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
    init_db()
    cleanup_stuck_processing_messages(engine)
    reset_stuck_npa_jobs(engine)
    try:
        seed_subscription_plans(engine)
    except Exception:
        LOGGER.exception("Failed to seed subscription_plans; billing UI may break")
    try:
        ensure_admin_bucket()
    except Exception:
        LOGGER.exception("Failed to ensure admin NPA bucket; admin uploads may fail")
    try:
        seed_deal_types_into_s3()
    except Exception:
        LOGGER.exception("Failed to seed deal-types catalog; will fall back to bundled JSON")
    preload_startup_models()
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(settings.run_timeout_seconds + 60, connect=10.0))
    app.state.http_client = http_client
    cases_langgraph.set_http_client(http_client)
    LOGGER.info("FastAPI backend started env=%s", settings.env)
    try:
        yield
    finally:
        cases_langgraph.clear_http_client()
        await http_client.aclose()


app = FastAPI(title="Diploma Agent API", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
    session_cookie="diploma_session",
    https_only=settings.session_https_only,
    same_site=settings.session_same_site,
    max_age=14 * 24 * 60 * 60,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.frontend_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PATCH", "PUT", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(billing_router)
app.include_router(billing_webhook_router)
app.include_router(preferences_router)
app.include_router(cases_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    LOGGER.info("HTTP %s %s", request.method, request.url.path)
    response = await call_next(request)
    LOGGER.info("HTTP %s %s status=%s", request.method, request.url.path, response.status_code)
    return response


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def api_root() -> dict:
    return {"service": "PactumAI API", "frontend": settings.frontend_origins[0] if settings.frontend_origins else "/"}


@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Real-time channel: pushes message/case updates to the user's tabs.

    Authenticates via the same Starlette session cookie as HTTP requests.
    Each browser tab opens one WS, and the server fans out events for all
    of that user's cases to it.
    """
    user = optional_user_from_request(websocket)
    if user is None:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    await WS_MANAGER.register(user.id, websocket)
    try:
        while True:
            # Read keeps the socket open and lets disconnects propagate.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        LOGGER.exception("WS error user_id=%s", user.id)
    finally:
        await WS_MANAGER.unregister(user.id, websocket)


@app.get("/api/studio-link")
def studio_link(request: Request, _: User = Depends(require_user)) -> dict:
    host = request.url.hostname or "localhost"
    return {"url": f"https://smith.langchain.com/studio/?baseUrl=http://{host}:2024"}
