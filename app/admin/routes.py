"""Admin REST + WebSocket endpoints for the NPA management panel.

All routes (except /login) require a valid admin session. Operator
flow lives in app/admin/tasks.py — these routes just validate inputs,
persist DB rows, and either return JSON or schedule a background job.

Frontend lives in frontend/static/ (vanilla JS, served by nginx):
    /admin/login    -> POST /api/admin/login
    /admin          -> upload UI + source list + index modal
                       uses GET /api/admin/sources, POST /api/admin/upload,
                       POST /api/admin/sources/{id}/index, etc.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from urllib.parse import quote
from uuid import uuid4

from config import settings
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import Response
from pydantic import BaseModel, Field

from admin import crud as admin_crud
from admin import storage as admin_storage
from admin import tasks as admin_tasks
from admin.auth import (
    OWNER_ID,
    AdminPrincipal,
    authenticate_admin,
    authenticate_owner,
    current_admin,
    generate_password,
    get_admin_by_id,
    hash_password,
    require_admin,
    require_owner,
)
from admin.realtime import ADMIN_WS
from agent import deal_types as deal_types_catalog
from db.models import FREE_PLAN_CODE, AdminUser, NpaSource, NpaStatus, SubscriptionPlan
from rag.indexer import GENERAL_GROUP, PRIMAL_GROUP, SECONDARY_GROUP, chunk_text
from rag.source_registry import canonical_source_name

LOGGER = logging.getLogger("app.admin.routes")

router = APIRouter(prefix="/api/admin", tags=["admin"])

VALID_SOURCE_GROUPS = {GENERAL_GROUP, PRIMAL_GROUP, SECONDARY_GROUP}
ALLOWED_EXT = {"txt"}
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._Ѐ-ӿ -]")


def _sanitize_filename(name: str) -> str:
    name = name.strip().replace("\\", "/").split("/")[-1]
    name = _FILENAME_SAFE_RE.sub("_", name)
    return name[:200] or "upload.txt"


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _content_disposition(filename: str) -> str:
    """Build a Content-Disposition value that handles Cyrillic safely.

    HTTP headers are latin-1 only — putting Cyrillic into `filename="..."`
    raises UnicodeEncodeError inside Starlette and bubbles up as 500.
    Per RFC 5987 we provide an ASCII fallback plus a UTF-8 encoded
    `filename*` for modern browsers.
    """
    ascii_fallback = filename.encode("ascii", errors="replace").decode("ascii").replace("?", "_")
    encoded = quote(filename, safe="")
    return f"attachment; filename=\"{ascii_fallback}\"; filename*=UTF-8''{encoded}"


# ---- DTOs ----


class LoginRequest(BaseModel):
    username: str
    password: str


class NpaSourceDTO(BaseModel):
    id: str
    original_filename: str
    source_name: str
    raw_format: str
    status: str
    chunks_count: int | None
    last_indexed_collection: str | None
    last_indexed_with_refs: bool | None
    last_indexed_at: str | None
    error_text: str | None
    created_at: str
    updated_at: str
    has_chunks_json: bool


class IndexRequest(BaseModel):
    source_group: str = Field(..., description="general | primal | secondary")
    recreate_collection: bool = False


class UploadResultItem(BaseModel):
    filename: str
    npa: NpaSourceDTO | None = None
    chunks_preview_count: int | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    results: list[UploadResultItem]


class PlanAdminDTO(BaseModel):
    code: str
    title: str
    price_rub: float
    duration_days: int | None
    monthly_generation_limit: int | None
    allow_edit: bool
    description_md: str
    display_order: int
    is_active: bool
    updated_at: str


class PlanUpdateRequest(BaseModel):
    title: str | None = None
    price_rub: float | None = Field(default=None, ge=0)
    duration_days: int | None = Field(default=None, ge=0)
    monthly_generation_limit: int | None = Field(default=None, ge=0)
    clear_limit: bool = False
    allow_edit: bool | None = None
    description_md: str | None = None
    is_active: bool | None = None


class PlanCreateRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=64)
    title: str = Field(..., min_length=1)
    price_rub: float = Field(default=0.0, ge=0)
    duration_days: int | None = Field(default=None, ge=1)
    monthly_generation_limit: int | None = Field(default=None, ge=0)
    allow_edit: bool = True
    description_md: str = ""
    is_active: bool = True


class PlanReorderRequest(BaseModel):
    order: list[str] = Field(..., description="Plan codes left→right; absent codes stay at the end")


class AdminUserDTO(BaseModel):
    id: str
    username: str
    created_at: str


class CreateAdminRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)


def _serialize_admin_user(admin: AdminUser) -> AdminUserDTO:
    return AdminUserDTO(
        id=str(admin.id),
        username=admin.username,
        created_at=admin.created_at.isoformat(),
    )


def _serialize_plan(plan: SubscriptionPlan) -> PlanAdminDTO:
    return PlanAdminDTO(
        code=plan.code,
        title=plan.title,
        price_rub=float(plan.price_rub),
        duration_days=plan.duration_days,
        monthly_generation_limit=plan.monthly_generation_limit,
        allow_edit=plan.allow_edit,
        description_md=plan.description_md or "",
        display_order=plan.display_order or 0,
        is_active=plan.is_active,
        updated_at=plan.updated_at.isoformat(),
    )


def _serialize(npa: NpaSource) -> NpaSourceDTO:
    return NpaSourceDTO(
        id=str(npa.id),
        original_filename=npa.original_filename,
        source_name=npa.source_name,
        raw_format=npa.raw_format,
        status=npa.status.value if hasattr(npa.status, "value") else str(npa.status),
        chunks_count=npa.chunks_count,
        last_indexed_collection=npa.last_indexed_collection,
        last_indexed_with_refs=npa.last_indexed_with_refs,
        last_indexed_at=npa.last_indexed_at.isoformat() if npa.last_indexed_at else None,
        error_text=npa.error_text,
        created_at=npa.created_at.isoformat(),
        updated_at=npa.updated_at.isoformat(),
        has_chunks_json=bool(npa.chunks_json_s3_key),
    )


# ---- Auth ----


@router.post("/login")
def login(payload: LoginRequest, request: Request) -> dict:
    # Owner takes priority — checked against env vars, no DB lookup.
    if authenticate_owner(payload.username, payload.password):
        request.session["admin_id"] = OWNER_ID
        request.session["is_owner"] = True
        LOGGER.info("Owner logged in username=%s", payload.username)
        return {"id": OWNER_ID, "username": payload.username, "is_owner": True}
    admin = authenticate_admin(payload.username, payload.password)
    if admin is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    request.session["admin_id"] = str(admin.id)
    request.session.pop("is_owner", None)
    LOGGER.info("Admin logged in username=%s", admin.username)
    return {"id": str(admin.id), "username": admin.username, "is_owner": False}


@router.post("/logout")
def logout(request: Request, _: AdminPrincipal = Depends(require_admin)) -> dict:
    request.session.pop("admin_id", None)
    request.session.pop("is_owner", None)
    return {"status": "ok"}


@router.get("/me")
def me(admin: AdminPrincipal | None = Depends(current_admin)) -> dict:
    if admin is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return {"id": admin.id, "username": admin.username, "is_owner": admin.is_owner}


# ---- Source registry (for the upload dropdown) ----


# ---- NPA list / get / delete ----


@router.get("/sources")
def list_sources(_: AdminPrincipal = Depends(require_admin)) -> list[NpaSourceDTO]:
    return [_serialize(npa) for npa in admin_crud.list_npa_sources()]


@router.get("/sources/{npa_id}")
def get_source(npa_id: str, _: AdminPrincipal = Depends(require_admin)) -> NpaSourceDTO:
    npa = admin_crud.get_npa_source(npa_id)
    if npa is None:
        raise HTTPException(404, "Not found")
    return _serialize(npa)


@router.delete("/sources/{npa_id}")
def delete_source(npa_id: str, _: AdminPrincipal = Depends(require_admin)) -> dict:
    npa = admin_crud.get_npa_source(npa_id)
    if npa is None:
        raise HTTPException(404, "Not found")
    if npa.status in (NpaStatus.CHUNKING, NpaStatus.INDEXING):
        raise HTTPException(409, "Cannot delete: job in progress")

    keys_to_delete = [npa.raw_txt_s3_key]
    if npa.chunks_json_s3_key:
        keys_to_delete.append(npa.chunks_json_s3_key)
    try:
        admin_storage.delete_objects(keys_to_delete)
    except Exception:
        LOGGER.exception("S3 delete failed for npa_id=%s; deleting DB row anyway", npa_id)

    admin_crud.delete_npa_source(npa_id)
    return {"status": "deleted"}


@router.get("/sources/{npa_id}/chunks.json")
def download_chunks(npa_id: str, _: AdminPrincipal = Depends(require_admin)) -> Response:
    npa = admin_crud.get_npa_source(npa_id)
    if npa is None or not npa.chunks_json_s3_key:
        raise HTTPException(404, "Chunks JSON not available")
    try:
        body = admin_storage.download_bytes(npa.chunks_json_s3_key)
    except Exception:
        LOGGER.exception("Failed to fetch chunks JSON from S3 npa_id=%s", npa_id)
        raise HTTPException(502, "Storage unavailable") from None
    safe_name = npa.original_filename.rsplit(".", 1)[0]
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": _content_disposition(f"{safe_name}.chunks.json")},
    )


@router.get("/sources/{npa_id}/raw.txt")
def download_raw(npa_id: str, _: AdminPrincipal = Depends(require_admin)) -> Response:
    npa = admin_crud.get_npa_source(npa_id)
    if npa is None:
        raise HTTPException(404, "Not found")
    try:
        body = admin_storage.download_bytes(npa.raw_txt_s3_key)
    except Exception:
        LOGGER.exception("Failed to fetch raw TXT from S3 npa_id=%s", npa_id)
        raise HTTPException(502, "Storage unavailable") from None
    return Response(
        content=body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": _content_disposition(npa.original_filename)},
    )


# ---- Upload (TXT) ----


async def _process_one_upload(file: UploadFile, admin_id, extract_references: bool) -> UploadResultItem:
    raw_filename = _sanitize_filename(file.filename or "upload.txt")
    ext = _ext(raw_filename)
    if ext not in ALLOWED_EXT:
        return UploadResultItem(filename=raw_filename, error=f"Поддерживается только .txt (получили .{ext})")

    # Source name = filename without extension, with underscores → spaces.
    source_name = canonical_source_name(raw_filename)
    if not source_name:
        return UploadResultItem(filename=raw_filename, error="Пустое имя источника")

    max_bytes = settings.admin_npa_max_upload_mb * 1024 * 1024
    body = await file.read(max_bytes + 1)
    if len(body) > max_bytes:
        return UploadResultItem(filename=raw_filename, error=f"Файл больше {settings.admin_npa_max_upload_mb} МБ")
    if not body:
        return UploadResultItem(filename=raw_filename, error="Пустой файл")

    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = body.decode("cp1251")
        except UnicodeDecodeError:
            return UploadResultItem(filename=raw_filename, error="TXT должен быть в UTF-8 или CP1251")

    if not text.strip():
        return UploadResultItem(filename=raw_filename, error="Файл не содержит текста")

    npa_id = uuid4()
    raw_key = admin_storage.raw_txt_key(str(npa_id))
    chunks_key = admin_storage.chunks_json_key(str(npa_id))

    try:
        await asyncio.to_thread(admin_storage.upload_text, raw_key, text)
        chunks = await asyncio.to_thread(
            chunk_text,
            text,
            source_name,
            PRIMAL_GROUP,  # placeholder for is_general flag — actual group
            # picked at indexing time, patched into chunks then.
            "llm" if extract_references else "morphology",
        )
        if not extract_references:
            # Operator opted out of reference detection entirely. Wipe any
            # refs the chunker may have produced from rule-based regexes.
            for chunk in chunks:
                chunk["references"] = []
        chunks_json = json.dumps(chunks, ensure_ascii=False, indent=2)
        await asyncio.to_thread(admin_storage.upload_text, chunks_key, chunks_json, "application/json")
    except Exception as exc:
        LOGGER.exception("Upload pipeline failed for %s", raw_filename)
        return UploadResultItem(filename=raw_filename, error=str(exc))

    from sqlalchemy import insert

    from db.session import engine

    with engine.begin() as conn:
        conn.execute(
            insert(NpaSource).values(
                id=npa_id,
                created_by=admin_id,
                original_filename=raw_filename,
                source_name=source_name,
                raw_format=ext,
                conversion_options={"extract_references": bool(extract_references)},
                raw_txt_s3_key=raw_key,
                chunks_json_s3_key=chunks_key,
                chunks_count=len(chunks),
                status=NpaStatus.READY.value,
            )
        )
    npa = admin_crud.get_npa_source(npa_id)
    return UploadResultItem(
        filename=raw_filename,
        npa=_serialize(npa),
        chunks_preview_count=len(chunks),
    )


@router.post("/upload", response_model=UploadResponse)
async def upload(
    files: list[UploadFile] = File(...),
    extract_references: bool = Form(False),
    admin: AdminPrincipal = Depends(require_admin),
) -> UploadResponse:
    """Multi-file TXT upload. Source name is derived from each filename.

    `extract_references=True` switches chunking to LLM-based reference
    extraction (slow — minutes per file, depends on REFERENCE_LLM_MODEL).
    Without it, we run the deterministic morphology extractor.
    """
    if not files:
        raise HTTPException(400, "No files provided")
    results = [await _process_one_upload(f, admin.db_id, extract_references) for f in files]
    return UploadResponse(results=results)


# ---- Indexing job ----


@router.post("/sources/{npa_id}/index")
async def start_index(
    npa_id: str,
    payload: IndexRequest,
    background_tasks: BackgroundTasks,
    admin: AdminPrincipal = Depends(require_admin),
) -> dict:
    npa = admin_crud.get_npa_source(npa_id)
    if npa is None:
        raise HTTPException(404, "Not found")
    if payload.source_group not in VALID_SOURCE_GROUPS:
        raise HTTPException(400, f"Invalid source_group {payload.source_group!r}")
    if npa.status in (NpaStatus.CHUNKING, NpaStatus.INDEXING):
        raise HTTPException(409, "Job already running for this source")

    background_tasks.add_task(
        admin_tasks.run_indexing_job,
        admin_id=str(admin.id),
        npa_id=str(npa.id),
        source_group=payload.source_group,
        recreate_collection=payload.recreate_collection,
    )
    return {"status": "scheduled"}


# ---- Subscription plans (catalog editor) ----


@router.get("/plans")
def list_plans(_: AdminPrincipal = Depends(require_admin)) -> list[PlanAdminDTO]:
    return [_serialize_plan(plan) for plan in admin_crud.list_plans()]


@router.post("/plans", status_code=201)
def create_plan(
    payload: PlanCreateRequest,
    _: AdminPrincipal = Depends(require_admin),
) -> PlanAdminDTO:
    code_str = payload.code.strip().lower()
    if code_str == FREE_PLAN_CODE:
        raise HTTPException(400, "free is reserved")
    try:
        plan = admin_crud.create_plan(
            code=code_str,
            title=payload.title,
            price_rub=payload.price_rub,
            duration_days=payload.duration_days,
            monthly_generation_limit=payload.monthly_generation_limit,
            allow_edit=payload.allow_edit,
            description_md=payload.description_md,
            is_active=payload.is_active,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _serialize_plan(plan)


@router.patch("/plans/{plan_code}")
def patch_plan(
    plan_code: str,
    payload: PlanUpdateRequest,
    _: AdminPrincipal = Depends(require_admin),
) -> PlanAdminDTO:
    code_str = plan_code.strip().lower()
    if code_str == FREE_PLAN_CODE and payload.price_rub not in (None, 0, 0.0):
        raise HTTPException(400, "Free plan price must be 0")
    try:
        plan = admin_crud.update_plan(
            code_str,
            title=payload.title,
            price_rub=payload.price_rub,
            duration_days=payload.duration_days,
            monthly_generation_limit=payload.monthly_generation_limit,
            clear_limit=payload.clear_limit,
            allow_edit=payload.allow_edit,
            description_md=payload.description_md,
            is_active=payload.is_active,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return _serialize_plan(plan)


@router.delete("/plans/{plan_code}", status_code=204)
def delete_plan_route(
    plan_code: str,
    _: AdminPrincipal = Depends(require_admin),
) -> Response:
    code_str = plan_code.strip().lower()
    try:
        admin_crud.delete_plan(code_str)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(status_code=204)


@router.post("/plans/order")
def reorder_plans(
    payload: PlanReorderRequest,
    _: AdminPrincipal = Depends(require_admin),
) -> list[PlanAdminDTO]:
    plans = admin_crud.reorder_plans([str(c).strip().lower() for c in payload.order])
    return [_serialize_plan(p) for p in plans]


# ---- Supported deal types catalog ----


@router.get("/deal-types")
def get_deal_types(_: AdminPrincipal = Depends(require_admin)) -> dict[str, str]:
    """Return the current supported-deal-types catalog (preferring S3 over fallback)."""
    return deal_types_catalog.get_supported_deal_type_descriptions()


@router.put("/deal-types")
def put_deal_types(
    payload: dict[str, str],
    _: AdminPrincipal = Depends(require_admin),
) -> dict[str, str]:
    """Replace the catalog in S3 wholesale.

    Validation: payload must be a non-empty {str: str} dict. Local cache
    is invalidated immediately; the langgraph_dev container picks the
    change up via TTL refresh within a few seconds.
    """
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(400, "Payload must be a non-empty object")
    if not all(isinstance(k, str) and k.strip() and isinstance(v, str) for k, v in payload.items()):
        raise HTTPException(400, "All keys must be non-empty strings, all values must be strings")

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        admin_storage.upload_text(
            "config/supported_deal_types.json",
            text,
            content_type="application/json; charset=utf-8",
        )
    except Exception as exc:
        LOGGER.exception("Failed to upload deal types JSON to S3")
        raise HTTPException(502, "Failed to save catalog to storage") from exc

    deal_types_catalog.invalidate_cache()
    return payload


# ---- Admin user management (owner only) ----


@router.get("/admins")
def list_admins(_: AdminPrincipal = Depends(require_owner)) -> list[AdminUserDTO]:
    return [_serialize_admin_user(a) for a in admin_crud.list_admin_users()]


@router.post("/admins", status_code=201)
def create_admin(
    payload: CreateAdminRequest,
    _: AdminPrincipal = Depends(require_owner),
) -> dict:
    """Create a new admin and return the plain password once (never stored in plain form)."""
    username = payload.username.strip()
    if not username:
        raise HTTPException(400, "Username cannot be empty")
    plain_password = generate_password()
    password_hash = hash_password(plain_password)
    try:
        admin = admin_crud.create_admin_user(username, password_hash)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    LOGGER.info("Owner created new admin username=%s", username)
    return {
        "id": str(admin.id),
        "username": admin.username,
        "created_at": admin.created_at.isoformat(),
        "password": plain_password,  # shown once; never returned again
    }


@router.delete("/admins/{admin_id}", status_code=204)
def delete_admin(
    admin_id: str,
    _: AdminPrincipal = Depends(require_owner),
) -> Response:
    deleted = admin_crud.delete_admin_user(admin_id)
    if not deleted:
        raise HTTPException(404, "Admin not found")
    LOGGER.info("Owner deleted admin admin_id=%s", admin_id)
    return Response(status_code=204)


# ---- WebSocket for progress ----


@router.websocket("/ws")
async def admin_ws(ws: WebSocket) -> None:
    admin_id = ws.session.get("admin_id") if hasattr(ws, "session") else None
    if not admin_id:
        await ws.close(code=4401)
        return
    # Owner is authenticated via is_owner flag; regular admins via DB lookup.
    if admin_id == OWNER_ID:
        if not ws.session.get("is_owner"):
            await ws.close(code=4401)
            return
    elif get_admin_by_id(str(admin_id)) is None:
        await ws.close(code=4401)
        return
    await ws.accept()
    await ADMIN_WS.register(admin_id, ws)
    try:
        while True:
            # We don't expect inbound messages — just keep the socket open.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        LOGGER.exception("Admin WS error")
    finally:
        await ADMIN_WS.unregister(admin_id, ws)
