"""HTTP endpoints for the cases/chat domain.

Thin controllers — parsing, ownership checks, billing gate, fan-out to
service. The actual graph run happens in `service.process_chat_run`.
"""

from __future__ import annotations

import asyncio
import logging

from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response
from realtime import emit_to_user
from storage import DOCX_CONTENT_TYPE, download_object, is_s3_uri

from auth.session import require_user
from billing import service as billing_service
from cases import langgraph, service
from cases.docx_paths import canonical_docx_path, resolve_docx_path
from cases.schemas import (
    CaseDetailDTO,
    CaseSummaryDTO,
    ChatRequest,
    ChatStartResponseDTO,
    CreateCaseRequest,
    VersionDTO,
)
from db.crud import (
    add_message,
    case_belongs_to_owner,
    create_case,
    get_case,
    get_contract_versions,
    get_latest_message,
    list_cases_for_owner,
)
from db.models import Case, MessageStatus, User

LOGGER = logging.getLogger("cases.routes")

router = APIRouter()


def _ensure_owner(case_id: str, user: User) -> Case:
    case = get_case(case_id)
    if case is None or case.user_id != user.id:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.get("/api/cases", response_model=list[CaseSummaryDTO])
def api_list_cases(user: User = Depends(require_user)) -> list[CaseSummaryDTO]:
    return [service.case_summary(case) for case in list_cases_for_owner(user.id)]


@router.post("/api/cases", response_model=CaseDetailDTO)
async def api_create_case(
    payload: CreateCaseRequest,
    request: Request,
    user: User = Depends(require_user),
) -> CaseDetailDTO:
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    title = prompt[:80] if prompt else "Новый кейс"
    case = create_case(title=title, deal_description=prompt, owner_id=user.id)
    return service.case_detail(case, request, {"processing_stage": "default"})


@router.get("/api/cases/{case_id}", response_model=CaseDetailDTO)
async def api_get_case(case_id: str, request: Request, user: User = Depends(require_user)) -> CaseDetailDTO:
    case = _ensure_owner(case_id, user)
    state = await langgraph.get_thread_state(case_id)
    return service.case_detail(case, request, state)


@router.get("/api/cases/{case_id}/versions", response_model=list[VersionDTO])
def api_get_versions(case_id: str, user: User = Depends(require_user)) -> list[VersionDTO]:
    _ensure_owner(case_id, user)
    return [service.version_dto(version) for version in get_contract_versions(case_id)]


@router.get("/api/cases/{case_id}/versions/{version_number}/download")
def api_download_version(
    case_id: str,
    version_number: int,
    user: User = Depends(require_user),
):
    _ensure_owner(case_id, user)
    versions = get_contract_versions(case_id)
    version = next((item for item in versions if item.version_number == version_number), None)
    if version is None:
        raise HTTPException(status_code=404, detail="DOCX not found")

    if not version.docx_path:
        canonical = canonical_docx_path(case_id, version_number)
        if canonical is not None and canonical.exists():
            return FileResponse(canonical, media_type=DOCX_CONTENT_TYPE, filename=canonical.name)
        raise HTTPException(status_code=404, detail="DOCX file missing")

    if is_s3_uri(version.docx_path):
        try:
            content = download_object(version.docx_path)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
            if error_code in {"NoSuchKey", "NoSuchBucket", "404"}:
                LOGGER.warning("DOCX missing in S3 uri=%s code=%s", version.docx_path, error_code)
                raise HTTPException(status_code=404, detail="DOCX file missing") from exc
            LOGGER.exception("S3 error fetching DOCX uri=%s", version.docx_path)
            raise HTTPException(status_code=502, detail="Storage backend error") from exc
        except Exception:
            LOGGER.exception("Unexpected error fetching DOCX uri=%s", version.docx_path)
            raise HTTPException(status_code=502, detail="Storage backend error") from None
        return Response(
            content,
            media_type=DOCX_CONTENT_TYPE,
            headers={"Content-Disposition": f'attachment; filename="v{version_number}.docx"'},
        )

    canonical = canonical_docx_path(case_id, version_number)
    if canonical is not None and canonical.exists():
        return FileResponse(canonical, media_type=DOCX_CONTENT_TYPE, filename=canonical.name)

    path = resolve_docx_path(version.docx_path)
    if path is None:
        raise HTTPException(status_code=404, detail="DOCX file missing")
    return FileResponse(path, media_type=DOCX_CONTENT_TYPE, filename=path.name)


@router.post("/api/chat", response_model=ChatStartResponseDTO, status_code=202)
async def api_chat(
    payload: ChatRequest,
    user: User = Depends(require_user),
) -> ChatStartResponseDTO:
    """Schedule processing of a new chat message.

    Returns immediately (202 Accepted) with the user message (DONE) and a
    placeholder assistant message (PROCESSING). The actual graph run happens
    in a background task; status updates are pushed to the WebSocket.
    """
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    case_id = payload.case_id
    new_case = False
    if case_id is None:
        # Quota gate; edit-mode for free users is blocked downstream by gate_free_plan node.
        try:
            await asyncio.to_thread(billing_service.enforce_create_case, user.id)
        except billing_service.QuotaExceeded as exc:
            raise HTTPException(
                status_code=402,
                detail={"error": "quota_exceeded", "message": str(exc)},
            ) from None
        case = create_case(title=prompt[:80], deal_description=prompt, owner_id=user.id)
        case_id = str(case.id)
        new_case = True
        LOGGER.info("Created case for API chat case_id=%s user=%s", case_id, user.id)
    elif not case_belongs_to_owner(case_id, user.id):
        raise HTTPException(status_code=404, detail="Case not found")

    # Server-side guard against stale tabs sending a second message while one is in flight.
    latest = get_latest_message(case_id)
    if latest is not None and latest.status == MessageStatus.PROCESSING:
        raise HTTPException(
            status_code=409,
            detail="A previous message is still being processed for this case",
        )

    user_message = add_message(case_id, "user", prompt, status=MessageStatus.DONE)
    assistant_message = add_message(
        case_id,
        "assistant",
        "",
        status=MessageStatus.PROCESSING,
    )

    user_dto = service.message_dto(user_message)
    assistant_dto = service.message_dto(assistant_message)

    # Push WS so other tabs see the new chat and placeholder bubble immediately.
    if new_case:
        await emit_to_user(
            user.id,
            {
                "type": "case_created",
                "case_id": case_id,
                "case": service.case_summary(get_case(case_id)).model_dump(),
            },
        )
    await emit_to_user(
        user.id,
        {"type": "message_added", "case_id": case_id, "message": user_dto.model_dump()},
    )
    await emit_to_user(
        user.id,
        {"type": "message_added", "case_id": case_id, "message": assistant_dto.model_dump()},
    )

    service.spawn_background(
        service.process_chat_run(
            user_id=str(user.id),
            case_id=case_id,
            user_prompt=prompt,
            assistant_message_id=str(assistant_message.id),
        )
    )
    return ChatStartResponseDTO(
        case_id=case_id,
        user_message=user_dto,
        assistant_message=assistant_dto,
    )
