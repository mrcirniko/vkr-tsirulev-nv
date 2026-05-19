"""Business logic for the cases/chat domain.

Spans three concerns: (1) ORM → DTO mapping for cases/messages/versions,
(2) the background chat run pipeline that drives the LangGraph agent, and
(3) WS broadcasts that surface progress and outcomes to the user's open
tabs. Routes are kept thin and delegate here.
"""

from __future__ import annotations

import asyncio
import logging

from config import settings
from fastapi import Request
from realtime import emit_to_user

from cases import langgraph
from cases.docx_paths import canonical_docx_path
from cases.schemas import CaseDetailDTO, CaseSummaryDTO, MessageDTO, VersionDTO
from db.crud import (
    get_case,
    get_contract_versions,
    get_latest_version,
    get_messages,
    update_case_description,
    update_case_status,
    update_case_title,
    update_message,
)
from messages import ERROR_GENERIC_TEXT, ERROR_TIMEOUT_TEXT
from db.models import Case, CaseStatus, ContractVersion, Message, MessageStatus

LOGGER = logging.getLogger("cases.service")

_case_locks: dict[str, asyncio.Lock] = {}
_case_locks_guard = asyncio.Lock()

# Strong refs prevent the event loop's weakref from dropping tasks mid-flight.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def spawn_background(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return task


async def _case_lock(case_id: str) -> asyncio.Lock:
    async with _case_locks_guard:
        lock = _case_locks.get(case_id)
        if lock is None:
            lock = asyncio.Lock()
            _case_locks[case_id] = lock
        return lock


def message_dto(message: Message) -> MessageDTO:
    return MessageDTO(
        id=str(message.id),
        role=message.role.value,
        content=message.content,
        status=message.status.value,
        created_at=message.created_at.isoformat(),
        updated_at=message.updated_at.isoformat(),
        error_text=message.error_text,
    )


def version_dto(version: ContractVersion) -> VersionDTO:
    canonical = canonical_docx_path(str(version.case_id), version.version_number)
    has_docx = bool(version.docx_path) or (canonical is not None and canonical.exists())
    return VersionDTO(
        id=str(version.id),
        version_number=version.version_number,
        created_at=version.created_at.isoformat(),
        content_md=version.content_md,
        docx_url=(f"/api/cases/{version.case_id}/versions/{version.version_number}/download" if has_docx else None),
    )


def case_summary(case: Case) -> CaseSummaryDTO:
    latest = get_latest_version(case.id)
    latest_url = None
    if latest:
        canonical = canonical_docx_path(str(case.id), latest.version_number)
        if latest.docx_path or (canonical is not None and canonical.exists()):
            latest_url = f"/api/cases/{case.id}/versions/{latest.version_number}/download"
    return CaseSummaryDTO(
        id=str(case.id),
        title=case.title,
        status=case.status.value,
        deal_type=case.deal_type,
        created_at=case.created_at.isoformat(),
        updated_at=case.updated_at.isoformat(),
        latest_docx_url=latest_url,
    )


def case_detail(case: Case, request: Request, state: dict | None = None) -> CaseDetailDTO:
    versions = get_contract_versions(case.id)
    messages = get_messages(case.id)
    state_values = langgraph.state_values(state) if state else {}
    host = request.url.hostname or "localhost"
    studio = f"https://smith.langchain.com/studio/?baseUrl=http://{host}:2024"
    summary = case_summary(case)
    return CaseDetailDTO(
        **summary.model_dump(),
        deal_description=case.deal_description,
        messages=[message_dto(message) for message in messages],
        versions=[version_dto(version) for version in versions],
        clarification_needed=bool(state_values.get("clarification_needed", False)),
        clarification_question=state_values.get("clarification_question"),
        processing_stage=state_values.get("processing_stage"),
        studio_url=studio,
    )


def _case_summary_dict(case: Case | None) -> dict:
    if case is None:
        return {}
    return case_summary(case).model_dump()


async def _emit_message_updated(user_id: str, case_id: str, message: Message) -> None:
    await emit_to_user(
        user_id,
        {
            "type": "message_updated",
            "case_id": case_id,
            "message": message_dto(message).model_dump(),
        },
    )


async def _mark_case_error(user_id: str, case_id: str) -> None:
    """Flip case.status to ERROR and broadcast a case_updated event.

    Called from the background task when the run times out, errors out, or
    fails to start. Keeps the sidebar "Требует внимания" indicator in sync
    with the per-message error state. A subsequent successful run flips
    status back to COMPLETED via save_result.
    """
    try:
        case = update_case_status(case_id, CaseStatus.ERROR.value)
        await emit_to_user(
            user_id,
            {
                "type": "case_updated",
                "case_id": case_id,
                "case": _case_summary_dict(case),
            },
        )
    except Exception:
        LOGGER.exception("Failed to mark case as ERROR case_id=%s", case_id)


async def _regenerate_case_title(user_id: str, case_id: str, description: str) -> None:
    """Generate a short chat title via LLM, persist it, broadcast via WS.

    Triggered after a run that changed the deal_type. We pass the entire
    deal_description (merged user input). The LLM call is sync — wrap with
    asyncio.to_thread so we don't block the FastAPI loop.
    """
    description = (description or "").strip()
    if not description:
        return
    try:
        from agent.nodes import generate_case_title

        title = await asyncio.to_thread(generate_case_title, description)
        if not title:
            return
        case = update_case_title(case_id, title)
        LOGGER.info("Updated case title case_id=%s title=%r", case_id, title)
        await emit_to_user(
            user_id,
            {
                "type": "case_title_changed",
                "case_id": case_id,
                "title": case.title,
                "case": _case_summary_dict(case),
            },
        )
    except Exception:
        LOGGER.exception("Failed to regenerate case title case_id=%s", case_id)


def merged_user_description(case_id: str) -> str:
    parts = [
        message.content.strip()
        for message in get_messages(case_id)
        if message.role.value == "user" and message.content.strip()
    ]
    return "\n".join(parts)


async def process_chat_run(
    *,
    user_id: str,
    case_id: str,
    user_prompt: str,
    assistant_message_id: str,
) -> None:
    """Background task: drives the LangGraph run, manages the placeholder
    assistant message, applies a watchdog timeout, broadcasts WS events."""
    lock = await _case_lock(case_id)
    async with lock:
        prev_case = get_case(case_id)
        prev_deal_type = prev_case.deal_type if prev_case else None
        prev_case_status = prev_case.status.value if prev_case else None
        prev_latest_version = get_latest_version(case_id)
        prev_max_version_number = prev_latest_version.version_number if prev_latest_version else 0

        try:
            current_state = await langgraph.get_thread_state(case_id)
            merged_description = merged_user_description(case_id)
            update_case_description(case_id, merged_description)

            run_id: str | None = None
            handled_inline = False
            if current_state and langgraph.has_pending_clarification(current_state):
                run_id, handled_inline = await langgraph.start_resume_run(
                    case_id=case_id,
                    clarification_answer=user_prompt,
                    current_state=current_state,
                )
            else:
                already_completed = bool(current_state and current_state.get("result_saved"))
                run_id = await langgraph.start_fresh_run(
                    case_id=case_id,
                    prompt=user_prompt,
                    merged_description=merged_description,
                    is_followup=already_completed,
                    user_id=user_id,
                )

            if handled_inline:
                # Re-ask path: state was patched directly, no graph to wait
                # for. Just pull the latest assistant text from state.
                state = await langgraph.get_thread_state(case_id)
                final_text = langgraph.extract_assistant_text(state) or "Уточните, пожалуйста, ответ."
                final_message = update_message(
                    assistant_message_id,
                    content=final_text,
                    status=MessageStatus.DONE,
                )
                await _emit_message_updated(user_id, case_id, final_message)
            elif run_id is None:
                # Failed to even kick off the run.
                err_message = update_message(
                    assistant_message_id,
                    content=ERROR_GENERIC_TEXT,
                    status=MessageStatus.ERROR,
                    error_text="Failed to start LangGraph run",
                )
                await _emit_message_updated(user_id, case_id, err_message)
                await _mark_case_error(user_id, case_id)
                return
            else:
                tagged_message = update_message(assistant_message_id, langgraph_run_id=run_id)
                await _emit_message_updated(user_id, case_id, tagged_message)

                async def _stage_emit(stage: str) -> None:
                    await emit_to_user(
                        user_id,
                        {
                            "type": "case_stage_changed",
                            "case_id": case_id,
                            "stage": stage,
                        },
                    )

                try:
                    await asyncio.wait_for(
                        langgraph.wait_for_run(case_id, run_id, on_stage_change=_stage_emit),
                        timeout=settings.run_timeout_seconds,
                    )
                except TimeoutError:
                    LOGGER.warning(
                        "Run timed out case_id=%s run_id=%s after %ss",
                        case_id,
                        run_id,
                        settings.run_timeout_seconds,
                    )
                    await langgraph.cancel_run(case_id, run_id)
                    err_message = update_message(
                        assistant_message_id,
                        content=ERROR_TIMEOUT_TEXT,
                        status=MessageStatus.ERROR,
                        error_text="run_timeout",
                    )
                    await _emit_message_updated(user_id, case_id, err_message)
                    await _mark_case_error(user_id, case_id)
                    return

                state = await langgraph.get_thread_state(case_id)
                final_text = langgraph.extract_assistant_text(state)
                final_message = update_message(
                    assistant_message_id,
                    content=final_text,
                    status=MessageStatus.DONE,
                )
                await _emit_message_updated(user_id, case_id, final_message)
        except Exception as exc:
            LOGGER.exception("Background chat run failed case_id=%s", case_id)
            try:
                err_message = update_message(
                    assistant_message_id,
                    content=ERROR_GENERIC_TEXT,
                    status=MessageStatus.ERROR,
                    error_text=str(exc)[:500],
                )
                await _emit_message_updated(user_id, case_id, err_message)
            except Exception:
                LOGGER.exception("Failed to mark message as ERROR case_id=%s", case_id)
            await _mark_case_error(user_id, case_id)
            return

    refreshed = get_case(case_id)
    if refreshed is None:
        return

    # Emit one WS event per fresh contract version so the right panel updates without F5.
    try:
        all_versions = get_contract_versions(case_id)
        for version in all_versions:
            if version.version_number > prev_max_version_number:
                await emit_to_user(
                    user_id,
                    {
                        "type": "case_version_added",
                        "case_id": case_id,
                        "version": version_dto(version).model_dump(),
                    },
                )
    except Exception:
        LOGGER.exception("Failed to enumerate new versions case_id=%s", case_id)

    if refreshed.status.value != prev_case_status:
        await emit_to_user(
            user_id,
            {
                "type": "case_updated",
                "case_id": case_id,
                "case": _case_summary_dict(refreshed),
            },
        )

    # Emit clarification state unconditionally so the banner appears/hides without a GET.
    try:
        final_state = await langgraph.get_thread_state(case_id)
        state_values = langgraph.state_values(final_state) if final_state else {}
        await emit_to_user(
            user_id,
            {
                "type": "case_clarification_changed",
                "case_id": case_id,
                "clarification_needed": bool(state_values.get("clarification_needed", False)),
                "clarification_question": state_values.get("clarification_question"),
                "processing_stage": state_values.get("processing_stage"),
            },
        )
    except Exception:
        LOGGER.exception("Failed to emit clarification state case_id=%s", case_id)

    new_deal_type = refreshed.deal_type
    if new_deal_type and new_deal_type != prev_deal_type:
        spawn_background(
            _regenerate_case_title(
                user_id=user_id,
                case_id=case_id,
                description=refreshed.deal_description or "",
            )
        )
