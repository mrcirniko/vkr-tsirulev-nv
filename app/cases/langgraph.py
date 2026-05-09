"""HTTP client for the langgraph_dev runtime.

In production the agent graph runs inside a separate `langgraph_dev`
container; FastAPI talks to it over HTTP using these helpers. The shared
`httpx.AsyncClient` is owned by the FastAPI lifespan — it calls
`set_http_client` on startup and `clear_http_client` on shutdown.

The resume helper handles three clarification stages (classification,
data_sufficiency, optional_contract_generation) and an inline yes/no
re-ask path that patches thread state without kicking off a new run.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
from config import settings

from billing import service as billing_service
from db.crud import get_user_preferences

LOGGER = logging.getLogger("cases.langgraph")

_HTTP_CLIENT: httpx.AsyncClient | None = None


def set_http_client(client: httpx.AsyncClient) -> None:
    global _HTTP_CLIENT
    _HTTP_CLIENT = client


def clear_http_client() -> None:
    global _HTTP_CLIENT
    _HTTP_CLIENT = None


def _http_client() -> httpx.AsyncClient:
    if _HTTP_CLIENT is None:
        raise RuntimeError("LangGraph HTTP client not initialized; FastAPI lifespan must call set_http_client first")
    return _HTTP_CLIENT


def state_values(state: dict | None) -> dict:
    if not state:
        return {}
    values = state.get("values")
    return values if isinstance(values, dict) else state


def has_pending_clarification(state: dict | None) -> bool:
    values = state_values(state)
    return bool(values.get("clarification_needed") and values.get("clarification_stage"))


def extract_assistant_text(state: dict | None) -> str:
    values = state_values(state)
    messages = values.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, dict):
            role = message.get("role") or message.get("type")
            content = message.get("content")
            if role in {"assistant", "ai"} and isinstance(content, str) and content.strip():
                return content
            if role in {"assistant", "ai"} and isinstance(content, list):
                joined = "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
                if joined.strip():
                    return joined
    recommendations = values.get("recommendations")
    if isinstance(recommendations, str) and recommendations.strip():
        return recommendations
    clarification = values.get("clarification_question")
    if isinstance(clarification, str) and clarification.strip():
        return clarification
    return "Ответ агента не получен."


async def get_thread_state(case_id: str) -> dict:
    response = await _http_client().get(
        f"{settings.langgraph_api_url}/threads/{case_id}/state",
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    if response.status_code == 404:
        return {}
    response.raise_for_status()
    return response.json()


async def start_fresh_run(
    case_id: str,
    prompt: str,
    merged_description: str,
    is_followup: bool = False,
    user_id: str | None = None,
) -> str | None:
    """Create a new LangGraph run (non-blocking) and return its run_id.

    Uses POST /runs (not /runs/wait) so the call returns immediately —
    the caller is then responsible for awaiting completion via wait_for_run.
    """
    initial_stage = "default" if is_followup else "classify_contract"
    plan_snapshot = await asyncio.to_thread(billing_service.effective_plan, user_id) if user_id is not None else None
    prefs = (
        await asyncio.to_thread(get_user_preferences, user_id)
        if user_id is not None
        else {"contract_generation_policy": "always_ask", "ask_personal_data": True}
    )
    payload = {
        "assistant_id": settings.langgraph_assistant_id,
        "input": {
            "case_id": case_id,
            "deal_description": merged_description,
            "messages": [{"role": "user", "content": prompt}],
            "iteration_count": 0,
            "max_iterations": settings.max_iterations,
            "max_classification_clarifications": settings.max_classification_clarifications,
            "processing_stage": initial_stage,
            "user_plan": plan_snapshot.code if plan_snapshot else "free",
            "allow_edit": plan_snapshot.allow_edit if plan_snapshot else True,
            "contract_generation_policy": prefs["contract_generation_policy"],
            "ask_personal_data": bool(prefs["ask_personal_data"]),
        },
        "config": {"configurable": {"thread_id": case_id}},
        "if_not_exists": "create",
    }
    LOGGER.info("LANGGRAPH start_run thread_id=%s", case_id)
    response = await _http_client().post(
        f"{settings.langgraph_api_url}/threads/{case_id}/runs",
        json=payload,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    response.raise_for_status()
    body = response.json()
    run_id = body.get("run_id") or body.get("id")
    LOGGER.info("LANGGRAPH started thread_id=%s run_id=%s", case_id, run_id)
    return run_id


async def wait_for_run(
    case_id: str,
    run_id: str,
    on_stage_change=None,
) -> None:
    """Block until the LangGraph run finishes (success, error, or cancel).

    Polls GET /threads/{thread_id}/runs/{run_id} every second until the run's
    status leaves the in-progress states. Optionally also polls the thread
    state on each tick and invokes `on_stage_change(stage)` whenever the
    processing_stage value changes — used to push per-stage loader updates
    to the frontend over WS.

    Wrapped by the caller in asyncio.wait_for(timeout=...) for the watchdog.
    """
    in_flight = {"pending", "running"}
    poll_interval = 1.0
    last_stage: str | None = None
    while True:
        response = await _http_client().get(
            f"{settings.langgraph_api_url}/threads/{case_id}/runs/{run_id}",
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        if response.status_code == 404:
            raise RuntimeError(f"LangGraph run vanished: {case_id}/{run_id}")
        response.raise_for_status()
        run = response.json()
        status = (run.get("status") or "").lower()

        if on_stage_change is not None:
            try:
                state = await get_thread_state(case_id)
                stage = state_values(state).get("processing_stage")
                if isinstance(stage, str) and stage and stage != last_stage:
                    last_stage = stage
                    await on_stage_change(stage)
            except Exception:
                LOGGER.debug("stage poll failed thread_id=%s", case_id, exc_info=True)

        if status not in in_flight:
            LOGGER.info(
                "LANGGRAPH run done thread_id=%s run_id=%s status=%s",
                case_id,
                run_id,
                status,
            )
            if status not in {"success", "completed"}:
                raise RuntimeError(f"LangGraph run ended with status={status}")
            return
        await asyncio.sleep(poll_interval)


async def cancel_run(case_id: str, run_id: str) -> None:
    """Best-effort cancel of an in-flight run. Errors are logged, not raised."""
    try:
        await _http_client().post(
            f"{settings.langgraph_api_url}/threads/{case_id}/runs/{run_id}/cancel",
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        LOGGER.info("LANGGRAPH cancelled thread_id=%s run_id=%s", case_id, run_id)
    except Exception:
        LOGGER.warning("Failed to cancel LANGGRAPH run thread_id=%s run_id=%s", case_id, run_id)


async def _classify_yesno(question: str, answer: str) -> str:
    """Ask the LLM to classify the user's free-form answer as yes/no/unclear.

    Returns one of "yes", "no", "unclear". Delegates the judgement to the
    LLM via JSON-formatted output. The langchain call is sync, so we run it
    in a worker thread to keep the FastAPI event loop responsive.
    """
    from agent.nodes import classify_yesno

    return await asyncio.to_thread(classify_yesno, question, answer)


async def start_resume_run(
    case_id: str,
    clarification_answer: str,
    current_state: dict | None,
) -> tuple[str | None, bool]:
    """Resume a thread paused at a clarification.

    Returns a (run_id, handled_inline) tuple:
    - run_id is non-None when a graph run was kicked off and the caller
      should await its completion via wait_for_run.
    - handled_inline=True means the resume turned out to be a yesno re-ask:
      we patched the thread state with a clarifying question without
      starting a run. The caller should pull the new clarification text out
      of state and treat the assistant message as DONE.
    """
    values = state_values(current_state)
    clarification_stage = values.get("clarification_stage")
    if clarification_stage not in {"classification", "data_sufficiency", "optional_contract_generation"}:
        return None, False

    description = str(values.get("deal_description") or "").strip()
    merged_description = (description + "\nУточнение пользователя: " + clarification_answer).strip()

    decision: str | None = None
    if clarification_stage == "optional_contract_generation":
        original_question = (values.get("clarification_question") or "Сгенерировать договор?").strip()
        decision = await _classify_yesno(original_question, clarification_answer)
        if decision == "unclear":
            reask = (
                "Я не уверен, как трактовать ваш ответ. Нужно ли сгенерировать "
                "письменный договор? Ответьте «да» или «нет» (или сформулируйте "
                "развёрнуто, что именно вы хотите получить от агента)."
            )
            update_payload = {
                "values": {
                    "deal_description": merged_description,
                    "clarification_needed": True,
                    "clarification_stage": "optional_contract_generation",
                    "clarification_question": reask,
                    "messages": [
                        {"role": "user", "content": clarification_answer},
                        {"role": "assistant", "content": reask},
                    ],
                },
                "as_node": "ask_clarification",
            }
            client = _http_client()
            update_response = await client.post(
                f"{settings.langgraph_api_url}/threads/{case_id}/state",
                json=update_payload,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
            update_response.raise_for_status()
            LOGGER.info("LANGGRAPH yesno re-ask thread_id=%s", case_id)
            return None, True

    if clarification_stage == "classification":
        next_node = "classify_deal"
    elif clarification_stage == "data_sufficiency":
        next_node = "check_data_sufficiency"
    else:
        next_node = "optional_contract_answer"

    state_update = {
        "deal_description": merged_description,
        "clarification_needed": False,
        "clarification_question": None,
        "clarification_stage": None,
    }
    if clarification_stage == "optional_contract_generation":
        is_yes = decision == "yes"
        state_update["generate_optional_contract"] = is_yes
        state_update["processing_stage"] = "generate_contract" if is_yes else "generate_recommendations"
    else:
        state_update["processing_stage"] = (
            "classify_contract" if clarification_stage == "classification" else "analyze_norms"
        )

    update_payload = {"values": state_update, "as_node": next_node}
    run_payload = {
        "assistant_id": settings.langgraph_assistant_id,
        "input": None,
        "config": {"configurable": {"thread_id": case_id}},
        "if_not_exists": "create",
    }

    LOGGER.info("LANGGRAPH resume thread_id=%s stage=%s", case_id, clarification_stage)
    client = _http_client()
    update_response = await client.post(
        f"{settings.langgraph_api_url}/threads/{case_id}/state",
        json=update_payload,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    update_response.raise_for_status()
    run_response = await client.post(
        f"{settings.langgraph_api_url}/threads/{case_id}/runs",
        json=run_payload,
        timeout=httpx.Timeout(60.0, connect=10.0),
    )
    run_response.raise_for_status()
    body = run_response.json()
    run_id = body.get("run_id") or body.get("id")
    LOGGER.info("LANGGRAPH resume started thread_id=%s run_id=%s", case_id, run_id)
    return run_id, False
