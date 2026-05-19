from __future__ import annotations

import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from uuid import UUID

from config import settings
from contract_docx import contract_html_to_text, generate_docx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from messages import ERROR_GENERIC_TEXT
from storage import upload_contract_docx

from agent.deal_types import get_supported_deal_types
from agent.json_utils import coerce_json
from agent.prompts import (
    BUILD_RETRIEVAL_QUERY_PROMPT,
    CHECK_GENERAL_NORMS_PROMPT,
    CHECK_WRITTEN_FORM_PROMPT,
    CLASSIFY_FOLLOWUP_INTENT_PROMPT,
    CLASSIFY_YESNO_PROMPT,
    CONTRACT_HTML_BODY_TEMPLATE,
    DETECT_VAGUE_REFERENCES_IN_NORMS_PROMPT,
    DETECT_VAGUE_REFERENCES_PROMPT,
    EDIT_CONTRACT_PROMPT,
    FILTER_RETRIEVED_NORMS_PROMPT,
    FOLLOWUP_RESPONSE_PROMPT,
    FOLLOWUP_SUBAGENT_PROMPT,
    FOLLOWUP_TRIAGE_PROMPT,
    GENERATE_CASE_TITLE_PROMPT,
    GENERATE_CONTRACT_PROMPT,
    GENERATE_RECOMMENDATIONS_PROMPT,
    INFORM_USER_PROMPT,
    INTEGRATE_ENRICHMENTS_PROMPT,
    ITERATIVE_RELEVANCE_JUDGE_PROMPT,
    ITERATIVE_RETRIEVAL_PLANNER_PROMPT,
    LEGAL_ASSISTANT_SYSTEM_PROMPT,
    RAG_SUBAGENT_PROMPT,
    SECTION_LABEL_DEAL_DESCRIPTION,
    SECTION_LABEL_DEAL_STRUCTURE,
    SECTION_LABEL_DEAL_TYPE,
    SECTION_LABEL_GENERAL_NORMS,
    SECTION_LABEL_HISTORY,
    SECTION_LABEL_NORMS,
    SECTION_LABEL_SPECIFIC_NORMS,
    SECTION_LABEL_VALIDATION_ERRORS,
    VALIDATE_CONTRACT_PROMPT,
    build_check_data_sufficiency_prompt,
    build_classify_deal_prompt,
    build_classify_rag_assist_prompt,
    build_inform_unsupported_deal_prompt,
)
from agent.state import ContractAgentState
from db.crud import get_latest_version, save_contract_version, session_scope, update_case_status
from db.models import Case, CaseStatus
from rag.chunker import parse_reference
from rag.ollama import unload_ollama_model
from rag.retriever import (
    fetch_reference_chunks_by_payload,
    rerank_chunks,
    retrieve_general,
    retrieve_secondary,
    retrieve_specific,
    search_primal_bi_encoder,
)
from rag.source_registry import infer_specific_source_filter

LOGGER = logging.getLogger("agent.nodes")
DEFAULT_TOP_K = settings.retrieval_context_top_k
LOG_SEPARATOR = "=" * 40
RETRYABLE_CLASSIFICATION_CONFIDENCES = {"low", "medium"}
STAGE_DEFAULT = "default"
STAGE_CLASSIFY_CONTRACT = "classify_contract"
STAGE_RETRIEVE_NORMS = "retrieve_norms"
STAGE_ANALYZE_NORMS = "analyze_norms"
STAGE_GENERATE_RECOMMENDATIONS = "generate_recommendations"
STAGE_ENRICH_RECOMMENDATIONS = "enrich_recommendations"
STAGE_GENERATE_CONTRACT = "generate_contract"
STAGE_EDIT_CONTRACT = "edit_contract"
STAGE_VALIDATE_CONTRACT = "validate_contract"
STAGE_FOLLOWUP_SUBAGENT = "followup_subagent"
NEW_VERSION_SAVED_AS_DOCX_MESSAGE = "Договор сохранен в виде DOCX — откройте его в панели «Договоры» справа."
FOLLOWUP_SUBAGENT_GIVE_UP_MESSAGE = (
    "К сожалению, в доступных мне источниках законодательства РФ не нашлось "
    "однозначного ответа на ваш вопрос. Попробуйте уточнить вопрос или "
    "сформулировать его иначе."
)


_DUMP_FILENAME_SAFE_RE = re.compile(r"[^\w.-]+", re.UNICODE)


def _dump_final_exchange(
    filename_kind: str,
    deal_type: str | None,
    system_prompt: str,
    human_prompt: str,
    response_text: str,
) -> None:
    """Persist the full final LLM exchange to {model}_{deal_type}_{kind}.txt.

    Toggled via settings.llm_dump_final_enabled. Files are overwritten so the
    artifact tracks the latest run for the (model, deal_type) pair.
    """
    if not settings.llm_dump_final_enabled:
        return
    try:
        deal = _DUMP_FILENAME_SAFE_RE.sub("_", (deal_type or "unknown").strip()).strip("_") or "unknown"
        model = _DUMP_FILENAME_SAFE_RE.sub("_", settings.llm_model).strip("_") or "model"
        out_dir = Path(settings.llm_dump_final_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{model}_{deal}_{filename_kind}.txt"
        body = (
            f"{LOG_SEPARATOR}\nLLM REQUEST\nSYSTEM:\n{system_prompt}\n\n"
            f"HUMAN:\n{human_prompt}\n{LOG_SEPARATOR}\n"
            f"LLM RESPONSE\n{response_text}\n{LOG_SEPARATOR}\n"
        )
        path.write_text(body, encoding="utf-8")
        LOGGER.info("Dumped final LLM exchange (%s) to %s", filename_kind, path)
    except Exception as exc:
        LOGGER.warning("Failed to dump final LLM exchange (%s): %s", filename_kind, exc)


def _log_llm_exchange(kind: str, system_prompt: str, human_prompt: str, response_text: str) -> None:
    LOGGER.info(
        "%s\nLLM %s REQUEST\nSYSTEM:\n%s\n\nHUMAN:\n%s\n%s\nLLM %s RESPONSE\n%s\n%s",
        LOG_SEPARATOR,
        kind,
        system_prompt,
        human_prompt,
        LOG_SEPARATOR,
        kind,
        response_text,
        LOG_SEPARATOR,
    )


@lru_cache(maxsize=1)
def _llm() -> ChatOllama:
    kwargs: dict[str, object] = {
        "model": settings.llm_model,
        "base_url": settings.ollama_base_url,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "reasoning": settings.llm_reasoning,
        "num_ctx": settings.llm_num_ctx,
        "num_predict": settings.llm_num_predict,
    }
    if settings.memory_swap_mode:
        kwargs["keep_alive"] = settings.ollama_keep_alive
    LOGGER.info(
        "Initializing ChatOllama model=%s num_ctx=%s num_predict=%s",
        settings.llm_model,
        settings.llm_num_ctx,
        settings.llm_num_predict,
    )
    return ChatOllama(**kwargs)


@lru_cache(maxsize=1)
def _llm_json() -> ChatOllama:
    """Same model bound to Ollama's JSON-mode (format='json').

    Ollama guarantees the response is a syntactically valid JSON object —
    no regex extraction needed.
    """
    return _llm().bind(format="json")


def _invoke_messages(
    messages: list[object],
    *,
    json_mode: bool = False,
    kind: str | None = None,
) -> str:
    client = _llm_json() if json_mode else _llm()
    try:
        response = client.invoke(messages)
    finally:
        if settings.memory_swap_mode:
            unload_ollama_model(settings.llm_model)
    _log_token_usage(kind, response)
    if isinstance(response.content, str):
        return response.content
    return str(response.content)


def _log_token_usage(kind: str | None, response: object) -> None:
    """Log exact token counts reported by Ollama for a single LLM call.

    Ollama returns prompt_eval_count / eval_count in the chat response
    payload; LangChain surfaces them via response.response_metadata. These
    are the real counts from the inference engine, not heuristic estimates.
    """
    meta = getattr(response, "response_metadata", None) or {}
    prompt_tokens = meta.get("prompt_eval_count")
    output_tokens = meta.get("eval_count")
    eval_ns = meta.get("eval_duration") or 0
    total_ns = meta.get("total_duration") or 0
    total_tokens = (prompt_tokens or 0) + (output_tokens or 0)
    LOGGER.info(
        "LLM_USAGE kind=%s prompt=%s output=%s total=%s eval_ms=%.0f total_ms=%.0f",
        kind or "-",
        prompt_tokens,
        output_tokens,
        total_tokens,
        eval_ns / 1e6,
        total_ns / 1e6,
    )


def _invoke_json(kind: str, system_prompt: str, human_prompt: str, fallback: dict) -> dict:
    try:
        content = _invoke_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ],
            json_mode=True,
            kind=kind,
        )
        _log_llm_exchange(kind, system_prompt, human_prompt, content)
        return coerce_json(content)
    except Exception as exc:
        LOGGER.warning("JSON LLM call failed in %s: %s | content=%r", kind, exc, locals().get("content", "")[:500])
        return fallback


def _invoke_text(kind: str, system_prompt: str, human_prompt: str, fallback: str) -> str:
    try:
        content = _invoke_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ],
            kind=kind,
        ).strip()
        _log_llm_exchange(kind, system_prompt, human_prompt, content)
        return content
    except Exception as exc:
        LOGGER.warning("Text LLM call failed in %s: %s", kind, exc)
        return fallback


def _invoke_text_capturing(
    kind: str, system_prompt: str, human_prompt: str, fallback: str
) -> tuple[str, dict[str, str]]:
    """Same as _invoke_text but also returns the {system, human, response} exchange."""
    try:
        content = _invoke_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ],
            kind=kind,
        ).strip()
        _log_llm_exchange(kind, system_prompt, human_prompt, content)
        return content, {"system": system_prompt, "human": human_prompt, "response": content}
    except Exception as exc:
        LOGGER.warning("Text LLM call failed in %s: %s", kind, exc)
        return fallback, {}


def _invoke_json_capturing(
    kind: str, system_prompt: str, human_prompt: str, fallback: dict
) -> tuple[dict, dict[str, str]]:
    """Same as _invoke_json but also returns the {system, human, response} exchange."""
    try:
        content = _invoke_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ],
            json_mode=True,
            kind=kind,
        )
        _log_llm_exchange(kind, system_prompt, human_prompt, content)
        return coerce_json(content), {"system": system_prompt, "human": human_prompt, "response": content}
    except Exception as exc:
        LOGGER.warning("JSON LLM call failed in %s: %s | content=%r", kind, exc, locals().get("content", "")[:500])
        return fallback, {}


def generate_case_title(description: str) -> str:
    """Produce a short Russian title for a case based on its deal description.

    This is a standalone helper (not a graph node): server.py fires it as an
    asyncio task after the graph completes, when the deal_type changed and a
    fresh title is needed. Sync because ChatOllama.invoke is sync; callers
    should wrap with asyncio.to_thread.
    """
    description = (description or "").strip()
    if not description:
        return ""
    title = _invoke_text(
        "generate_case_title",
        LEGAL_ASSISTANT_SYSTEM_PROMPT,
        GENERATE_CASE_TITLE_PROMPT.format(description=description[:2000]),
        fallback="",
    ).strip()
    title = title.strip(' "«»\n\r\t')
    if title.endswith("."):
        title = title.rstrip(".").rstrip()
    if len(title) > 80:
        title = title[:80].rstrip()
    return title


def _last_user_message(state: ContractAgentState) -> str:
    messages = state.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, dict):
            msg_type = str(message.get("type") or message.get("role") or "").lower()
            content = message.get("content") or ""
        else:
            msg_type = str(getattr(message, "type", "") or getattr(message, "role", "") or "").lower()
            content = getattr(message, "content", "") or ""
        if msg_type in {"human", "user"} and isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def classify_followup_intent(state: ContractAgentState, last_user_message: str) -> str:
    """Classify the user's new message into one of {edit, regenerate, followup}.

    `edit` means a targeted change to the existing contract (price, single
    clause, etc.) — handled by the edit_contract node, no full regeneration.
    `regenerate` means rebuild the contract from scratch (deal type changed,
    user explicitly asked for a fresh draft). `followup` means a Q&A turn —
    no contract change.
    """
    fallback = {"intent": "followup", "reason": "fallback to safe option"}
    result = _invoke_json(
        "classify_followup_intent",
        CLASSIFY_FOLLOWUP_INTENT_PROMPT,
        (
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{state.get('deal_description', '')}\n\n"
            f"Новое сообщение пользователя:\n{last_user_message}"
        ),
        fallback=fallback,
    )
    intent = str(result.get("intent", "followup")).strip().lower()
    if intent in {"edit", "regenerate", "followup"}:
        return intent
    return "followup"


def route_user_message(state: ContractAgentState) -> dict:
    """First node of the graph. Branches between full regeneration and follow-up Q&A.

    Follow-up only makes sense if a previous run actually produced something
    the user can ask about — concretely, recommendations text or a contract
    HTML. Without an artifact we always go through the full pipeline,
    regardless of the `result_saved` flag (which can be set by terminal
    branches like `inform_unsupported_deal` even though no real output was
    produced).
    """
    has_artifact = bool((state.get("contract_html") or "").strip() or (state.get("recommendations") or "").strip())
    if not state.get("result_saved") or not has_artifact:
        return {"intent": "regenerate", "processing_stage": STAGE_CLASSIFY_CONTRACT}

    last_user = _last_user_message(state)
    if not last_user:
        return {"intent": "followup", "processing_stage": STAGE_DEFAULT}

    intent = classify_followup_intent(state, last_user)
    LOGGER.info("route_user_message intent=%s last_user=%r", intent, last_user[:120])

    if intent == "edit":
        # Fall back to regenerate if no prior contract exists to edit.
        if not (state.get("contract_html") or "").strip():
            LOGGER.info("route_user_message edit→regenerate (no prior contract)")
            intent = "regenerate"
        else:
            return {
                "intent": "edit",
                "validation_errors": [],
                "iteration_count": 0,
                "contract_valid": None,
                "result_saved": False,
                "edit_summary": None,
                "edit_changed_sections": None,
                "processing_stage": STAGE_EDIT_CONTRACT,
            }

    if intent == "regenerate":
        # Reset fields that should be regenerated fresh on this iteration.
        return {
            "intent": "regenerate",
            "contract_html": None,
            "contract_md": None,
            "recommendations": None,
            "validation_errors": [],
            "iteration_count": 0,
            "contract_valid": None,
            "result_saved": False,
            "result_docx_path": None,
            "edit_summary": None,
            "edit_changed_sections": None,
            "processing_stage": STAGE_CLASSIFY_CONTRACT,
        }

    return {"intent": "followup", "processing_stage": STAGE_DEFAULT}


# Re-exported from agent/gate.py so tests can import without pulling LangChain/Ollama/html2docx.
from agent.gate import FREE_EDIT_REFUSAL_TEXT, gate_free_plan  # noqa: E402, F401


def _followup_state_context(state: ContractAgentState, last_user: str) -> str:
    """Common context block used by the follow-up triage and as a fallback
    answer path. Centralizes the section layout so both callers see the same
    view of the case."""
    contract_preview = (state.get("contract_md") or "").strip() or "Договор пока не сформирован."
    recommendations = (state.get("recommendations") or "").strip() or "Рекомендации пока не сформированы."
    return (
        f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
        f"{SECTION_LABEL_DEAL_TYPE} {state.get('deal_type') or '-'}\n\n"
        f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{state.get('deal_description', '')}\n\n"
        f"{SECTION_LABEL_GENERAL_NORMS}\n{_norms_to_context(state.get('general_norms'))}\n\n"
        f"{SECTION_LABEL_SPECIFIC_NORMS}\n{_norms_to_context(state.get('retrieved_norms'))}\n\n"
        f"Текущие рекомендации (ранее отправлены пользователю):\n{recommendations}\n\n"
        f"Текущий проект договора (текстовое превью):\n{contract_preview}\n\n"
        f"Новое сообщение пользователя:\n{last_user}"
    )


def _clear_subagent_state() -> dict:
    """State patch that resets sub-agent fields so the next follow-up turn
    starts clean. Used both on `answer` mode and after the sub-agent loop
    terminates (found / give_up / max_attempts)."""
    return {
        "subagent_active": False,
        "subagent_brief": None,
        "subagent_user_question": None,
        "subagent_query": None,
        "subagent_query_history": None,
        "subagent_attempts": 0,
    }


def followup_response(state: ContractAgentState) -> dict:
    """Triage a follow-up turn: either answer from existing state or delegate
    to the iterative `followup_subagent_search` loop.

    Returns a JSON-mode decision from the LLM. On `answer` we emit the
    message and END. On `delegate` we stash the sub-agent brief + initial
    query in state; the conditional edge then routes to the search loop.
    """
    last_user = _last_user_message(state) or "(пустое сообщение)"
    context_block = _followup_state_context(state, last_user)

    decision = _invoke_json(
        "followup_triage",
        FOLLOWUP_TRIAGE_PROMPT,
        context_block,
        fallback={"mode": "answer", "answer": ""},
    )
    mode = str(decision.get("mode") or "").strip().lower()
    max_attempts = max(1, int(settings.followup_subagent_max_attempts))

    if mode == "delegate":
        brief = str(decision.get("subagent_brief") or "").strip()
        initial_query = str(decision.get("initial_query") or "").strip()
        if brief and initial_query:
            LOGGER.info(
                "followup_response delegating to sub-agent brief_chars=%s initial_query=%r",
                len(brief),
                initial_query,
            )
            return {
                "intent": "followup",
                "processing_stage": STAGE_FOLLOWUP_SUBAGENT,
                "subagent_active": True,
                "subagent_brief": brief,
                "subagent_user_question": last_user,
                "subagent_query": initial_query,
                "subagent_query_history": [],
                "subagent_attempts": 0,
                "subagent_max_attempts": max_attempts,
            }
        LOGGER.warning("followup_response delegate mode missing brief/query; falling back to answer")

    answer = (decision.get("answer") or "").strip() if isinstance(decision.get("answer"), str) else ""
    if not answer:
        # Re-invoke with the legacy text prompt as a fallback for empty/malformed JSON.
        answer = _invoke_text(
            "followup_response_fallback",
            FOLLOWUP_RESPONSE_PROMPT,
            context_block,
            fallback=(
                "Похоже, у меня сейчас нет дополнительной информации по вашему вопросу. "
                "Уточните, пожалуйста, что именно вас интересует."
            ),
        )

    update = {
        **_state_messages_update(answer),
        "intent": "followup",
        "processing_stage": STAGE_DEFAULT,
    }
    update.update(_clear_subagent_state())
    return update


def followup_subagent_search(state: ContractAgentState) -> dict:
    """One iteration of the follow-up sub-agent search loop.

    Retrieves SECONDARY chunks for the current query, asks the LLM to either
    answer with citations, reformulate the query, or give up. Loops back to
    itself (via the graph's conditional edge) until an answer is found, the
    LLM gives up, or `subagent_max_attempts` is reached.

    State invariants:
      - `subagent_brief` and `subagent_user_question` are set by the parent
        and never change inside the loop.
      - `subagent_query` is replaced on each `continue` iteration.
      - `subagent_query_history` accumulates queries already tried so the LLM
        doesn't repeat them.
      - chunks retrieved during one iteration do NOT carry over.
    """
    brief = (state.get("subagent_brief") or "").strip()
    user_question = (state.get("subagent_user_question") or "").strip()
    query = (state.get("subagent_query") or "").strip()
    attempts = int(state.get("subagent_attempts", 0))
    max_attempts = max(1, int(state.get("subagent_max_attempts", settings.followup_subagent_max_attempts)))
    history = list(state.get("subagent_query_history") or [])

    if not brief or not query:
        LOGGER.warning("followup_subagent_search: missing brief or query, ending loop")
        return {
            **_state_messages_update(FOLLOWUP_SUBAGENT_GIVE_UP_MESSAGE),
            "intent": "followup",
            "processing_stage": STAGE_DEFAULT,
            **_clear_subagent_state(),
        }

    # Per-iteration fresh SECONDARY retrieval — chunks do not persist across attempts.
    chunks = retrieve_secondary(query, top_k=settings.followup_subagent_top_k)
    LOGGER.info(
        "followup_subagent_search attempt=%s/%s query=%r chunks=%s",
        attempts + 1,
        max_attempts,
        query,
        len(chunks),
    )

    history_block = (
        "\n".join(f"- {q}" for q in history) if history else "(пока не было предыдущих запросов)"
    )

    human_prompt = (
        f"Бриф от родительского агента:\n{brief}\n\n"
        f"Вопрос пользователя:\n{user_question or '(не указан)'}\n\n"
        f"Текущий поисковый запрос:\n{query}\n\n"
        f"Уже использованные формулировки запроса:\n{history_block}\n\n"
        f"Найденные нормы (выборка по текущему запросу):\n{_norms_to_context(chunks, limit=len(chunks))}"
    )

    decision = _invoke_json(
        f"followup_subagent_attempt_{attempts + 1}",
        FOLLOWUP_SUBAGENT_PROMPT,
        human_prompt,
        fallback={"status": "give_up", "reason": "LLM-вызов не удался"},
    )
    status = str(decision.get("status") or "").strip().lower()
    new_history = [*history, query]

    if status == "found":
        answer_text = (decision.get("answer") or "").strip() if isinstance(decision.get("answer"), str) else ""
        if not answer_text:
            answer_text = FOLLOWUP_SUBAGENT_GIVE_UP_MESSAGE
        LOGGER.info("followup_subagent_search found answer after %s attempt(s)", attempts + 1)
        return {
            **_state_messages_update(answer_text),
            "intent": "followup",
            "processing_stage": STAGE_DEFAULT,
            **_clear_subagent_state(),
        }

    next_attempts = attempts + 1
    if status == "continue" and next_attempts < max_attempts:
        next_query = (decision.get("next_query") or "").strip() if isinstance(decision.get("next_query"), str) else ""
        # Treat blank/duplicate next_query as give_up — don't burn attempts on duplicates.
        if not next_query or next_query == query or next_query in new_history:
            LOGGER.info("followup_subagent_search continue with degenerate next_query=%r — giving up", next_query)
        else:
            return {
                "intent": "followup",
                "processing_stage": STAGE_FOLLOWUP_SUBAGENT,
                "subagent_active": True,
                "subagent_query": next_query,
                "subagent_query_history": new_history,
                "subagent_attempts": next_attempts,
            }

    # give_up, max attempts reached, or degenerate continue.
    if next_attempts >= max_attempts:
        LOGGER.info("followup_subagent_search exhausted %s attempts without an answer", next_attempts)
    else:
        LOGGER.info("followup_subagent_search gave up: %s", decision.get("reason"))
    return {
        **_state_messages_update(FOLLOWUP_SUBAGENT_GIVE_UP_MESSAGE),
        "intent": "followup",
        "processing_stage": STAGE_DEFAULT,
        **_clear_subagent_state(),
    }


def classify_yesno(question: str, answer: str) -> str:
    """Use the LLM to classify a free-form reply as 'yes', 'no' or 'unclear'.

    On LLM failure we fall back to 'unclear' so the agent re-asks instead of
    silently picking a side-effecting branch (e.g. generating a contract).
    """
    fallback = {"decision": "unclear", "reason": "LLM call failed; defaulting to unclear"}
    result = _invoke_json(
        "classify_yesno",
        CLASSIFY_YESNO_PROMPT,
        f"Вопрос агента:\n{question}\n\nОтвет пользователя:\n{answer}",
        fallback=fallback,
    )
    decision = str(result.get("decision", "unclear")).strip().lower()
    return decision if decision in {"yes", "no"} else "unclear"


def _norms_to_context(
    norms: list[dict] | None,
    limit: int = 12,
    text_limit: int = 25000,
    max_total_chars: int = 200000,
) -> str:
    if not norms:
        return "Нормы не найдены."
    parts: list[str] = []
    used_chars = 0
    for chunk in norms[:limit]:
        text = (chunk.get("text") or "").strip()
        clipped_text = text[:text_limit] if len(text) > text_limit else text
        rendered = (
            f"Источник: {chunk.get('source', '-')}; "
            f"Статья: {chunk.get('article', '-')}; "
            f"Глава: {chunk.get('chapter', '-')};\n{clipped_text}"
        )
        separator_len = 7 if parts else 0
        if used_chars + separator_len + len(rendered) > max_total_chars:
            remaining = max_total_chars - used_chars - separator_len
            if remaining <= 120:
                break
            rendered = rendered[:remaining].rstrip()
        parts.append(rendered)
        used_chars += separator_len + len(rendered)
        if used_chars >= max_total_chars:
            break
    return "\n\n---\n\n".join(parts)


def _source_filter_for_deal(deal_type: str | None, deal_description: str | None = None) -> str | None:
    """Source filter disabled; signature kept so callers don't need to change if it's restored."""
    return None


def _state_messages_update(text: str) -> dict:
    return {"messages": [AIMessage(content=text)]}


def _build_retrieval_query(state: ContractAgentState) -> str:
    fallback = (
        f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
        f"{SECTION_LABEL_DEAL_TYPE} {state.get('deal_type') or 'не определен'}\n"
        f"{SECTION_LABEL_DEAL_DESCRIPTION} {state.get('deal_description', '')}"
    )
    return _invoke_text(
        "build_retrieval_query",
        LEGAL_ASSISTANT_SYSTEM_PROMPT,
        (
            f"{BUILD_RETRIEVAL_QUERY_PROMPT}\n\n"
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_TYPE} {state.get('deal_type') or '-'}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{state.get('deal_description', '')}"
        ),
        fallback=fallback,
    )


def _conversation_context(state: ContractAgentState) -> str:
    messages = state.get("messages") or []
    if not messages:
        return "Conversation history is empty."

    rendered: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            msg_type = str(message.get("type") or message.get("role") or "").lower()
            content = message.get("content") or ""
        else:
            msg_type = str(getattr(message, "type", "") or getattr(message, "role", "") or "").lower()
            content = getattr(message, "content", "") or ""

        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if not content:
            continue

        if msg_type in {"human", "user"}:
            role = "User"
        elif msg_type in {"ai", "assistant"}:
            role = "Assistant"
        elif msg_type == "system":
            role = "System"
        else:
            role = "Message"

        rendered.append(f"{role}: {content}")

    return "\n".join(rendered) if rendered else "Conversation history is empty."


def load_general_norms(state: ContractAgentState) -> dict:
    # No-op kept to preserve graph shape and saved-run compatibility.
    return {"processing_stage": STAGE_CLASSIFY_CONTRACT}


_PAREN_TAIL_RE = re.compile(r"\s*\([^)]*\)\s*")


def _resolve_deal_type(raw: object) -> str | None:
    """Map an LLM-returned deal_type to a canonical entry in the catalog.

    The classifier prompt asks for the exact catalog name, but in practice
    models drift: different case, trailing whitespace, dropped parenthetical
    suffixes ("договор безвозмездного пользования" vs catalog's
    "договор безвозмездного пользования (ссуды)"). Strict `in` matching
    silently fails on all of these and routes the case to
    `inform_unsupported_deal`, contradicting the model's intent.

    Lookup layers, in order:
      1. exact match (fast path, returns canonical as-is)
      2. case-insensitive equality
      3. parenthetical-tolerant case-insensitive equality
    Returns the canonical catalog name on match, None on no match.
    """
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None

    supported = get_supported_deal_types()

    if cleaned in supported:
        return cleaned

    cleaned_lower = cleaned.lower()
    for canonical in supported:
        if canonical.lower() == cleaned_lower:
            return canonical

    cleaned_stripped = _PAREN_TAIL_RE.sub(" ", cleaned_lower).strip()
    if not cleaned_stripped:
        return None
    for canonical in supported:
        canonical_stripped = _PAREN_TAIL_RE.sub(" ", canonical.lower()).strip()
        if canonical_stripped == cleaned_stripped:
            LOGGER.info(
                "Resolved LLM deal_type %r to canonical %r via parenthetical-tolerant match",
                cleaned,
                canonical,
            )
            return canonical

    LOGGER.info("Deal type %r not in supported catalog — treating as unsupported", cleaned)
    return None


_RAG_ASSIST_CHUNK_PREVIEW_CHARS = 350


def _summarize_chunks_for_classifier(chunks: list[dict]) -> str:
    """Compact representation of bi-encoder hits for the RAG-assist sub-agent
    prompt. Used purely as transient context — never persisted to state."""
    if not chunks:
        return "(ничего не нашлось)"
    parts: list[str] = []
    for chunk in chunks:
        source = (chunk.get("source") or "-").strip()
        article = (chunk.get("article") or chunk.get("article_number") or "-")
        text = (chunk.get("text") or "").strip().replace("\n", " ")
        if len(text) > _RAG_ASSIST_CHUNK_PREVIEW_CHARS:
            text = text[:_RAG_ASSIST_CHUNK_PREVIEW_CHARS].rstrip() + "..."
        parts.append(f"- [{source}, {article}] {text}")
    return "\n".join(parts)


def _classify_with_rag_assist(
    description: str,
    fallback_deal_type: str | None,
) -> tuple[str | None, str]:
    """Sub-agent loop: classifier issues bounded bi-encoder queries to PRIMAL
    to ground its judgement, then returns (deal_type, confidence).

    No side effects: retrieved chunks live only in this function's local
    variables. Nothing leaks into state, messages, or `retrieved_norms` —
    consistent with the user's spec that RAG-assist must be invisible to the
    rest of the pipeline.
    """
    max_queries = max(1, int(settings.classify_rag_assist_max_queries))
    top_k = max(1, int(settings.classify_rag_assist_top_k_per_query))
    LOGGER.info(
        "classify_rag_assist start max_queries=%s top_k=%s fallback_type=%r",
        max_queries,
        top_k,
        fallback_deal_type,
    )

    queries_history: list[dict[str, str]] = []  # [{"query": ..., "summary": ...}]

    for attempt in range(1, max_queries + 1):
        history_block = (
            "\n\n".join(
                f"### Запрос #{i}: {entry['query']}\n{entry['summary']}"
                for i, entry in enumerate(queries_history, 1)
            )
            or "(пока запросов не было)"
        )
        human_prompt = (
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{description}\n\n"
            f"Уже выполненные запросы и краткая выдача:\n{history_block}\n\n"
            f"Текущая попытка: {attempt} из {max_queries}."
        )
        result = _invoke_json(
            f"classify_rag_assist_{attempt}",
            build_classify_rag_assist_prompt(),
            human_prompt,
            fallback={"finished": True, "deal_type": fallback_deal_type, "confidence": "low"},
        )
        if not isinstance(result, dict):
            break

        if bool(result.get("finished")):
            deal_type = _resolve_deal_type(result.get("deal_type"))
            confidence = str(result.get("confidence", "medium")).lower()
            if confidence not in {"low", "medium", "high"}:
                confidence = "medium"
            LOGGER.info(
                "classify_rag_assist finished attempts=%s deal_type=%r confidence=%s reason=%r",
                attempt,
                deal_type,
                confidence,
                result.get("reason"),
            )
            return deal_type, confidence

        next_query = (result.get("next_query") or "").strip() if isinstance(result.get("next_query"), str) else ""
        if not next_query or any(entry["query"] == next_query for entry in queries_history):
            LOGGER.info("classify_rag_assist degenerate next_query=%r — stopping", next_query)
            break

        # Bi-encoder only — no rerank, no expansion, no side effects.
        chunks = search_primal_bi_encoder(query=next_query, source_filter=None, top_k=top_k)
        summary = _summarize_chunks_for_classifier(chunks)
        queries_history.append({"query": next_query, "summary": summary})
        LOGGER.info(
            "classify_rag_assist attempt=%s query=%r chunks=%s reason=%r",
            attempt,
            next_query,
            len(chunks),
            result.get("reason"),
        )

    LOGGER.info("classify_rag_assist exhausted without finalising — keeping fallback")
    return _resolve_deal_type(fallback_deal_type), "low"


def classify_deal(state: ContractAgentState) -> dict:
    description = state.get("deal_description", "").strip()
    current_attempts = int(state.get("classification_clarification_attempts", 0))
    max_attempts = int(state.get("max_classification_clarifications", settings.max_classification_clarifications))
    fallback = {
        "deal_type": None,
        "confidence": "low",
        "clarification_needed": False,
        "clarification_question": None,
    }
    if not description:
        return {
            "deal_type": None,
            "deal_type_confidence": "low",
            "deal_classify_confidence": "low",
            "clarification_needed": False,
            "clarification_question": None,
            "clarification_stage": None,
            "classification_clarification_attempts": current_attempts,
            "max_classification_clarifications": max_attempts,
            "processing_stage": STAGE_DEFAULT,
        }

    result = _invoke_json(
        "classify_deal",
        build_classify_deal_prompt(),
        (
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{description}\n\n"
            f"Current classification clarification attempt: {current_attempts} of {max_attempts}.\n"
            "Если уверенность low или medium и нужен вопрос для уточнения, задай один уточняющий вопрос."
        ),
        fallback=fallback,
    )
    deal_type = _resolve_deal_type(result.get("deal_type"))

    confidence = str(result.get("confidence", "low")).lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

    # RAG-assist on low/medium confidence; result accepted only on high or when primary returned None.
    if settings.classify_rag_assist_enabled and confidence != "high":
        refined_type, refined_conf = _classify_with_rag_assist(description, deal_type)
        if refined_conf == "high" or (refined_type is not None and deal_type is None):
            LOGGER.info(
                "classify_deal: RAG-assist refined type=%r → %r confidence=%s → %s",
                deal_type,
                refined_type,
                confidence,
                refined_conf,
            )
            deal_type = refined_type
            confidence = refined_conf
            # Подавим clarification: модель уже ответила за пользователя.
            result = {
                **result,
                "deal_type": refined_type,
                "confidence": refined_conf,
                "clarification_needed": False,
                "clarification_question": None,
            }

    clarification_question = result.get("clarification_question")
    wants_clarification = bool(result.get("clarification_needed", False))
    should_clarify = (
        confidence in RETRYABLE_CLASSIFICATION_CONFIDENCES
        and current_attempts < max_attempts
        and bool(clarification_question)
        and wants_clarification
    )

    update = {
        "deal_type": deal_type,
        "deal_type_confidence": confidence,
        "deal_classify_confidence": confidence,
        "max_classification_clarifications": max_attempts,
    }
    if should_clarify:
        update.update(
            {
                "clarification_needed": True,
                "clarification_question": str(clarification_question),
                "clarification_stage": "classification",
                "classification_clarification_attempts": current_attempts + 1,
                "processing_stage": STAGE_DEFAULT,
            }
        )
    else:
        unsupported = deal_type is None
        update.update(
            {
                "clarification_needed": False
                if state.get("clarification_stage") == "classification"
                else state.get("clarification_needed", False),
                "clarification_question": None
                if state.get("clarification_stage") == "classification"
                else state.get("clarification_question"),
                "clarification_stage": None
                if state.get("clarification_stage") == "classification"
                else state.get("clarification_stage"),
                "classification_clarification_attempts": current_attempts,
                "processing_stage": STAGE_DEFAULT if unsupported else STAGE_RETRIEVE_NORMS,
            }
        )
    return update


def inform_unsupported_deal(state: ContractAgentState) -> dict:
    """Friendly message when the deal does not match any supported type."""
    fallback = (
        "К сожалению, описанный тип сделки пока не поддерживается PactumAI.\n\n"
        "Сейчас система умеет работать со следующими типами договоров:\n"
        + "\n".join(f"- {t}" for t in get_supported_deal_types())
        + "\n\nПопробуйте описать другой договор из списка."
    )
    message = _invoke_text(
        "inform_unsupported_deal",
        build_inform_unsupported_deal_prompt(),
        (
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{state.get('deal_description', '')}"
        ),
        fallback=fallback,
    )
    # result_saved intentionally left False so the next user message reruns the full pipeline.
    case_id = state.get("case_id")
    if case_id:
        try:
            update_case_status(case_id, CaseStatus.ERROR.value)
        except Exception as exc:
            LOGGER.warning("Failed to update case status from inform_unsupported_deal: %s", exc)

    return {
        **_state_messages_update(message),
        "processing_stage": STAGE_DEFAULT,
    }


def check_general_norms(state: ContractAgentState) -> dict:
    description = state.get("deal_description", "")
    retrieval_query = _build_retrieval_query(state).strip()
    general_norms = retrieve_general(
        query=retrieval_query,
        top_k=settings.retrieval_general_top_k,
        candidate_top_k=settings.retrieval_candidate_top_k,
    )
    norms_context = _norms_to_context(general_norms, limit=settings.retrieval_general_top_k)
    fallback = {"passed": True, "issues": [], "explanation": None}
    result = _invoke_json(
        "check_general_norms",
        CHECK_GENERAL_NORMS_PROMPT,
        (
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{description}\n\n"
            f"{SECTION_LABEL_GENERAL_NORMS}\n{norms_context}"
        ),
        fallback=fallback,
    )
    issues = result.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    return {
        "retrieval_query": retrieval_query,
        "general_norms": general_norms,
        "general_check_passed": bool(result.get("passed", True)),
        "general_check_issues": [str(item) for item in issues],
        "general_check_explanation": result.get("explanation"),
        "processing_stage": STAGE_RETRIEVE_NORMS if bool(result.get("passed", True)) else STAGE_ANALYZE_NORMS,
    }


def inform_user(state: ContractAgentState) -> dict:
    issues = state.get("general_check_issues") or []
    explanation = state.get("general_check_explanation")
    fallback = "Сделка не соответствует общим нормам гражданского права и требует уточнения."
    message = _invoke_text(
        "inform_user",
        LEGAL_ASSISTANT_SYSTEM_PROMPT,
        (
            f"{INFORM_USER_PROMPT}\n\n"
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{state.get('deal_description', '')}\n\n"
            f"Проблемы:\n- " + "\n- ".join(issues or ["Нарушения не конкретизированы"]) + "\n\n"
            f"Пояснение:\n{explanation or '-'}"
        ),
        fallback=fallback,
    )

    case_id = state.get("case_id")
    if case_id:
        try:
            update_case_status(case_id, CaseStatus.ERROR.value)
        except Exception as exc:
            LOGGER.warning("Failed to update case status from inform_user: %s", exc)

    return {**_state_messages_update(message), "processing_stage": STAGE_DEFAULT}


def _format_chunk_for_filter(chunk: dict, text_limit: int) -> str:
    chunk_id = chunk.get("chunk_id") or "(no-id)"
    source = chunk.get("source") or "-"
    article = chunk.get("article") or "-"
    text = (chunk.get("text") or "").strip()
    if len(text) > text_limit:
        text = text[:text_limit].rstrip() + "..."
    return f"chunk_id: {chunk_id}\nИсточник: {source}; Статья: {article}\n{text}"


def _filter_one_batch(
    batch_chunks: list[dict],
    deal_type: str | None,
    deal_description: str,
    kind_suffix: str = "",
) -> set[str]:
    """Run one LLM filter call over a batch of chunks. Returns the SET of
    chunk_id strings the model voted to keep.

    Conservative on failure: any flaky response (non-list, empty list,
    hallucinated ids) falls back to "keep the entire batch". This keeps the
    overall pipeline's recall stable — the filter can only HELP precision,
    never destroy recall.
    """
    all_ids = {str(c["chunk_id"]) for c in batch_chunks if c.get("chunk_id")}
    if not all_ids:
        return all_ids

    text_limit = max(100, int(settings.retrieval_filter_text_preview_chars))
    chunks_block = "\n\n---\n\n".join(_format_chunk_for_filter(c, text_limit) for c in batch_chunks)
    fallback = {"relevant_chunk_ids": list(all_ids)}

    result = _invoke_json(
        f"filter_retrieved_norms{kind_suffix}",
        FILTER_RETRIEVED_NORMS_PROMPT,
        (
            f"{SECTION_LABEL_DEAL_TYPE} {deal_type or '-'}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{deal_description}\n\n"
            f"Найденные выдержки:\n{chunks_block}"
        ),
        fallback=fallback,
    )
    keep_ids_raw = result.get("relevant_chunk_ids") if isinstance(result, dict) else None
    if not isinstance(keep_ids_raw, list):
        LOGGER.warning("Filter LLM returned non-list relevant_chunk_ids; keeping batch as-is")
        return all_ids

    keep_ids = {str(item) for item in keep_ids_raw if isinstance(item, (str, int, float)) and str(item).strip()}
    if not keep_ids:
        LOGGER.warning("Filter LLM returned empty relevant_chunk_ids; keeping batch as-is")
        return all_ids

    valid = keep_ids & all_ids
    if not valid:
        LOGGER.warning(
            "Filter LLM returned ids that don't match this batch (sent=%s, got=%s); keeping batch",
            len(all_ids),
            len(keep_ids),
        )
        return all_ids
    return valid


def _filter_relevant_chunks(
    chunks: list[dict],
    deal_type: str | None,
    deal_description: str,
) -> list[dict]:
    """LLM filter pass over retrieve_specific output.

    Asks the model which `chunk_id`s actually relate to the deal and keeps
    only those. Conservative by design — the prompt instructs the model to
    drop only clearly-irrelevant chunks. When the chunk count exceeds
    `settings.retrieval_filter_batch_size` (and that value is > 0), the
    filter is split into batches and the union of kept ids is taken; this
    avoids "lost in the middle" degradation on large top_k. Multiple
    fail-safes guarantee we never wipe the entire context on a flaky LLM
    response (see `_filter_one_batch`).
    """
    if not chunks or not settings.retrieval_filter_enabled:
        return chunks
    # No stable chunk_id means we can't map model output back — skip rather than guess.
    chunks_with_id = [c for c in chunks if c.get("chunk_id")]
    if not chunks_with_id:
        LOGGER.info("Filter skipped: no chunk_id on incoming chunks (legacy index?)")
        return chunks

    batch_size = int(settings.retrieval_filter_batch_size or 0)
    if batch_size <= 0 or len(chunks_with_id) <= batch_size:
        kept_ids = _filter_one_batch(chunks_with_id, deal_type, deal_description)
    else:
        kept_ids = set()
        n_batches = (len(chunks_with_id) + batch_size - 1) // batch_size
        LOGGER.info(
            "Filter batched: chunks=%s batch_size=%s batches=%s",
            len(chunks_with_id),
            batch_size,
            n_batches,
        )
        for batch_idx in range(n_batches):
            batch = chunks_with_id[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_kept = _filter_one_batch(
                batch, deal_type, deal_description, kind_suffix=f"_batch_{batch_idx + 1}"
            )
            LOGGER.info(
                "Filter batch %s/%s: input=%s kept=%s",
                batch_idx + 1,
                n_batches,
                len(batch),
                len(batch_kept),
            )
            kept_ids |= batch_kept

    if not kept_ids:
        # Safety net against accidental full-context wipe.
        LOGGER.warning("Filter dropped everything across batches; keeping all chunks (safety net)")
        return chunks

    filtered = [c for c in chunks if str(c.get("chunk_id") or "") in kept_ids]
    if not filtered:
        LOGGER.warning(
            "Filter kept_ids don't match any returned chunk (got=%s); keeping all",
            len(kept_ids),
        )
        return chunks

    dropped = len(chunks) - len(filtered)
    LOGGER.info(
        "Filter pass: kept=%s dropped=%s (of %s) deal_type=%r",
        len(filtered),
        dropped,
        len(chunks),
        deal_type,
    )
    return filtered


def _chunk_unique_key(chunk: dict) -> tuple[str, str, str]:
    """Stable identity for a chunk regardless of which retrieval path produced
    it. Used to dedup across iterative-loop iterations and across the
    PRIMAL/SECONDARY merge in the iterative flow."""
    chunk_id = (chunk.get("chunk_id") or "").strip()
    if chunk_id:
        return ("id", chunk_id, "")
    return ("sa", str(chunk.get("source") or ""), str(chunk.get("article") or ""))


def _summarize_approved_chunks(chunks: list[dict], text_chars: int = 180) -> str:
    """Short, planner-readable summary of approved norms. Keeps the planner
    prompt bounded — for 20 chunks at 180 chars + headers this is ~4 KB."""
    if not chunks:
        return "(пока ничего не одобрено)"
    parts = []
    for chunk in chunks:
        source = chunk.get("source", "-")
        article = chunk.get("article", "-")
        text = (chunk.get("text") or "").strip().replace("\n", " ")
        if len(text) > text_chars:
            text = text[:text_chars].rstrip() + "..."
        parts.append(f"- {source}, {article}: {text}")
    return "\n".join(parts)


def _plan_next_iterative_query(
    deal_type: str | None,
    deal_description: str,
    approved: list[dict],
    history: list[str],
    attempt_idx: int,
) -> tuple[bool, str | None]:
    """Ask the LLM to either propose the next query or signal done.

    Returns (finished, next_query). Defensive on bad output: any malformed
    response is treated as `finished=True` so we don't loop on nothing.
    """
    history_block = "\n".join(f"- {q}" for q in history) if history else "(пока не было запросов)"
    summary = _summarize_approved_chunks(approved)
    result = _invoke_json(
        f"iterative_planner_{attempt_idx}",
        ITERATIVE_RETRIEVAL_PLANNER_PROMPT,
        (
            f"{SECTION_LABEL_DEAL_TYPE} {deal_type or '-'}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{deal_description}\n\n"
            f"Уже одобренные нормы (всего {len(approved)}):\n{summary}\n\n"
            f"Уже использованные формулировки запроса:\n{history_block}"
        ),
        fallback={"finished": True, "reason": "planner LLM unavailable"},
    )
    if not isinstance(result, dict):
        return True, None
    if bool(result.get("finished")):
        return True, None
    next_query = result.get("next_query")
    if not isinstance(next_query, str) or not next_query.strip():
        return True, None
    return False, next_query.strip()


def _judge_relevance_strict(
    candidates: list[dict],
    deal_type: str | None,
    deal_description: str,
    attempt_idx: int,
) -> set[str]:
    """Strict-judge LLM call: returns the set of chunk_ids the model
    explicitly approved as definitely-relevant.

    Unlike `_filter_one_batch` (conservative — keep on doubt), this judge
    drops on doubt. Used inside the iterative loop to keep accumulated set
    clean. Falls back to empty set on failure (next loop iteration tries a
    different query); the loop's hard cap prevents infinite work.
    """
    chunks_with_id = [c for c in candidates if c.get("chunk_id")]
    if not chunks_with_id:
        return set()
    text_limit = max(100, int(settings.retrieval_filter_text_preview_chars))
    chunks_block = "\n\n---\n\n".join(_format_chunk_for_filter(c, text_limit) for c in chunks_with_id)
    result = _invoke_json(
        f"iterative_judge_{attempt_idx}",
        ITERATIVE_RELEVANCE_JUDGE_PROMPT,
        (
            f"{SECTION_LABEL_DEAL_TYPE} {deal_type or '-'}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{deal_description}\n\n"
            f"Найденные выдержки:\n{chunks_block}"
        ),
        fallback={"relevant_chunk_ids": []},  # strict: empty on failure
    )
    keep_ids_raw = result.get("relevant_chunk_ids") if isinstance(result, dict) else None
    if not isinstance(keep_ids_raw, list):
        return set()
    all_ids = {str(c["chunk_id"]) for c in chunks_with_id}
    keep_ids = {str(item) for item in keep_ids_raw if isinstance(item, (str, int, float)) and str(item).strip()}
    return keep_ids & all_ids


def _expand_references_for_chunks(chunks: list[dict]) -> list[dict]:
    """For each approved chunk, fetch its `references` from SECONDARY.
    Returns only NEW chunks not already in the input (deduped)."""
    seen = {_chunk_unique_key(c) for c in chunks}
    collected: list[dict] = []
    for chunk in chunks:
        for reference in chunk.get("references") or []:
            source, article_number = parse_reference(reference)
            if not source:
                continue
            for payload in fetch_reference_chunks_by_payload(source, article_number):
                key = _chunk_unique_key(payload)
                if key in seen:
                    continue
                seen.add(key)
                payload.setdefault("score", None)
                collected.append(payload)
    return collected


def _enrich_norms_with_vague_refs(
    chunks: list[dict],
    deal_type: str | None,
    deal_description: str,
) -> list[dict]:
    """Detect vague references in the retrieved norms and pull resolving
    chunks from SECONDARY. Returns only the new chunks (deduped vs input).

    Mirrors the `enrich_recommendations` pattern but the output is chunks
    to add to the retrieved set rather than text woven into a narrative.
    """
    if not chunks:
        return []
    text_limit = max(200, int(settings.retrieval_filter_text_preview_chars))
    norms_block = _norms_to_context(chunks, limit=len(chunks), text_limit=text_limit)
    detection = _invoke_json(
        "detect_vague_refs_in_norms",
        DETECT_VAGUE_REFERENCES_IN_NORMS_PROMPT,
        f"{SECTION_LABEL_DEAL_TYPE} {deal_type or '-'}\n\n"
        f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{deal_description}\n\n"
        f"Найденные нормы:\n{norms_block}",
        fallback={"items": []},
    )
    items = detection.get("items") if isinstance(detection, dict) else []
    if not isinstance(items, list) or not items:
        return []
    max_items = max(0, int(settings.retrieval_iterative_vague_enrichment_max_items))
    items = items[:max_items]

    seen = {_chunk_unique_key(c) for c in chunks}
    new_chunks: list[dict] = []
    for idx, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        query = (item.get("search_query") or "").strip()
        if not query:
            continue
        candidates = retrieve_secondary(query, top_k=settings.retrieval_secondary_enrichment_top_k)
        LOGGER.info(
            "Iterative vague-ref enrichment[%s]: vague=%r query=%r chunks=%s",
            idx,
            (item.get("vague_reference") or "")[:80],
            query,
            len(candidates),
        )
        for payload in candidates:
            key = _chunk_unique_key(payload)
            if key in seen:
                continue
            seen.add(key)
            payload.setdefault("score", None)
            new_chunks.append(payload)
    return new_chunks


def _retrieve_specific_iterative(state: ContractAgentState) -> list[dict]:
    """Sub-agent-driven retrieval pipeline (opt-in via
    `settings.retrieval_iterative_enabled`).

    Phases:
      1. Iterative loop — planner LLM proposes queries, bi-encoder pulls
         candidates, strict-judge LLM approves the ones definitely relevant.
         Stops at TARGET_ARTICLES approved or MAX_QUERIES attempts.
      2. Reference expansion — for each approved chunk, fetch its references
         (SECONDARY) by exact (source, article) lookup.
      3. Joint rerank — cross-encoder ranks the union and trims to
         `retrieval_specific_expanded_top_k`.
      4. Vague-reference enrichment — detect non-specific citations in the
         union's text, search SECONDARY semantically for each, add resolving
         chunks.
    No final LLM filter — the strict-judge in phase 1 already enforces
    relevance, and phase 4 only adds chunks the planner+detector deemed
    worth chasing.
    """
    deal_type = state.get("deal_type")
    deal_description = state.get("deal_description", "")
    initial_query = (state.get("retrieval_query") or _build_retrieval_query(state)).strip()
    source_filter = _source_filter_for_deal(deal_type, deal_description)

    target_articles = max(1, int(settings.retrieval_iterative_target_articles))
    max_queries = max(1, int(settings.retrieval_iterative_max_queries))
    top_k_per_query = max(1, int(settings.retrieval_iterative_top_k_per_query))

    approved: list[dict] = []
    approved_keys: set[tuple[str, str, str]] = set()
    seen_candidate_keys: set[tuple[str, str, str]] = set()
    query_history: list[str] = []
    current_query: str | None = initial_query

    LOGGER.info(
        "Iterative retrieval start: target_articles=%s max_queries=%s top_k_per_query=%s source_filter=%s",
        target_articles,
        max_queries,
        top_k_per_query,
        source_filter,
    )

    for attempt in range(1, max_queries + 1):
        if len(approved) >= target_articles:
            LOGGER.info("Iterative retrieval reached target_articles=%s, stopping", target_articles)
            break
        if not current_query:
            LOGGER.info("Iterative retrieval: no current query to run, stopping")
            break
        if current_query in query_history:
            LOGGER.info("Iterative retrieval: planner suggested duplicate query %r, stopping", current_query)
            break

        # Phase 1a: bi-encoder search.
        candidates = search_primal_bi_encoder(
            query=current_query, source_filter=source_filter, top_k=top_k_per_query
        )
        fresh = []
        for c in candidates:
            key = _chunk_unique_key(c)
            if key in seen_candidate_keys or key in approved_keys:
                continue
            seen_candidate_keys.add(key)
            fresh.append(c)
        LOGGER.info(
            "Iterative attempt %s/%s query=%r candidates=%s fresh=%s",
            attempt,
            max_queries,
            current_query,
            len(candidates),
            len(fresh),
        )
        query_history.append(current_query)

        if fresh:
            # Phase 1b: strict-judge.
            kept_ids = _judge_relevance_strict(fresh, deal_type, deal_description, attempt)
            for c in fresh:
                if str(c.get("chunk_id") or "") in kept_ids:
                    key = _chunk_unique_key(c)
                    if key in approved_keys:
                        continue
                    approved.append(c)
                    approved_keys.add(key)
            LOGGER.info(
                "Iterative attempt %s judged kept=%s (approved so far=%s/%s)",
                attempt,
                len(kept_ids),
                len(approved),
                target_articles,
            )

        if len(approved) >= target_articles:
            break

        # Phase 1c: plan next query.
        finished, next_q = _plan_next_iterative_query(
            deal_type, deal_description, approved, query_history, attempt + 1
        )
        if finished or not next_q:
            LOGGER.info("Iterative retrieval planner finished after %s attempts", attempt)
            break
        current_query = next_q

    # Cap to target — judge may approve more on a single batch when N > 1.
    if len(approved) > target_articles:
        approved = approved[:target_articles]

    LOGGER.info("Iterative phase 1 done: approved=%s queries_used=%s", len(approved), len(query_history))
    if not approved:
        return []

    # Phase 2: reference expansion.
    expanded = _expand_references_for_chunks(approved)
    LOGGER.info("Iterative phase 2 (references): added=%s", len(expanded))

    # Phase 3: joint rerank.
    union = list(approved) + expanded
    final_top_k = max(target_articles, int(settings.retrieval_specific_expanded_top_k))
    reranked = rerank_chunks(initial_query, union, final_top_k)
    LOGGER.info("Iterative phase 3 (rerank): union=%s reranked=%s", len(union), len(reranked))

    # Phase 4: vague-reference enrichment.
    enrichment = _enrich_norms_with_vague_refs(reranked, deal_type, deal_description)
    if enrichment:
        LOGGER.info("Iterative phase 4 (vague-ref enrichment): added=%s", len(enrichment))
    final_norms = reranked + enrichment

    LOGGER.info(
        "Iterative retrieval done: final=%s (approved=%s + refs=%s after rerank + enrichment=%s)",
        len(final_norms),
        len(approved),
        len(expanded),
        len(enrichment),
    )
    return final_norms


def retrieve_norms(state: ContractAgentState) -> dict:
    # Captures node-only latency for eval harness, independent of classify/check_general_norms.
    started = time.monotonic()

    deal_type = state.get("deal_type")
    description = state.get("deal_description", "")
    retrieval_query = (state.get("retrieval_query") or _build_retrieval_query(state)).strip()

    if settings.retrieval_iterative_enabled:
        norms = _retrieve_specific_iterative(state)
        LOGGER.info(
            "Iterative retriever response\nchunks=%s\nsources=%s",
            len(norms),
            [item.get("source") for item in norms[:10]],
        )
    else:
        # Classic path: semantic search → reference expansion → joint rerank → LLM filter.
        source_filter = _source_filter_for_deal(deal_type, description)
        LOGGER.info(
            "Retriever request\nretrieval_query=%s\nsource_filter=%s\ntop_k=%s",
            retrieval_query,
            source_filter,
            DEFAULT_TOP_K,
        )
        norms = retrieve_specific(
            query=retrieval_query,
            source_filter=source_filter,
            top_k=DEFAULT_TOP_K,
            candidate_top_k=settings.retrieval_candidate_top_k,
        )
        LOGGER.info(
            "Retriever response\nchunks=%s\nsources=%s",
            len(norms),
            [item.get("source") for item in norms[:10]],
        )
        norms = _filter_relevant_chunks(norms, deal_type=deal_type, deal_description=description)

    elapsed_ms = (time.monotonic() - started) * 1000.0
    LOGGER.info("retrieve_norms latency=%.1f ms chunks=%s", elapsed_ms, len(norms))
    return {
        "retrieved_norms": norms,
        "retrieve_norms_latency_ms": elapsed_ms,
        "processing_stage": STAGE_ANALYZE_NORMS,
    }


def check_data_sufficiency(state: ContractAgentState) -> dict:
    fallback = {
        "missing_fields": [],
        "clarification_question": None,
        "deal_structure": state.get("deal_structure") or {},
    }
    ask_personal_data = state.get("ask_personal_data", True)
    validation_errors = [str(e) for e in (state.get("validation_errors") or []) if str(e).strip()]
    validation_section = (
        f"\n\n{SECTION_LABEL_VALIDATION_ERRORS}\n" + "\n".join(f"- {e}" for e in validation_errors)
        if validation_errors
        else ""
    )
    result = _invoke_json(
        "check_data_sufficiency",
        build_check_data_sufficiency_prompt(ask_personal_data),
        (
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_TYPE} {state.get('deal_type') or '-'}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{state.get('deal_description', '')}\n\n"
            f"{SECTION_LABEL_NORMS}\n{_norms_to_context(state.get('retrieved_norms'))}"
            f"{validation_section}"
        ),
        fallback=fallback,
    )
    missing_fields = result.get("missing_fields") or []
    if not isinstance(missing_fields, list):
        missing_fields = [str(missing_fields)]
    missing_fields = [str(item) for item in missing_fields if str(item).strip()]
    sufficient = not missing_fields
    return {
        "clarification_needed": not sufficient,
        "clarification_question": result.get("clarification_question") if not sufficient else None,
        "clarification_stage": "data_sufficiency" if not sufficient else None,
        "missing_fields": missing_fields,
        "deal_structure": result.get("deal_structure") or state.get("deal_structure") or {},
        "processing_stage": STAGE_DEFAULT if not sufficient else STAGE_GENERATE_CONTRACT,
    }


def ask_clarification(state: ContractAgentState) -> dict:
    question = state.get("clarification_question") or "Уточните, пожалуйста, недостающие существенные условия сделки."
    return {**_state_messages_update(question), "processing_stage": STAGE_DEFAULT}


def check_written_form(state: ContractAgentState) -> dict:
    fallback = {"requires_written_form": True, "reason": "письменная форма предполагается по умолчанию"}
    result = _invoke_json(
        "check_written_form",
        CHECK_WRITTEN_FORM_PROMPT,
        (
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_TYPE} {state.get('deal_type') or '-'}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{state.get('deal_description', '')}\n\n"
            f"{SECTION_LABEL_NORMS}\n{_norms_to_context(state.get('retrieved_norms'))}"
        ),
        fallback=fallback,
    )
    requires_written_form = bool(result.get("requires_written_form", True))
    update = {
        "requires_written_form": requires_written_form,
        "written_form_reason": result.get("reason"),
        "processing_stage": STAGE_GENERATE_CONTRACT if requires_written_form else STAGE_DEFAULT,
    }
    # always_ask is the only policy that surfaces a clarification when the law doesn't require it.
    policy = state.get("contract_generation_policy") or "always_ask"
    if not requires_written_form and policy == "always_ask":
        reason = result.get("reason") or "по найденным нормам письменная форма не является обязательной"
        update.update(
            {
                "clarification_needed": True,
                "clarification_stage": "optional_contract_generation",
                "clarification_question": (
                    f"Письменная форма для этой сделки может не требоваться: {reason}. "
                    "Сгенерировать договор в письменной форме?"
                ),
            }
        )
    return update


def generate_recommendations(state: ContractAgentState) -> dict:
    fallback = (
        "## Рекомендации\n\n"
        f"- Тип сделки: {state.get('deal_type') or 'не определен'}\n"
        f"- Письменная форма: {'требуется' if state.get('requires_written_form') else 'может не требоваться'}\n"
    )
    recommendations, exchange = _invoke_text_capturing(
        "generate_recommendations",
        GENERATE_RECOMMENDATIONS_PROMPT,
        (
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_TYPE} {state.get('deal_type') or '-'}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{state.get('deal_description', '')}\n\n"
            f"{SECTION_LABEL_GENERAL_NORMS}\n{_norms_to_context(state.get('general_norms'))}\n\n"
            f"{SECTION_LABEL_SPECIFIC_NORMS}\n{_norms_to_context(state.get('retrieved_norms'))}"
        ),
        fallback=fallback,
    )
    return {
        "recommendations": recommendations,
        "final_recommendations_exchange": exchange or None,
        "processing_stage": STAGE_ENRICH_RECOMMENDATIONS,
    }


def enrich_recommendations(state: ContractAgentState) -> dict:
    """Detect vague external-law references in recommendations and enrich them.

    For each detected vague reference we spawn an isolated sub-agent call:
    only the small set of secondary-collection chunks relevant to that
    specific reference is passed in, so the main agent's context is not
    polluted by N×K chunks. Each sub-agent returns a short clarification
    with a concrete [source, article] citation. A final integration call
    weaves these clarifications back into the original recommendations as
    coherent prose.
    """
    recommendations = (state.get("recommendations") or "").strip()
    if not recommendations:
        return {"processing_stage": STAGE_DEFAULT}

    detection = _invoke_json(
        "detect_vague_references",
        DETECT_VAGUE_REFERENCES_PROMPT,
        f"Текст рекомендаций:\n{recommendations}",
        fallback={"items": []},
    )
    items = detection.get("items") if isinstance(detection, dict) else []
    if not isinstance(items, list) or not items:
        LOGGER.info("enrich_recommendations: no vague references detected")
        return {"processing_stage": STAGE_DEFAULT}

    max_items = settings.recommendation_enrichment_max_items
    items = items[:max_items]
    LOGGER.info("enrich_recommendations: detected %d vague references (cap=%d)", len(items), max_items)

    deal_type = state.get("deal_type") or "-"
    deal_description = state.get("deal_description", "")

    enrichments: list[dict] = []
    for idx, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        span = (item.get("span") or "").strip()
        query = (item.get("search_query") or "").strip()
        ctx = (item.get("context_for_subagent") or "").strip()
        vague = (item.get("vague_reference") or "").strip()
        if not span or not query:
            continue
        chunks = retrieve_secondary(query, top_k=settings.retrieval_secondary_enrichment_top_k)
        if not chunks:
            LOGGER.info("enrich_recommendations[%d]: query=%r yielded no chunks; skipping", idx, query)
            continue

        sub_human = (
            f"{SECTION_LABEL_DEAL_TYPE} {deal_type}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{deal_description}\n\n"
            f"Исходный фрагмент рекомендаций:\n{span}\n\n"
            f"Висячая отсылка: {vague or '(см. фрагмент)'}\n"
            f"Что нужно прояснить: {ctx or '(см. фрагмент)'}\n"
            f"Поисковый запрос: {query}\n\n"
            f"{SECTION_LABEL_NORMS}\n{_norms_to_context(chunks, limit=len(chunks))}"
        )
        clarification = _invoke_text(
            f"rag_subagent_{idx}",
            RAG_SUBAGENT_PROMPT,
            sub_human,
            fallback="",
        ).strip()
        if not clarification:
            continue
        if "не удалось установить" in clarification.lower():
            LOGGER.info("enrich_recommendations[%d]: sub-agent could not establish a norm; skipping", idx)
            continue
        enrichments.append({"span": span, "clarification": clarification})

    if not enrichments:
        LOGGER.info("enrich_recommendations: no usable enrichments produced")
        return {"processing_stage": STAGE_DEFAULT}

    enrichments_block = "\n\n".join(
        f"### Уточнение {i}\nИсходный фрагмент: {entry['span']}\nУточнение: {entry['clarification']}"
        for i, entry in enumerate(enrichments, 1)
    )
    integrate_human = f"Текущий текст рекомендаций:\n{recommendations}\n\nУточнения:\n{enrichments_block}"
    integrated, integrate_exchange = _invoke_text_capturing(
        "integrate_enrichments",
        INTEGRATE_ENRICHMENTS_PROMPT,
        integrate_human,
        fallback=recommendations,
    )
    integrated = integrated.strip()
    if not integrated:
        integrated = recommendations
    LOGGER.info("enrich_recommendations: integrated %d enrichments", len(enrichments))
    update: dict = {"recommendations": integrated, "processing_stage": STAGE_DEFAULT}
    # Successful integration supersedes generate_recommendations' exchange.
    if integrate_exchange:
        update["final_recommendations_exchange"] = integrate_exchange
    return update


def generate_contract(state: ContractAgentState) -> dict:
    fallback_html = CONTRACT_HTML_BODY_TEMPLATE
    contract_html, exchange = _invoke_text_capturing(
        "generate_contract",
        GENERATE_CONTRACT_PROMPT,
        (
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_TYPE} {state.get('deal_type') or '-'}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{state.get('deal_description', '')}\n\n"
            f"{SECTION_LABEL_DEAL_STRUCTURE}\n{json.dumps(state.get('deal_structure') or {}, ensure_ascii=False, indent=2)}\n\n"
            f"{SECTION_LABEL_GENERAL_NORMS}\n{_norms_to_context(state.get('general_norms'))}\n\n"
            f"{SECTION_LABEL_SPECIFIC_NORMS}\n{_norms_to_context(state.get('retrieved_norms'))}\n\n"
            f"{SECTION_LABEL_VALIDATION_ERRORS}\n- " + "\n- ".join(state.get("validation_errors") or ["нет"])
        ),
        fallback=fallback_html,
    )
    contract_preview = contract_html_to_text(contract_html)
    return {
        "contract_html": contract_html,
        "contract_md": contract_preview,
        "final_contract_exchange": exchange or None,
        "processing_stage": STAGE_VALIDATE_CONTRACT,
    }


def edit_contract(state: ContractAgentState) -> dict:
    """Apply a targeted user-driven edit to the existing contract.

    Skips classification/retrieval/recommendations and only rewrites the
    affected pieces of the existing HTML, preserving structure and numbering.
    Produces a short summary for chat ("changed clause 4.1: price...") and
    leaves the rest of the contract verbatim.
    """
    last_user = _last_user_message(state) or "(пустое сообщение)"
    current_html = (state.get("contract_html") or "").strip() or CONTRACT_HTML_BODY_TEMPLATE
    fallback = {
        "contract_html": current_html,
        "edit_summary": "Не удалось точно определить, что нужно изменить — уточните запрос.",
        "changed_sections": [],
    }
    result, exchange = _invoke_json_capturing(
        "edit_contract",
        EDIT_CONTRACT_PROMPT,
        (f"Запрос пользователя:\n{last_user}\n\nТекущий HTML-договор:\n{current_html}"),
        fallback=fallback,
    )
    new_html = (result.get("contract_html") or current_html).strip() or current_html
    summary = str(result.get("edit_summary") or "").strip()
    if not summary:
        summary = "Договор обновлён по запросу пользователя."
    changed_sections = result.get("changed_sections") or []
    if not isinstance(changed_sections, list):
        changed_sections = [str(changed_sections)]
    changed_sections = [str(item).strip() for item in changed_sections if str(item).strip()]
    contract_preview = contract_html_to_text(new_html)
    return {
        "contract_html": new_html,
        "contract_md": contract_preview,
        "edit_summary": summary,
        "edit_changed_sections": changed_sections,
        "final_contract_exchange": exchange or None,
        "processing_stage": STAGE_VALIDATE_CONTRACT,
    }


def validate_contract(state: ContractAgentState) -> dict:
    fallback = {"valid": bool(state.get("contract_md")), "errors": []}
    contract_text = state.get("contract_md") or ""
    result = _invoke_json(
        "validate_contract",
        VALIDATE_CONTRACT_PROMPT,
        (
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_TYPE} {state.get('deal_type') or '-'}\n\n"
            f"Договор:\n{contract_text}\n\n"
            f"{SECTION_LABEL_GENERAL_NORMS}\n{_norms_to_context(state.get('general_norms'))}\n\n"
            f"{SECTION_LABEL_SPECIFIC_NORMS}\n{_norms_to_context(state.get('retrieved_norms'))}"
        ),
        fallback=fallback,
    )
    errors = result.get("errors") or []
    if not isinstance(errors, list):
        errors = [str(errors)]
    errors = [str(item) for item in errors]
    contract_valid = bool(result.get("valid", False)) or not errors
    return {
        "validation_errors": errors,
        "contract_valid": contract_valid,
        "processing_stage": STAGE_DEFAULT if contract_valid else STAGE_GENERATE_CONTRACT,
    }


def handle_validation_error(state: ContractAgentState) -> dict:
    return {
        "iteration_count": int(state.get("iteration_count", 0)) + 1,
        "processing_stage": STAGE_GENERATE_CONTRACT,
    }


def save_result(state: ContractAgentState) -> dict:
    case_id = state.get("case_id")
    if not case_id:
        return {"result_saved": False, "processing_stage": STAGE_DEFAULT}

    # Terminal validation failure: exhausted all retries with errors remaining.
    validation_exhausted = bool(
        state.get("requires_written_form")
        and state.get("validation_errors")
        and int(state.get("iteration_count", 0)) >= int(state.get("max_iterations", settings.max_iterations))
    )
    general_blocked = not state.get("general_check_passed", True)

    status = CaseStatus.ERROR.value if (general_blocked or validation_exhausted) else CaseStatus.COMPLETED.value

    docx_path = None
    # Only produce a DOCX when the contract is actually valid and ready to deliver.
    if state.get("contract_html") and status == CaseStatus.COMPLETED.value:
        try:
            latest_version = get_latest_version(case_id)
            next_version = 1 if latest_version is None else latest_version.version_number + 1
            local_docx_path = generate_docx(state["contract_html"], str(case_id), next_version, state.get("deal_type"))
            docx_path = local_docx_path
            if settings.s3_enabled:
                s3_uri = upload_contract_docx(local_docx_path, str(case_id), next_version)
                try:
                    Path(local_docx_path).unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    LOGGER.warning("Could not delete local DOCX after S3 upload (%s): %s", local_docx_path, cleanup_exc)
                docx_path = s3_uri
            save_contract_version(case_id, state.get("contract_md") or state["contract_html"], docx_path)
        except Exception as exc:
            LOGGER.warning("Failed to save contract version: %s", exc)
            status = CaseStatus.ERROR.value
            docx_path = None

    try:
        update_case_status(case_id, status)
    except Exception as exc:
        LOGGER.warning("Failed to update case status: %s", exc)

    # Dump only on COMPLETED runs to avoid polluting artifacts with failed/exhausted attempts.
    if status == CaseStatus.COMPLETED.value and settings.llm_dump_final_enabled:
        deal_type = state.get("deal_type")
        contract_exchange = state.get("final_contract_exchange") or {}
        if docx_path and contract_exchange:
            _dump_final_exchange(
                "contract",
                deal_type,
                contract_exchange.get("system", ""),
                contract_exchange.get("human", ""),
                contract_exchange.get("response", ""),
            )
        recommendations_exchange = state.get("final_recommendations_exchange") or {}
        if (state.get("recommendations") or "").strip() and recommendations_exchange:
            _dump_final_exchange(
                "recomendations",
                deal_type,
                recommendations_exchange.get("system", ""),
                recommendations_exchange.get("human", ""),
                recommendations_exchange.get("response", ""),
            )

    try:
        with session_scope() as session:
            case = session.get(Case, UUID(str(case_id)))
            if case is not None:
                case.deal_type = state.get("deal_type")
    except Exception as exc:
        LOGGER.warning("Failed to persist deal_type: %s", exc)

    recommendations_text = (state.get("recommendations") or "").strip()
    docx_saved = bool(docx_path)
    edit_summary = (state.get("edit_summary") or "").strip()
    is_edit_run = state.get("intent") == "edit" and bool(edit_summary)

    if validation_exhausted:
        final_text = (
            f"{recommendations_text}\n\n---\n\n{ERROR_GENERIC_TEXT}" if recommendations_text else ERROR_GENERIC_TEXT
        )
    elif is_edit_run:
        final_text = f"{edit_summary}\n\n{NEW_VERSION_SAVED_AS_DOCX_MESSAGE}"
    elif recommendations_text and docx_saved:
        final_text = f"{recommendations_text}\n\n---\n\n{NEW_VERSION_SAVED_AS_DOCX_MESSAGE}"
    elif recommendations_text:
        final_text = recommendations_text
    elif docx_saved:
        final_text = NEW_VERSION_SAVED_AS_DOCX_MESSAGE
    else:
        final_text = "Результат обработки сохранён."

    update = {
        "result_saved": True,
        "result_docx_path": docx_path,
        "processing_stage": STAGE_DEFAULT,
    }
    update.update(_state_messages_update(final_text))
    return update
