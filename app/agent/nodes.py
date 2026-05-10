from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from uuid import UUID

from config import settings
from contract_docx import contract_html_to_text, generate_docx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from storage import upload_contract_docx

from agent.deal_types import get_supported_deal_types
from agent.json_utils import coerce_json
from messages import ERROR_GENERIC_TEXT
from agent.prompts import (
    BUILD_RETRIEVAL_QUERY_PROMPT,
    build_check_data_sufficiency_prompt,
    CHECK_GENERAL_NORMS_PROMPT,
    CHECK_WRITTEN_FORM_PROMPT,
    CLASSIFY_FOLLOWUP_INTENT_PROMPT,
    CLASSIFY_YESNO_PROMPT,
    CONTRACT_HTML_BODY_TEMPLATE,
    DETECT_VAGUE_REFERENCES_PROMPT,
    EDIT_CONTRACT_PROMPT,
    FOLLOWUP_RESPONSE_PROMPT,
    GENERATE_CASE_TITLE_PROMPT,
    GENERATE_CONTRACT_PROMPT,
    GENERATE_RECOMMENDATIONS_PROMPT,
    INFORM_USER_PROMPT,
    INTEGRATE_ENRICHMENTS_PROMPT,
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
    build_classify_deal_prompt,
    build_inform_unsupported_deal_prompt,
)
from agent.state import ContractAgentState
from db.crud import get_latest_version, save_contract_version, session_scope, update_case_status
from db.models import Case, CaseStatus
from rag.ollama import unload_ollama_model
from rag.retriever import retrieve_general, retrieve_secondary, retrieve_specific
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
NEW_VERSION_SAVED_AS_DOCX_MESSAGE = (
    "Договор сохранен в виде DOCX — откройте его в панели «Договоры» справа."
)


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
        # Edit only makes sense when a contract was actually generated
        # before. If the prior run only produced recommendations (no
        # contract_html), fall back to regenerate so the user gets a real
        # draft to edit on the next round.
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


# Re-export for graph wiring + back-compat. The implementation lives in
# agent/gate.py so tests can import it without dragging in LangChain / Ollama
# / html2docx that this module needs at import time.
from agent.gate import FREE_EDIT_REFUSAL_TEXT, gate_free_plan  # noqa: E402, F401


def followup_response(state: ContractAgentState) -> dict:
    """Conversational answer using existing case context — no contract regeneration."""
    last_user = _last_user_message(state) or "(пустое сообщение)"
    contract_preview = (state.get("contract_md") or "").strip() or "Договор пока не сформирован."
    recommendations = (state.get("recommendations") or "").strip() or "Рекомендации пока не сформированы."
    fallback = (
        "Похоже, у меня сейчас нет дополнительной информации по вашему вопросу. "
        "Уточните, пожалуйста, что именно вас интересует."
    )

    answer = _invoke_text(
        "followup_response",
        FOLLOWUP_RESPONSE_PROMPT,
        (
            f"{SECTION_LABEL_HISTORY}\n{_conversation_context(state)}\n\n"
            f"{SECTION_LABEL_DEAL_TYPE} {state.get('deal_type') or '-'}\n\n"
            f"{SECTION_LABEL_DEAL_DESCRIPTION}\n{state.get('', '')}\n\n"
            f"{SECTION_LABEL_GENERAL_NORMS}\n{_norms_to_context(state.get('general_norms'))}\n\n"
            f"{SECTION_LABEL_SPECIFIC_NORMS}\n{_norms_to_context(state.get('retrieved_norms'))}\n\n"
            f"Текущие рекомендации (ранее отправлены пользователю):\n{recommendations}\n\n"
            f"Текущий проект договора (текстовое превью):\n{contract_preview}\n\n"
            f"Новое сообщение пользователя:\n{last_user}"
        ),
        fallback=fallback,
    )

    return {
        **_state_messages_update(answer),
        "intent": "followup",
        "processing_stage": STAGE_DEFAULT,
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
    text = " ".join(part for part in (deal_type or "", deal_description or "") if part).strip()
    return infer_specific_source_filter(text) if text else None


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
    # General norms are retrieved after classification using the case query.
    # Keeping this node as a no-op preserves the graph shape and old saved runs.
    return {"processing_stage": STAGE_CLASSIFY_CONTRACT}


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
    deal_type = result.get("deal_type")
    if deal_type not in get_supported_deal_types():
        deal_type = None

    confidence = str(result.get("confidence", "low")).lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "low"

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
    # NOTE: result_saved stays False — we did not actually produce any
    # artifact. The next user message must go through the full regenerate
    # path (classify_deal etc.) so re-described deals get a real chance.

    # Sync sidebar status: surface "Требует внимания" so the user can tell
    # something didn't go through. A subsequent successful run will reset
    # the status to COMPLETED via save_result.
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

    # Sync sidebar status: surface "Требует внимания" so the user sees that
    # the run terminated due to general-norms issues. A later successful run
    # will reset the status via save_result.
    case_id = state.get("case_id")
    if case_id:
        try:
            update_case_status(case_id, CaseStatus.ERROR.value)
        except Exception as exc:
            LOGGER.warning("Failed to update case status from inform_user: %s", exc)

    return {**_state_messages_update(message), "processing_stage": STAGE_DEFAULT}


def retrieve_norms(state: ContractAgentState) -> dict:
    deal_type = state.get("deal_type")
    description = state.get("deal_description", "")
    retrieval_query = (state.get("retrieval_query") or _build_retrieval_query(state)).strip()
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
    LOGGER.info("Retriever response\nchunks=%s\nsources=%s", len(norms), [item.get("source") for item in norms[:10]])
    return {"retrieved_norms": norms, "processing_stage": STAGE_ANALYZE_NORMS}


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
        # Sufficient data → next stage in the pipeline is generate_contract
        # (data sufficiency now sits right before it). Pause on clarification
        # gets the default "talking with user" stage.
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
    # Honour per-user preferences. `always_ask` is the legacy default — only
    # then do we surface a clarification when the law doesn't require a
    # written contract. `legal_only` and `always` skip the question entirely
    # (the router below picks the appropriate next branch).
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
    recommendations = _invoke_text(
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
    return {"recommendations": recommendations, "processing_stage": STAGE_ENRICH_RECOMMENDATIONS}


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
    integrated = _invoke_text(
        "integrate_enrichments",
        INTEGRATE_ENRICHMENTS_PROMPT,
        f"Текущий текст рекомендаций:\n{recommendations}\n\nУточнения:\n{enrichments_block}",
        fallback=recommendations,
    ).strip()
    if not integrated:
        integrated = recommendations
    LOGGER.info("enrich_recommendations: integrated %d enrichments", len(enrichments))
    return {"recommendations": integrated, "processing_stage": STAGE_DEFAULT}


def generate_contract(state: ContractAgentState) -> dict:
    fallback_html = CONTRACT_HTML_BODY_TEMPLATE
    contract_html = _invoke_text(
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
    result = _invoke_json(
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
            f"{recommendations_text}\n\n---\n\n{ERROR_GENERIC_TEXT}"
            if recommendations_text
            else ERROR_GENERIC_TEXT
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
