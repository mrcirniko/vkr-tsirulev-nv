"""Standalone, dependency-light gate node.

Lives in its own module so the test suite can import it without dragging in
LangChain / Ollama / html2docx — `agent.nodes` pulls those in eagerly. The
graph re-exports it through `agent.nodes` for registration symmetry.
"""

from __future__ import annotations

import logging

from agent.state import ContractAgentState

LOGGER = logging.getLogger("agent.gate")

FREE_EDIT_REFUSAL_TEXT = (
    "Правка договора доступна только на платных тарифах. "
    "Чтобы продолжить редактирование, обновите подписку в разделе «Тарифы» "
    "(меню профиля слева снизу)."
)


def gate_free_plan(state: ContractAgentState) -> dict:
    """Block edit-mode generations for users on the FREE plan.

    `route_user_message` runs the LLM-driven intent classifier; we can't
    decide before that. Once intent is known, this node short-circuits the
    graph with a refusal message instead of spending cycles on edit_contract.
    Followup-mode and regenerate-mode pass through unchanged: free users keep
    asking questions about their existing contract, but cannot mutate it.
    """
    if state.get("intent") != "edit":
        return {}
    if state.get("allow_edit", True):
        return {}
    LOGGER.info(
        "gate_free_plan blocking edit case_id=%s plan=%s",
        state.get("case_id"),
        state.get("user_plan"),
    )
    return {
        "intent": "followup",
        "messages": [{"role": "assistant", "content": FREE_EDIT_REFUSAL_TEXT}],
        "processing_stage": "default",
        "result_saved": True,
    }
