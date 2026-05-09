from __future__ import annotations

from config import settings

# from functools import lru_cache  # only used by commented-out get_contract_agent
# from pathlib import Path  # only used by commented-out save_graph_visualization
# from typing import Any  # only used by commented-out local-invocation API below
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.routing import (
    route_after_data_sufficiency as _route_after_data_sufficiency,
)
from agent.routing import (
    route_after_optional_contract_answer as _route_after_optional_contract_answer,
)
from agent.routing import (
    route_after_written_form as _route_after_written_form,
)
from agent.state import ContractAgentState


def _route_after_classification(state: ContractAgentState) -> str:
    if state.get("clarification_needed") and state.get("clarification_stage") == "classification":
        return "ask_clarification"
    if not state.get("deal_type"):
        return "inform_unsupported_deal"
    return "check_general_norms"


def _route_after_general_norms(state: ContractAgentState) -> str:
    return "inform_user" if not state.get("general_check_passed", True) else "retrieve_norms"


# `_route_after_data_sufficiency` and `_route_after_optional_contract_answer`
# live in agent/routing.py so the unit tests don't have to load the full
# LLM stack to exercise them.


def _route_after_validation(state: ContractAgentState) -> str:
    # In edit-mode runs we never loop back through generate_contract — the
    # whole point is to apply a targeted user-driven edit and stop. Validation
    # results are surfaced to the user via save_result; on persistent issues
    # the user can ask again.
    if state.get("intent") == "edit":
        return "save_result"
    if state.get("contract_valid"):
        return "generate_recommendations"
    iteration_count = int(state.get("iteration_count", 0))
    max_iterations = int(state.get("max_iterations", settings.max_iterations))
    if iteration_count < max_iterations:
        return "handle_validation_error"
    return "generate_recommendations"


def _route_after_user_message(state: ContractAgentState) -> str:
    intent = state.get("intent")
    if intent == "edit" and not state.get("allow_edit", True):
        # Free plan tried to edit a contract — gate_free_plan rewrites the
        # last assistant message with a refusal and ends the run.
        return "gate_free_plan"
    if intent == "followup":
        return "followup_response"
    if intent == "edit":
        return "edit_contract"
    return "load_general_norms"


def build_state_graph() -> StateGraph:
    graph = StateGraph(ContractAgentState)

    graph.add_node("route_user_message", nodes.route_user_message)
    graph.add_node("gate_free_plan", nodes.gate_free_plan)
    graph.add_node("followup_response", nodes.followup_response)
    graph.add_node("load_general_norms", nodes.load_general_norms)
    graph.add_node("classify_deal", nodes.classify_deal)
    graph.add_node("inform_unsupported_deal", nodes.inform_unsupported_deal)
    graph.add_node("check_general_norms", nodes.check_general_norms)
    graph.add_node("inform_user", nodes.inform_user)
    graph.add_node("retrieve_norms", nodes.retrieve_norms)
    graph.add_node("check_data_sufficiency", nodes.check_data_sufficiency)
    graph.add_node("ask_clarification", nodes.ask_clarification)
    graph.add_node("check_written_form", nodes.check_written_form)
    graph.add_node("optional_contract_answer", lambda state: {})
    graph.add_node("generate_recommendations", nodes.generate_recommendations)
    graph.add_node("enrich_recommendations", nodes.enrich_recommendations)
    graph.add_node("generate_contract", nodes.generate_contract)
    graph.add_node("edit_contract", nodes.edit_contract)
    graph.add_node("validate_contract", nodes.validate_contract)
    graph.add_node("handle_validation_error", nodes.handle_validation_error)
    graph.add_node("save_result", nodes.save_result)

    graph.add_edge(START, "route_user_message")
    graph.add_conditional_edges(
        "route_user_message",
        _route_after_user_message,
        {
            "followup_response": "followup_response",
            "edit_contract": "edit_contract",
            "load_general_norms": "load_general_norms",
            "gate_free_plan": "gate_free_plan",
        },
    )
    graph.add_edge("edit_contract", "validate_contract")
    graph.add_edge("followup_response", END)
    graph.add_edge("gate_free_plan", END)
    graph.add_edge("load_general_norms", "classify_deal")
    graph.add_conditional_edges(
        "classify_deal",
        _route_after_classification,
        {
            "ask_clarification": "ask_clarification",
            "inform_unsupported_deal": "inform_unsupported_deal",
            "check_general_norms": "check_general_norms",
        },
    )
    graph.add_edge("ask_clarification", END)
    graph.add_edge("inform_unsupported_deal", END)
    graph.add_conditional_edges(
        "check_general_norms",
        _route_after_general_norms,
        {
            "inform_user": "inform_user",
            "retrieve_norms": "retrieve_norms",
        },
    )
    graph.add_edge("inform_user", END)
    # New ordering: norms → written_form decision → (only when we'll generate
    # a contract) → data sufficiency → generate_contract. This prevents the
    # agent from harassing users for ФИО/паспорт/etc when the contract isn't
    # going to be generated anyway (e.g. legal_only + oral deal).
    graph.add_edge("retrieve_norms", "check_written_form")
    graph.add_conditional_edges(
        "check_written_form",
        _route_after_written_form,
        {
            "ask_clarification": "ask_clarification",
            "generate_recommendations": "generate_recommendations",
            "check_data_sufficiency": "check_data_sufficiency",
        },
    )
    graph.add_conditional_edges(
        "check_data_sufficiency",
        _route_after_data_sufficiency,
        {
            "ask_clarification": "ask_clarification",
            "generate_contract": "generate_contract",
        },
    )
    graph.add_conditional_edges(
        "optional_contract_answer",
        _route_after_optional_contract_answer,
        {
            "generate_recommendations": "generate_recommendations",
            "check_data_sufficiency": "check_data_sufficiency",
        },
    )
    graph.add_edge("generate_recommendations", "enrich_recommendations")
    graph.add_edge("enrich_recommendations", "save_result")
    graph.add_edge("generate_contract", "validate_contract")
    graph.add_conditional_edges(
        "validate_contract",
        _route_after_validation,
        {
            "handle_validation_error": "handle_validation_error",
            "generate_recommendations": "generate_recommendations",
            "save_result": "save_result",
        },
    )
    graph.add_edge("handle_validation_error", "generate_contract")
    graph.add_edge("save_result", END)

    return graph


def compile_contract_agent(*, with_checkpointer: bool):
    graph = build_state_graph()
    if with_checkpointer:
        return graph.compile(checkpointer=MemorySaver())
    return graph.compile()


# UNUSED at runtime — these helpers form the local-invocation API (see also
# the commented-out block below). Re-exported by agent/__init__.py for
# convenience, but no caller imports them from there.
# @lru_cache(maxsize=1)
# def get_contract_agent():
#     return compile_contract_agent(with_checkpointer=True)
#
#
# def get_graph_mermaid() -> str:
#     return get_contract_agent().get_graph().draw_mermaid()
#
#
# def get_graph_png() -> bytes | None:
#     try:
#         return get_contract_agent().get_graph().draw_mermaid_png()
#     except Exception:
#         return None


# ---------------------------------------------------------------------------
# Local-invocation API (UNUSED at runtime).
#
# In production the graph runs inside langgraph_dev container; server.py talks
# to it via HTTP (`_get_thread_state`, `_resume_after_clarification_remote` in
# server.py). The functions below are kept as a Python-side library — handy
# for ad-hoc debugging from `docker compose exec app python` — but they are
# not part of any active code path and are commented out.
# ---------------------------------------------------------------------------

# def save_graph_visualization(output_dir: str | Path) -> dict[str, str]:
#     directory = Path(output_dir)
#     directory.mkdir(parents=True, exist_ok=True)
#
#     mermaid_path = directory / "contract_agent_graph.mmd"
#     mermaid_path.write_text(get_graph_mermaid(), encoding="utf-8")
#
#     result = {"mermaid": str(mermaid_path)}
#     png_bytes = get_graph_png()
#     if png_bytes:
#         png_path = directory / "contract_agent_graph.png"
#         png_path.write_bytes(png_bytes)
#         result["png"] = str(png_path)
#     return result
#
#
# def run_contract_agent(initial_state: dict[str, Any], thread_id: str):
#     state = {
#         "iteration_count": 0,
#         "max_iterations": settings.max_iterations,
#         "classification_clarification_attempts": 0,
#         "max_classification_clarifications": settings.max_classification_clarifications,
#         **initial_state,
#     }
#     return get_contract_agent().invoke(state, config={"configurable": {"thread_id": thread_id}})
#
#
# def stream_contract_agent(initial_state: dict[str, Any], thread_id: str):
#     state = {
#         "iteration_count": 0,
#         "max_iterations": settings.max_iterations,
#         "classification_clarification_attempts": 0,
#         "max_classification_clarifications": settings.max_classification_clarifications,
#         **initial_state,
#     }
#     return get_contract_agent().stream(state, config={"configurable": {"thread_id": thread_id}})
#
#
# def get_thread_state(thread_id: str) -> dict[str, Any]:
#     snapshot = get_contract_agent().get_state({"configurable": {"thread_id": thread_id}})
#     return snapshot.values if snapshot else {}
#
#
# def resume_after_clarification(case_id: str, clarification_answer: str):
#     agent = get_contract_agent()
#     config = {"configurable": {"thread_id": case_id}}
#     current_state = agent.get_state(config)
#     previous = current_state.values if current_state else {}
#     description = previous.get("deal_description", "").strip()
#     merged_description = (description + "\nУточнение пользователя: " + clarification_answer).strip()
#     clarification_stage = previous.get("clarification_stage")
#     next_node = "classify_deal" if clarification_stage == "classification" else "check_data_sufficiency"
#     agent.update_state(
#         config,
#         {
#             "deal_description": merged_description,
#             "clarification_needed": False,
#             "clarification_question": None,
#             "clarification_stage": None,
#         },
#         as_node=next_node,
#     )
#     return agent.invoke(None, config=config)
