"""Pure routing functions for the contract agent graph.

Kept dependency-free (no LLM, no DB, no LangGraph imports) so unit tests can
load these helpers without dragging in the full agent stack.
"""

from __future__ import annotations

from agent.state import ContractAgentState


def route_after_data_sufficiency(state: ContractAgentState) -> str:
    """`check_data_sufficiency` decides between asking and generating.

    Sits right before `generate_contract` after the rewiring (data check no
    longer fires when we won't be generating a contract anyway).
    """
    if state.get("clarification_needed") and state.get("clarification_stage") == "data_sufficiency":
        return "ask_clarification"
    return "generate_contract"


def route_after_optional_contract_answer(state: ContractAgentState) -> str:
    """User answered the optional-contract yes/no.

    Yes → still need to verify we have ФИО/паспорт/etc before drafting the
    contract. No → write recommendations and stop.
    """
    return "check_data_sufficiency" if state.get("generate_optional_contract") else "generate_recommendations"


def route_after_written_form(state: ContractAgentState) -> str:
    """Branch on the user's contract_generation_policy preference.

    Returns the next graph node name:

    - "check_data_sufficiency" — we ARE going to generate a contract; first
      verify the required party/deal data is on hand. (This used to point at
      generate_contract directly, but we moved data sufficiency here so we
      don't pester the user for ФИО when the answer ends up being "no
      contract needed".)
    - "generate_recommendations" — no contract will be produced, skip data
      collection and just write the recommendations.
    - "ask_clarification" — pause and ask the user whether they want a
      written contract (only on `always_ask` when the law allows oral form).

    Policies:
    - always_ask: when the law requires a written form, generate; otherwise
      pause for a yes/no clarification (the legacy default).
    - legal_only: generate only if required by law; otherwise skip straight
      to recommendations without asking.
    - always: generate every time, even if the law allows an oral deal.

    `requires_written_form` is set by the LLM-driven `check_written_form`
    node; the policy gate above only fires after that has run. Unknown
    policies degrade to `always_ask`.
    """
    policy = state.get("contract_generation_policy") or "always_ask"
    requires_written_form = bool(state.get("requires_written_form", True))

    if policy == "always":
        return "check_data_sufficiency"
    if policy == "legal_only":
        return "check_data_sufficiency" if requires_written_form else "generate_recommendations"
    # always_ask (or anything unknown): paused on a clarification when not
    # required by law, otherwise fall through to the legal-status decision.
    if state.get("clarification_needed") and state.get("clarification_stage") == "optional_contract_generation":
        return "ask_clarification"
    return "check_data_sufficiency" if requires_written_form else "generate_recommendations"
