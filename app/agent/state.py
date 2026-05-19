from __future__ import annotations

from typing import Any

from langgraph.graph import MessagesState


class ContractAgentState(MessagesState, total=False):
    case_id: str
    deal_description: str
    deal_type: str | None
    deal_type_confidence: str | None
    deal_classify_confidence: str | None
    classification_clarification_attempts: int
    max_classification_clarifications: int
    clarification_stage: str | None
    deal_structure: dict[str, Any] | None
    retrieval_query: str | None
    general_norms: list[dict[str, Any]] | None
    retrieved_norms: list[dict[str, Any]] | None
    general_check_passed: bool
    general_check_issues: list[str] | None
    general_check_explanation: str | None
    clarification_needed: bool
    clarification_question: str | None
    missing_fields: list[str] | None
    requires_written_form: bool | None
    written_form_reason: str | None
    generate_optional_contract: bool | None
    contract_md: str | None
    contract_html: str | None
    recommendations: str | None
    validation_errors: list[str] | None
    iteration_count: int
    max_iterations: int
    result_docx_path: str | None
    result_saved: bool
    processing_stage: str | None
    intent: str | None
    contract_valid: bool | None
    edit_summary: str | None
    edit_changed_sections: list[str] | None
    # Billing/preferences injected by server.py per run — graph nodes read without extra DB calls.
    user_plan: str | None
    allow_edit: bool
    contract_generation_policy: str | None  # legal_only | always | always_ask
    ask_personal_data: bool
    # Last-pass LLM exchanges for save_result dump; overwritten on each retry.
    final_contract_exchange: dict[str, str] | None
    final_recommendations_exchange: dict[str, str] | None
    # Wall-clock latency of retrieve_norms only (captured inside the node).
    retrieve_norms_latency_ms: float | None
    # Follow-up sub-agent state: brief, question and query history persist across attempts;
    # per-attempt chunks are not carried forward.
    subagent_active: bool
    subagent_brief: str | None
    subagent_user_question: str | None
    subagent_query: str | None
    subagent_query_history: list[str] | None
    subagent_attempts: int
    subagent_max_attempts: int
