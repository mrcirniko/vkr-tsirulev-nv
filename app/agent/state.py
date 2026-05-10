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
    # Billing context: pushed in by server.py when starting a run so
    # `gate_free_plan` can short-circuit edit requests on the free tier
    # without an extra DB call from inside graph nodes.
    user_plan: str | None
    allow_edit: bool
    # User preferences (also injected by server.py per run). They override the
    # default written-form heuristic — see `_route_after_written_form`.
    contract_generation_policy: str | None  # legal_only | always | always_ask
    ask_personal_data: bool
    # Stashed LLM exchanges (system / human prompts and raw response) for the
    # final-version dump in save_result. Overwritten on each retry of
    # generate_contract / edit_contract and on each pass through the
    # recommendations pipeline; only the last (post-validation) value lands on
    # disk. Cheap to carry, small enough not to bloat checkpoints.
    final_contract_exchange: dict[str, str] | None
    final_recommendations_exchange: dict[str, str] | None
