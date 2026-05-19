"""Graph-routing tests for the user-preferences feature.

Covers `route_after_written_form` × `contract_generation_policy` matrix.
After the rewiring (data-sufficiency now sits AFTER written-form decision),
"go generate a contract" is encoded as `check_data_sufficiency` — that node
either pauses for a clarification or hands off to `generate_contract`.
"""

from __future__ import annotations

from agent.routing import route_after_written_form


def _state(**overrides):
    base = {
        "requires_written_form": False,
        "contract_generation_policy": "always_ask",
        "clarification_needed": False,
        "clarification_stage": None,
    }
    base.update(overrides)
    return base


def test_always_ask_with_pending_clarification_routes_to_ask():
    state = _state(
        clarification_needed=True,
        clarification_stage="optional_contract_generation",
    )
    assert route_after_written_form(state) == "ask_clarification"


def test_always_ask_with_legal_requirement_skips_clarification():
    """When the law requires a written form, even `always_ask` stops asking."""
    state = _state(requires_written_form=True)
    assert route_after_written_form(state) == "check_data_sufficiency"


def test_always_ask_without_legal_requirement_falls_through_when_no_clarification():
    state = _state(requires_written_form=False)
    assert route_after_written_form(state) == "generate_recommendations"


def test_legal_only_skips_optional_contract():
    state = _state(
        contract_generation_policy="legal_only",
        clarification_needed=True,  # ignored under this policy
        clarification_stage="optional_contract_generation",
    )
    assert route_after_written_form(state) == "generate_recommendations"


def test_legal_only_still_collects_data_when_required_by_law():
    state = _state(contract_generation_policy="legal_only", requires_written_form=True)
    assert route_after_written_form(state) == "check_data_sufficiency"


def test_always_collects_data_regardless_of_legal_status():
    state = _state(contract_generation_policy="always", requires_written_form=False)
    assert route_after_written_form(state) == "check_data_sufficiency"


def test_always_ignores_pending_clarification_too():
    """An `always_ask` clarification could leak in if the policy was changed
    mid-thread. The router treats `always` as authoritative either way."""
    state = _state(
        contract_generation_policy="always",
        clarification_needed=True,
        clarification_stage="optional_contract_generation",
    )
    assert route_after_written_form(state) == "check_data_sufficiency"


def test_unknown_policy_falls_back_to_always_ask():
    """Forward-compat: a policy code we don't recognise behaves like the
    permissive default rather than 500-ing the run."""
    state = _state(contract_generation_policy="something_new")
    assert route_after_written_form(state) == "generate_recommendations"


def test_default_policy_when_state_missing_field():
    """Old threads created before this feature shipped have no policy field —
    they should keep the legacy `always_ask` behaviour."""
    state = _state()
    state.pop("contract_generation_policy", None)
    assert route_after_written_form(state) == "generate_recommendations"


def test_data_sufficiency_router_paths():
    """Light coverage of the sister router that runs AFTER written-form."""
    from agent.routing import route_after_data_sufficiency

    paused = {"clarification_needed": True, "clarification_stage": "data_sufficiency"}
    assert route_after_data_sufficiency(paused) == "ask_clarification"

    ready = {"clarification_needed": False}
    assert route_after_data_sufficiency(ready) == "generate_contract"

    # An UNRELATED clarification (e.g. classification stage from earlier in the
    # graph) must NOT divert this router — only data_sufficiency stops here.
    other = {"clarification_needed": True, "clarification_stage": "classification"}
    assert route_after_data_sufficiency(other) == "generate_contract"


def test_optional_contract_answer_yes_routes_through_data_sufficiency():
    from agent.routing import route_after_optional_contract_answer

    yes = {"generate_optional_contract": True}
    assert route_after_optional_contract_answer(yes) == "check_data_sufficiency"

    no = {"generate_optional_contract": False}
    assert route_after_optional_contract_answer(no) == "generate_recommendations"
