"""Pure-function tests for the gate_free_plan graph node.

The node has no LLM, no DB, no IO — it inspects state and returns a patch
dict. We assert the patch shape for the three relevant cases.
"""

from __future__ import annotations

from agent.gate import FREE_EDIT_REFUSAL_TEXT, gate_free_plan


def test_passthrough_when_intent_is_not_edit():
    state = {"intent": "regenerate", "user_plan": "free", "allow_edit": False}
    assert gate_free_plan(state) == {}


def test_passthrough_when_edit_allowed():
    state = {"intent": "edit", "user_plan": "monthly", "allow_edit": True}
    assert gate_free_plan(state) == {}


def test_blocks_edit_for_free_plan():
    state = {"intent": "edit", "user_plan": "free", "allow_edit": False, "case_id": "c1"}
    patch = gate_free_plan(state)
    assert patch["intent"] == "followup"
    assert patch["result_saved"] is True
    assert patch["processing_stage"] == "default"
    msgs = patch["messages"]
    assert isinstance(msgs, list) and len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == FREE_EDIT_REFUSAL_TEXT


def test_default_allow_edit_is_truthy():
    """When `allow_edit` is missing from state, we should NOT block — old
    sessions started before this change shouldn't suddenly fail."""
    state = {"intent": "edit"}  # no allow_edit, no user_plan
    assert gate_free_plan(state) == {}
