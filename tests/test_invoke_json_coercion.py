"""Tests for the lenient JSON extractor used by every LLM-driven graph node.

Real-world Ollama replies sometimes come wrapped in a ```json fence``` or
prepended with a sentence. `coerce_json` is the surface that absorbs that
noise; if it regresses, every JSON-mode node degrades to its fallback.
"""

from __future__ import annotations

import json

import pytest

from agent.json_utils import coerce_json


def test_plain_json():
    assert coerce_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_strips_markdown_fence_with_lang_tag():
    raw = '```json\n{"requires_written_form": true, "reason": "ст. 161 ГК РФ"}\n```'
    assert coerce_json(raw) == {"requires_written_form": True, "reason": "ст. 161 ГК РФ"}


def test_strips_markdown_fence_without_lang_tag():
    assert coerce_json('```\n{"x": 42}\n```') == {"x": 42}


def test_extracts_object_from_chatty_reply():
    raw = 'Конечно! Вот результат: {"missing_fields": ["цена"]}\nГотов помочь дальше.'
    assert coerce_json(raw) == {"missing_fields": ["цена"]}


def test_handles_leading_whitespace_only():
    assert coerce_json('   \n\n{"ok": true}\n') == {"ok": True}


def test_handles_nested_object():
    raw = '{"deal_structure": {"цена": 5000, "срок": "2026-06-01"}, "missing_fields": []}'
    parsed = coerce_json(raw)
    assert parsed["deal_structure"]["цена"] == 5000
    assert parsed["missing_fields"] == []


def test_raises_on_no_json_at_all():
    with pytest.raises(json.JSONDecodeError):
        coerce_json("just some text no curly braces here at all")


def test_raises_on_empty_string():
    with pytest.raises(json.JSONDecodeError):
        coerce_json("")
