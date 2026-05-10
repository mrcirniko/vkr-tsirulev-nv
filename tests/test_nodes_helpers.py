"""Unit tests for pure helpers inside agent.nodes.

Avoids touching ChatOllama by patching `_invoke_messages` for the few
LLM-dependent functions we cover. Helpers like `_norms_to_context` and
`_conversation_context` are pure formatters and tested directly.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.nodes import (
    _conversation_context,
    _dump_final_exchange,
    _last_user_message,
    _norms_to_context,
    _state_messages_update,
    classify_yesno,
)


# ---------------------------------------------------------------- _norms_to_context


def test_norms_to_context_returns_placeholder_for_empty():
    assert _norms_to_context([]) == "Нормы не найдены."
    assert _norms_to_context(None) == "Нормы не найдены."


def test_norms_to_context_renders_each_chunk_with_metadata():
    norms = [
        {"text": "Текст первой статьи", "source": "ГК РФ", "article": "1", "chapter": "Глава 1"},
        {"text": "Текст второй", "source": "ГК РФ", "article": "2", "chapter": "Глава 1"},
    ]
    rendered = _norms_to_context(norms)
    assert "ГК РФ" in rendered
    assert "Статья: 1" in rendered
    assert "Статья: 2" in rendered
    assert "Текст первой статьи" in rendered
    # Two chunks separated by a divider.
    assert "---" in rendered


def test_norms_to_context_truncates_per_chunk_text():
    long_text = "x" * 30000
    norms = [{"text": long_text, "source": "S", "article": "1", "chapter": "-"}]
    rendered = _norms_to_context(norms, text_limit=100)
    # Per-chunk limit is enforced before assembly.
    assert "x" * 100 in rendered
    assert "x" * 200 not in rendered


def test_norms_to_context_respects_limit_count():
    norms = [
        {"text": f"text-{i}", "source": "S", "article": str(i), "chapter": "-"} for i in range(50)
    ]
    rendered = _norms_to_context(norms, limit=3)
    assert "text-0" in rendered
    assert "text-2" in rendered
    assert "text-3" not in rendered


def test_norms_to_context_caps_total_chars():
    """The function must stop appending once the cumulative character budget
    is reached, even if there are more chunks under the per-chunk limit."""
    norms = [
        {"text": "a" * 1000, "source": "S", "article": str(i), "chapter": "-"} for i in range(20)
    ]
    rendered = _norms_to_context(norms, limit=20, text_limit=2000, max_total_chars=3000)
    # Cumulative cap kicks in well before exhausting all 20 chunks.
    assert len(rendered) <= 3500  # cap + minor overhead from separators


# ---------------------------------------------------------------- _conversation_context


def test_conversation_context_renders_roles():
    state = {
        "messages": [
            HumanMessage(content="привет"),
            AIMessage(content="здравствуйте"),
            HumanMessage(content="как дела?"),
        ]
    }
    rendered = _conversation_context(state)
    assert "User: привет" in rendered
    assert "Assistant: здравствуйте" in rendered
    assert "User: как дела?" in rendered


def test_conversation_context_handles_empty_messages():
    assert _conversation_context({"messages": []}) == "Conversation history is empty."
    assert _conversation_context({}) == "Conversation history is empty."


def test_conversation_context_skips_blank_content():
    state = {"messages": [HumanMessage(content="   "), AIMessage(content="ответ")]}
    rendered = _conversation_context(state)
    assert "User:" not in rendered
    assert "Assistant: ответ" in rendered


def test_conversation_context_handles_dict_messages():
    """Internal LangGraph state may serialize messages as dicts after a
    checkpoint round-trip — both shapes must be supported."""
    state = {"messages": [{"role": "user", "content": "hi"}, {"type": "ai", "content": "hello"}]}
    rendered = _conversation_context(state)
    assert "User: hi" in rendered
    assert "Assistant: hello" in rendered


# ---------------------------------------------------------------- _last_user_message


def test_last_user_message_picks_most_recent_user_turn():
    state = {
        "messages": [
            HumanMessage(content="первый"),
            AIMessage(content="ответ"),
            HumanMessage(content="второй"),
            AIMessage(content="ещё ответ"),
        ]
    }
    assert _last_user_message(state) == "второй"


def test_last_user_message_returns_empty_when_no_user_messages():
    state = {"messages": [AIMessage(content="привет")]}
    assert _last_user_message(state) == ""


def test_last_user_message_strips_whitespace():
    state = {"messages": [HumanMessage(content="   текст с пробелами   ")]}
    assert _last_user_message(state) == "текст с пробелами"


# ---------------------------------------------------------------- _state_messages_update


def test_state_messages_update_wraps_text_into_ai_message():
    update = _state_messages_update("ответ агента")
    assert "messages" in update
    assert len(update["messages"]) == 1
    msg = update["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content == "ответ агента"


# ---------------------------------------------------------------- _dump_final_exchange


def test_dump_final_exchange_disabled_does_not_write(tmp_path, monkeypatch):
    """When the env flag is off the helper must be a no-op."""
    from config import settings

    object.__setattr__(settings, "llm_dump_final_enabled", False)
    object.__setattr__(settings, "llm_dump_final_dir", str(tmp_path))
    try:
        _dump_final_exchange("contract", "Договор подряда", "sys", "human", "response")
        assert list(tmp_path.iterdir()) == []
    finally:
        object.__setattr__(settings, "llm_dump_final_enabled", False)


def test_dump_final_exchange_writes_sanitized_filename(tmp_path, monkeypatch):
    from config import settings

    object.__setattr__(settings, "llm_dump_final_enabled", True)
    object.__setattr__(settings, "llm_dump_final_dir", str(tmp_path))
    object.__setattr__(settings, "llm_model", "qwen2.5:14b")
    try:
        _dump_final_exchange("contract", "Договор купли-продажи", "SYS", "HUM", "RESP")
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        # Filename: model and deal-type characters not in [\w.-] replaced with _
        assert files[0].name.endswith("_contract.txt")
        # Colon in model name and Cyrillic deal-type both survived (only
        # special chars like `:` get sanitized).
        assert "qwen2.5_14b" in files[0].name
        # Body contains all three pieces.
        body = files[0].read_text(encoding="utf-8")
        assert "SYS" in body
        assert "HUM" in body
        assert "RESP" in body
    finally:
        object.__setattr__(settings, "llm_dump_final_enabled", False)


def test_dump_final_exchange_overwrites_on_repeat_calls(tmp_path):
    from config import settings

    object.__setattr__(settings, "llm_dump_final_enabled", True)
    object.__setattr__(settings, "llm_dump_final_dir", str(tmp_path))
    object.__setattr__(settings, "llm_model", "m")
    try:
        _dump_final_exchange("contract", "deal", "s1", "h1", "r1")
        _dump_final_exchange("contract", "deal", "s2", "h2", "r2")
        files = list(tmp_path.iterdir())
        # Same (model, deal_type, kind) → single file, latest content.
        assert len(files) == 1
        body = files[0].read_text(encoding="utf-8")
        assert "r2" in body
        assert "r1" not in body
    finally:
        object.__setattr__(settings, "llm_dump_final_enabled", False)


def test_dump_final_exchange_falls_back_for_missing_deal_type(tmp_path):
    from config import settings

    object.__setattr__(settings, "llm_dump_final_enabled", True)
    object.__setattr__(settings, "llm_dump_final_dir", str(tmp_path))
    object.__setattr__(settings, "llm_model", "m")
    try:
        _dump_final_exchange("recomendations", None, "s", "h", "r")
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        # Fallback "unknown" segment.
        assert "unknown" in files[0].name
    finally:
        object.__setattr__(settings, "llm_dump_final_enabled", False)


# ---------------------------------------------------------------- classify_yesno


def test_classify_yesno_returns_yes(monkeypatch):
    monkeypatch.setattr(nodes, "_invoke_messages", lambda *a, **kw: '{"decision": "yes", "reason": "explicit"}')
    assert classify_yesno("вопрос?", "да, конечно") == "yes"


def test_classify_yesno_returns_no(monkeypatch):
    monkeypatch.setattr(nodes, "_invoke_messages", lambda *a, **kw: '{"decision": "no"}')
    assert classify_yesno("вопрос?", "нет") == "no"


def test_classify_yesno_falls_back_to_unclear_on_invalid_decision(monkeypatch):
    """LLM emits an unexpected value — must be treated as 'unclear' so the
    caller re-asks instead of guessing a side-effecting branch."""
    monkeypatch.setattr(nodes, "_invoke_messages", lambda *a, **kw: '{"decision": "maybe"}')
    assert classify_yesno("q", "a") == "unclear"


def test_classify_yesno_falls_back_to_unclear_on_llm_failure(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(nodes, "_invoke_messages", _raise)
    assert classify_yesno("q", "a") == "unclear"
