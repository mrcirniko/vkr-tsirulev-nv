"""Tests for rag.chunker pure functions and reference-extraction regexes.

The chunker is the gatekeeper between raw НПА text and Qdrant — its regex
rules decide what cross-references RAG can resolve. We focus on:
  * the four reference regexes (SAME_DOC_ARTICLE, ARTICLE_THEN_SAME_DOC,
    ARTICLE_WITH_POINT, ARTICLE_INLINE)
  * article splitting / chapter binding / chunk size limits
  * normalization, parsing, validation helpers

LLM-driven extraction is bypassed via monkeypatch so tests run offline.
"""

from __future__ import annotations

import json

import pytest

from rag import chunker
from rag.chunker import (
    ARTICLE_INLINE_RE,
    ARTICLE_RE,
    ARTICLE_THEN_SAME_DOC_RE,
    ARTICLE_WITH_POINT_RE,
    CHAPTER_RE,
    MAX_CHUNK_CHARS,
    SAME_DOC_ARTICLE_RE,
    _extract_article_number,
    _extract_json_array,
    _filter_self_references,
    _find_chapter,
    _iter_article_sections,
    _normalize_reference,
    _normalize_whitespace,
    _split_article_header,
    _split_long_section,
    _validate_extraction_mode,
    chunk_document,
    parse_reference,
)


# ---------------------------------------------------------------- regex: ARTICLE_INLINE_RE


@pytest.mark.parametrize(
    "text,expected",
    [
        ("статья 432", ["432"]),
        ("ст. 19.1", ["19.1"]),
        ("статьей 421 и статьи 422", ["421", "422"]),
        ("СТАТЬЯ 50", ["50"]),  # case-insensitive
        ("текст без ссылок", []),
        ("ст 100", ["100"]),  # bare "ст" with space
    ],
)
def test_article_inline_regex(text, expected):
    assert ARTICLE_INLINE_RE.findall(text) == expected


# ---------------------------------------------------------------- regex: SAME_DOC_ARTICLE_RE


def test_same_doc_re_matches_kodeks_then_article():
    text = "в соответствии с настоящим Кодексом, статьей 432, стороны..."
    assert SAME_DOC_ARTICLE_RE.findall(text) == ["432"]


def test_same_doc_re_matches_zakon_then_article():
    """The regex covers participle endings 'его|им|ем|ий' before 'закона' —
    'настоящему' is intentionally NOT in this set (real legal text uses
    'настоящего закона' / 'настоящим законом' phrasings)."""
    text = "согласно настоящего закона статья 5"
    assert SAME_DOC_ARTICLE_RE.findall(text) == ["5"]


def test_same_doc_re_does_not_match_across_sentence_boundary():
    """The regex window is bounded by sentence end (.;:); a period between
    "Кодекса" and "статья" must break the match — otherwise we'd mis-attribute
    references that belong to a different statute."""
    text = "настоящего Кодекса. Статья 100 другого закона."
    assert SAME_DOC_ARTICLE_RE.findall(text) == []


# ---------------------------------------------------------------- regex: ARTICLE_THEN_SAME_DOC_RE


def test_article_then_same_doc_re_matches_reverse_order():
    text = "статьей 421 настоящего Кодекса"
    assert ARTICLE_THEN_SAME_DOC_RE.findall(text) == ["421"]


def test_article_then_same_doc_re_with_zakon():
    text = "статья 5 настоящего закона"
    assert ARTICLE_THEN_SAME_DOC_RE.findall(text) == ["5"]


# ---------------------------------------------------------------- regex: ARTICLE_WITH_POINT_RE


@pytest.mark.parametrize(
    "text,expected",
    [
        ("пункт 2 статьи 432", ["432"]),
        ("п. 1 ст. 421", ["421"]),
        ("пункта 3 статьей 19.1", ["19.1"]),
    ],
)
def test_article_with_point_regex(text, expected):
    assert ARTICLE_WITH_POINT_RE.findall(text) == expected


def test_article_with_point_requires_point_prefix():
    """Without "пункт N" prefix the regex must NOT match — that's the whole
    purpose of this rule (point-then-article is a strong code-internal cue)."""
    assert ARTICLE_WITH_POINT_RE.findall("статьи 432") == []


# ---------------------------------------------------------------- regex: ARTICLE_RE / CHAPTER_RE


def test_article_re_matches_only_at_line_start():
    text = "Статья 1\nтекст\nСтатья 2.5\nдругой текст\nст. 3 в середине строки"
    matches = [m.group(1) for m in ARTICLE_RE.finditer(text)]
    assert matches == ["1", "2.5"]


def test_chapter_re_matches_chapter_heading():
    text = "Глава 28. Заключение договора\nтекст"
    match = CHAPTER_RE.search(text)
    assert match is not None
    assert match.group(0).strip() == "Глава 28. Заключение договора"


# ---------------------------------------------------------------- _normalize_whitespace


def test_normalize_whitespace_collapses_runs_and_strips():
    text = "  abc\xa0\xa0def\r\nghi   jkl\n\n\n\nmno  "
    result = _normalize_whitespace(text)
    # \xa0 → space, \r dropped, runs of spaces collapsed, 3+ newlines → 2
    assert result == "abc def\nghi jkl\n\nmno"


def test_normalize_whitespace_preserves_double_newline_paragraph_breaks():
    text = "para1\n\npara2"
    assert _normalize_whitespace(text) == "para1\n\npara2"


# ---------------------------------------------------------------- _find_chapter / _extract_article_number


def test_find_chapter_returns_most_recent_chapter_before_position():
    text = "Глава 1. Общие\nтекст\nГлава 2. Договор\nСтатья 5\nещё"
    article_pos = text.index("Статья 5")
    assert _find_chapter(text, article_pos).startswith("Глава 2")


def test_find_chapter_returns_empty_string_when_no_chapter_yet():
    text = "Статья 1\nтекст\nГлава 1. Common"
    assert _find_chapter(text, 0) == ""


def test_extract_article_number_handles_decimal():
    assert _extract_article_number("Статья 19.1") == "19.1"
    assert _extract_article_number("Статья 432") == "432"
    assert _extract_article_number("not an article header") is None


# ---------------------------------------------------------------- _split_article_header


def test_split_article_header_separates_first_line_as_title():
    section = "Статья 432. Основные положения о заключении договора\nтело статьи..."
    header, body = _split_article_header(section, "432")
    assert header.startswith("Статья 432")
    assert body == "тело статьи..."


def test_split_article_header_returns_fallback_for_empty_input():
    header, body = _split_article_header("", "10")
    assert header == "Статья 10"
    assert body == ""


# ---------------------------------------------------------------- _iter_article_sections


def test_iter_article_sections_yields_each_article_with_chapter():
    text = (
        "Глава 1. Общие положения\n"
        "Статья 1. Предмет\n"
        "тело первой\n\n"
        "Статья 2. Стороны\n"
        "тело второй"
    )
    sections = list(_iter_article_sections(text))
    assert len(sections) == 2
    art1, ch1, body1 = sections[0]
    art2, ch2, body2 = sections[1]
    assert art1.startswith("Статья 1")
    assert art2.startswith("Статья 2")
    assert ch1.startswith("Глава 1")
    assert ch2.startswith("Глава 1")
    assert "тело первой" in body1
    assert "тело второй" in body2


# ---------------------------------------------------------------- _split_long_section


def test_split_long_section_returns_single_chunk_when_under_limit():
    chunks = _split_long_section("Статья 1", "Глава 1", "короткое тело")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "короткое тело"


def test_split_long_section_handles_empty_body():
    chunks = _split_long_section("Статья 1", "Глава 1", "")
    assert len(chunks) == 1
    assert chunks[0]["text"] == ""


def test_split_long_section_splits_oversized_body_with_overlap():
    para = "Очень длинный параграф. " * 80  # ~ 1840 chars per
    body = "\n\n".join([para, para])  # ~ 3680 chars total
    chunks = _split_long_section("Статья 1", "Глава 1", body)
    assert len(chunks) >= 2
    # Each chunk respects MAX_CHUNK_CHARS (allowing overlap to slightly exceed
    # the target — the implementation flushes BEFORE adding a paragraph that
    # would overflow, then the overlap paragraph is included in the next).
    for chunk in chunks:
        assert len(chunk["text"]) <= MAX_CHUNK_CHARS + len(para)


# ---------------------------------------------------------------- _normalize_reference


def test_normalize_reference_returns_canonical_form():
    assert _normalize_reference("ГК РФ", "432") == "ГК РФ ст.432"


@pytest.mark.parametrize("source,article", [(None, "1"), ("ГК РФ", None), ("", "1"), ("ГК РФ", "")])
def test_normalize_reference_returns_none_on_missing_parts(source, article):
    assert _normalize_reference(source, article) is None


# ---------------------------------------------------------------- _filter_self_references


def test_filter_self_references_drops_current_article_reference():
    refs = ["ГК РФ ст.432", "ФЗ Об ипотеке ст.5", "ГК РФ"]
    filtered = _filter_self_references(refs, current_source="ГК РФ", current_article="Статья 432")
    # ст.432 is self → drop. Bare "ГК РФ" without article is also dropped
    # (implementation skips reference == current_source).
    assert "ГК РФ ст.432" not in filtered
    assert "ГК РФ" not in filtered
    assert "ФЗ Об ипотеке ст.5" in filtered


def test_filter_self_references_keeps_other_articles_in_same_source():
    refs = ["ГК РФ ст.432", "ГК РФ ст.421"]
    filtered = _filter_self_references(refs, current_source="ГК РФ", current_article="Статья 421")
    assert filtered == ["ГК РФ ст.432"]


def test_filter_self_references_passes_through_when_no_current_source():
    refs = ["ГК РФ ст.432"]
    assert _filter_self_references(refs, current_source=None, current_article="Статья 1") == refs


# ---------------------------------------------------------------- parse_reference


def test_parse_reference_round_trip():
    assert parse_reference("ГК РФ ст.432") == ("ГК РФ", "432")
    assert parse_reference("ФЗ Об ипотеке ст.19.1") == ("ФЗ Об ипотеке", "19.1")


def test_parse_reference_returns_none_for_unrecognized_format(monkeypatch):
    """`parse_reference` falls back to checking known sources for a bare name.
    Without a known-source match, it must return (None, None)."""
    monkeypatch.setattr(chunker, "get_known_sources", lambda: ())
    assert parse_reference("just some string") == (None, None)


# ---------------------------------------------------------------- _extract_json_array


def test_extract_json_array_handles_clean_payload():
    assert _extract_json_array('[{"source":"ГК РФ","article":"432"}]') == [{"source": "ГК РФ", "article": "432"}]


def test_extract_json_array_strips_surrounding_text():
    """LLM responses sometimes wrap JSON in prose like "Here is the result:".
    The helper must locate the array by `[ ... ]` boundaries."""
    content = 'Some preamble. [{"a":1}] trailing words.'
    assert _extract_json_array(content) == [{"a": 1}]


def test_extract_json_array_raises_when_no_array():
    with pytest.raises(ValueError):
        _extract_json_array("not json at all")


def test_extract_json_array_raises_on_malformed_inner_json():
    """If the substring between '[' and ']' is not valid JSON, the helper
    surfaces the parse error rather than returning partial garbage."""
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _extract_json_array("preamble [not, valid, json}] trailing")


# ---------------------------------------------------------------- _validate_extraction_mode


@pytest.mark.parametrize("mode", ["llm", "morphology", "hybrid"])
def test_validate_extraction_mode_accepts_known(mode):
    assert _validate_extraction_mode(mode) == mode


def test_validate_extraction_mode_rejects_unknown():
    with pytest.raises(ValueError):
        _validate_extraction_mode("regex_only")


# ---------------------------------------------------------------- chunk_document smoke


def test_chunk_document_produces_one_chunk_per_article(monkeypatch):
    """End-to-end smoke: feed two articles, get two chunks with stable IDs.
    `extract_references` is stubbed so we don't hit Ollama / morphology /
    the source registry."""
    monkeypatch.setattr(chunker, "extract_references", lambda *a, **kw: [])

    text = (
        "Глава 1. Общие положения\n"
        "Статья 1. Первая статья\n"
        "тело первой\n\n"
        "Статья 2. Вторая статья\n"
        "тело второй"
    )
    chunks = chunk_document(text, source="Тестовый закон", is_general=False, extraction_mode="morphology")
    assert len(chunks) == 2
    assert chunks[0]["article_number"] == "1"
    assert chunks[1]["article_number"] == "2"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
    assert chunks[0]["chunk_id"] == "Тестовый закон::1::0"
    assert chunks[1]["chunk_id"] == "Тестовый закон::2::1"
    assert chunks[0]["source"] == "Тестовый закон"
    assert chunks[0]["chapter"].startswith("Глава 1")


def test_chunk_document_respects_limit(monkeypatch):
    monkeypatch.setattr(chunker, "extract_references", lambda *a, **kw: [])
    text = "\n".join(f"Статья {i}\nтело {i}" for i in range(1, 6))
    chunks = chunk_document(text, source="X", is_general=False, limit=2, extraction_mode="morphology")
    assert len(chunks) == 2
