from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from functools import lru_cache

import httpx
from config import settings
from langchain_ollama import ChatOllama
from pymorphy3 import MorphAnalyzer

from rag.source_registry import (
    get_known_sources,
    get_source_registry,
    resolve_same_doc_reference,
    resolve_source_by_alias,
    source_by_name,
)

ARTICLE_RE = re.compile(r"(?m)^Статья\s+(\d+(?:\.\d+)?)")
CHAPTER_RE = re.compile(r"(?m)^Глава\s+[^\n]+")
ARTICLE_INLINE_RE = re.compile(r"(?i)ст(?:атья|\.|атьи|атьей|атье)?\s*(\d+(?:\.\d+)?)")
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")
WINDOW_SPLIT_RE = re.compile(r"\n\s*\n|(?<=[\.;:])\s+")
SAME_DOC_ARTICLE_RE = re.compile(
    r"(?i)(?:настоящ(?:его|им|ем|ий|им)?\s+кодекс(?:а|ом|е)?|настоящ(?:его|им|ем|ий|им)?\s+закона)"
    r"[^\n\.\(\)]{0,120}?ст(?:атья|\.|атьи|атьей|атье)?\s*(\d+(?:\.\d+)?)"
)
ARTICLE_THEN_SAME_DOC_RE = re.compile(
    r"(?i)ст(?:атья|\.|атьи|атьей|атье)?\s*(\d+(?:\.\d+)?)"
    r"[^\n\.\(\)]{0,120}?(?:настоящ(?:его|им|ем|ий|им)?\s+кодекс(?:а|ом|е)?|настоящ(?:его|им|ем|ий|им)?\s+закона)"
)
ARTICLE_WITH_POINT_RE = re.compile(
    r"(?i)(?:пункт(?:а|е|ом)?|п\.)\s*\d+(?:\.\d+)?\s+"
    r"ст(?:атья|\.|атьи|атьей|атье)?\s*(\d+(?:\.\d+)?)"
)
MAX_CHUNK_CHARS = 3000
OVERLAP_PARAGRAPHS = 1
EXTERNAL_ARTICLE_WINDOW = 120
EXTRACTION_MODES = {"llm", "morphology", "hybrid"}
LOGGER = logging.getLogger("rag.chunker")


def _reference_prompt(current_source: str | None = None) -> str:
    allowed = _reference_allowed_sources(current_source)
    allowed_block = (
        "\n- " + "\n- ".join(allowed)
        if allowed
        else " (нет проиндексированных источников; при необходимости упоминай "
        "канонические имена сам — постпроцессинг отфильтрует невалидные)"
    )
    return f"""Извлеки из текста все ссылки на статьи нормативных источников.
Верни только JSON-массив объектов {{"source": "...", "article": "..."}}.
Допустимые source:{allowed_block}

Правила:
- Не считай заголовок текущей статьи ссылкой.
- Извлекай только ссылки с конкретным номером статьи; упоминания закона без статьи пропускай.
- Внутридокументные ссылки ("настоящего Кодекса/закона", "пункт 2 статьи N", "(статья N)", "со статьей N") относи к текущему источнику.
- Для ГК всегда используй source "ГК РФ".
- article только номером, например "454" или "19.1".
- Если ссылок нет, верни [].
"""


def _reference_allowed_sources(current_source: str | None = None) -> tuple[str, ...]:
    """Sources LLM can target.

    Includes ALL groups (not just SECONDARY) — typical legal docs cite each
    other across groups. Plus the current source being chunked, so the very
    first upload can still produce self-references like "статьей 432
    настоящего Кодекса" when no other sources are indexed yet.
    """
    sources: list[str] = []
    seen: set[str] = set()
    for item in get_source_registry():
        if item.source in seen:
            continue
        seen.add(item.source)
        sources.append(item.source)
    if current_source and current_source not in seen:
        sources.append(current_source)
    return tuple(sources)


def _reference_output_schema(current_source: str | None = None) -> dict:
    """JSON schema for Ollama structured output.

    When the allowed-source list is empty (fresh install, nothing indexed
    yet, no current_source either), drop the enum constraint so Ollama
    doesn't choke on `enum: []` — the post-processing step in
    `_coerce_llm_item` re-canonicalizes whatever the LLM emits and drops
    the rest.
    """
    allowed = list(_reference_allowed_sources(current_source))
    source_schema: dict = {"type": "string"}
    if allowed:
        source_schema["enum"] = allowed
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "source": source_schema,
                "article": {"type": "string", "pattern": r"^\d+(?:\.\d+)?$"},
            },
            "required": ["source", "article"],
            "additionalProperties": False,
        },
    }


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_chapter(text: str, position: int) -> str:
    chapter = ""
    for match in CHAPTER_RE.finditer(text):
        if match.start() > position:
            break
        chapter = match.group(0).strip()
    return chapter


def _extract_article_number(article_header: str) -> str | None:
    match = ARTICLE_RE.match(article_header.strip())
    if not match:
        return None
    return match.group(1)


def _split_article_header(section_text: str, fallback_number: str) -> tuple[str, str]:
    section = section_text.strip()
    if not section:
        return f"Статья {fallback_number}", ""

    lines = [line.strip() for line in section.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)

    if not lines:
        return f"Статья {fallback_number}", ""

    header = lines[0]
    body = "\n".join(line for line in lines[1:]).strip()
    return header, body


def _iter_article_sections(text: str) -> Iterable[tuple[str, str, str]]:
    matches = list(ARTICLE_RE.finditer(text))
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        article, body = _split_article_header(section, match.group(1))
        chapter = _find_chapter(text, start)
        yield article, chapter, body


def _split_long_section(article: str, chapter: str, body_text: str) -> list[dict]:
    body_text = body_text.strip()
    if not body_text:
        return [{"article": article, "chapter": chapter, "text": ""}]

    if len(body_text) <= MAX_CHUNK_CHARS:
        return [{"article": article, "chapter": chapter, "text": body_text}]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body_text) if part.strip()]
    chunks: list[dict] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 2 > MAX_CHUNK_CHARS:
            chunk_text = "\n\n".join(current).strip()
            chunks.append({"article": article, "chapter": chapter, "text": chunk_text})
            overlap = current[-OVERLAP_PARAGRAPHS:] if OVERLAP_PARAGRAPHS else []
            current = overlap.copy()
            current_len = sum(len(item) for item in current)

        current.append(paragraph)
        current_len += len(paragraph) + 2

    if current:
        chunk_text = "\n\n".join(current).strip()
        chunks.append({"article": article, "chapter": chapter, "text": chunk_text})

    return chunks


def _normalize_reference(source: str | None, article_number: str | None) -> str | None:
    if not source or not article_number:
        return None
    return f"{source} ст.{article_number}"


def _append_unique(target: list[str], value: str | None) -> None:
    if value and value not in target:
        target.append(value)


def _validate_extraction_mode(extraction_mode: str) -> str:
    if extraction_mode not in EXTRACTION_MODES:
        raise ValueError(f"Unsupported extraction mode: {extraction_mode}")
    return extraction_mode


@lru_cache(maxsize=1)
def _morph_analyzer() -> MorphAnalyzer:
    return MorphAnalyzer()


@lru_cache(maxsize=16384)
def _lemma(word: str) -> str:
    return _morph_analyzer().parse(word)[0].normal_form


@lru_cache(maxsize=16384)
def _normalize_lemma_text(text: str) -> str:
    words = [match.group(0).lower() for match in WORD_RE.finditer(text)]
    if not words:
        return ""
    return " ".join(_lemma(word) for word in words)


@lru_cache(maxsize=1)
def _normalized_aliases() -> dict[str, tuple[str, ...]]:
    aliases: dict[str, set[str]] = {}
    for item in get_source_registry():
        current = aliases.setdefault(item.source, set())
        current.update(_normalize_lemma_text(alias) for alias in item.aliases)
        current.update(_normalize_lemma_text(alias) for alias in item.family_aliases)
    return {key: tuple(sorted(value)) for key, value in aliases.items()}


@lru_cache(maxsize=1)
def _normalized_same_doc_markers() -> dict[str, tuple[str, ...]]:
    return {
        source: tuple(_normalize_lemma_text(marker) for marker in info.same_doc_markers)
        for source, info in source_by_name().items()
    }


def _reference_windows(text: str) -> list[str]:
    parts = [part.strip() for part in WINDOW_SPLIT_RE.split(text) if part.strip()]
    return parts or [text.strip()]


def _extract_same_doc_references(text: str, current_source: str | None) -> list[str]:
    if not current_source:
        return []

    normalized = text.lower()
    references: list[str] = []

    for number in SAME_DOC_ARTICLE_RE.findall(text):
        resolved_source = resolve_same_doc_reference(current_source, number)
        _append_unique(references, _normalize_reference(resolved_source, number))
    for number in ARTICLE_THEN_SAME_DOC_RE.findall(text):
        resolved_source = resolve_same_doc_reference(current_source, number)
        _append_unique(references, _normalize_reference(resolved_source, number))
    is_code_source = "кодекс" in current_source.lower() or "гк рф" in current_source.lower()
    if is_code_source:
        for number in ARTICLE_WITH_POINT_RE.findall(text):
            resolved_source = resolve_same_doc_reference(current_source, number)
            _append_unique(references, _normalize_reference(resolved_source, number))

    if current_source in source_by_name():
        markers = source_by_name()[current_source].same_doc_markers
    else:
        # First-upload fallback: registry empty, derive markers from the name string.
        from rag.source_registry import _same_doc_markers

        markers = _same_doc_markers(current_source)
    for marker in markers:
        if marker not in normalized:
            continue
        for number in ARTICLE_INLINE_RE.findall(text):
            resolved_source = resolve_same_doc_reference(current_source, number)
            _append_unique(references, _normalize_reference(resolved_source, number))

    return references


def _iter_all_aliases() -> tuple[str, ...]:
    aliases = set()
    for item in get_source_registry():
        aliases.update(item.aliases)
        aliases.update(item.family_aliases)
    return tuple(sorted(aliases, key=len, reverse=True))


def _extract_external_references(text: str, current_source: str | None = None) -> list[str]:
    normalized = text.lower()
    references: list[str] = []

    for alias in _iter_all_aliases():
        start = 0
        while True:
            index = normalized.find(alias, start)
            if index == -1:
                break
            window = text[index : index + len(alias) + EXTERNAL_ARTICLE_WINDOW]
            article_match = ARTICLE_INLINE_RE.search(window)
            article_number = article_match.group(1) if article_match else None
            resolved_source = resolve_source_by_alias(alias, article_number, current_source=current_source)
            _append_unique(references, _normalize_reference(resolved_source, article_number))
            start = index + len(alias)

    return references


def _extract_same_doc_references_morphology(text: str, current_source: str | None) -> list[str]:
    if not current_source:
        return []

    references: list[str] = []
    normalized_markers = _normalized_same_doc_markers().get(current_source)
    if not normalized_markers:
        # First-upload fallback so self-references like "настоящего Кодекса" still match.
        from rag.source_registry import _same_doc_markers

        raw_markers = _same_doc_markers(current_source)
        normalized_markers = tuple(_normalize_lemma_text(m) for m in raw_markers if m)
    if not normalized_markers:
        return references

    for window in _reference_windows(text):
        normalized_window = _normalize_lemma_text(window)
        if not normalized_window:
            continue
        if not any(marker in normalized_window for marker in normalized_markers):
            continue
        for number in ARTICLE_INLINE_RE.findall(window):
            resolved_source = resolve_same_doc_reference(current_source, number)
            _append_unique(references, _normalize_reference(resolved_source, number))

    return references


def _extract_external_references_morphology(text: str, current_source: str | None = None) -> list[str]:
    references: list[str] = []
    normalized_aliases = _normalized_aliases()

    for window in _reference_windows(text):
        normalized_window = _normalize_lemma_text(window)
        if not normalized_window:
            continue

        for source, aliases in normalized_aliases.items():
            for normalized_alias in aliases:
                if normalized_alias not in normalized_window:
                    continue
                article_match = ARTICLE_INLINE_RE.search(window)
                article_number = article_match.group(1) if article_match else None
                resolved_source = (
                    resolve_source_by_alias(source, article_number, current_source=current_source) or source
                )
                _append_unique(references, _normalize_reference(resolved_source, article_number))
                break

    return references


def _rule_based_references(text: str, current_source: str | None = None) -> list[str]:
    references: list[str] = []
    for value in _extract_same_doc_references(text, current_source):
        _append_unique(references, value)
    for value in _extract_external_references(text, current_source=current_source):
        _append_unique(references, value)
    return references


def _morphology_references(text: str, current_source: str | None = None) -> list[str]:
    references: list[str] = []
    for value in _extract_same_doc_references_morphology(text, current_source):
        _append_unique(references, value)
    for value in _extract_external_references_morphology(text, current_source=current_source):
        _append_unique(references, value)
    return references


@lru_cache(maxsize=1)
def _reference_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.reference_llm_model,
        base_url=settings.reference_llm_base_url,
        temperature=0.1,
        num_ctx=2048,
        num_predict=512,
        top_p=0.3,
        top_k=20,
        reasoning=settings.reference_llm_reasoning,
    )


def _coerce_llm_item(item: object, current_source: str | None) -> str | None:
    if not isinstance(item, dict):
        return None

    raw_source = item.get("source")
    article = item.get("article")
    if not isinstance(raw_source, str):
        return None

    raw_source = raw_source.strip()
    article_number = None
    if isinstance(article, str):
        match = re.search(r"(\d+(?:\.\d+)?)", article)
        article_number = match.group(1) if match else None
    elif isinstance(article, (int, float)):
        article_number = str(article)

    resolved_source = resolve_source_by_alias(raw_source, article_number, current_source=current_source)
    return _normalize_reference(resolved_source, article_number)


def _extract_json_array(content: str) -> list[object]:
    content = content.strip()
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("JSON array not found in LLM response")
    parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("LLM response is not a JSON array")
    return parsed


def _reference_message(chunk_text: str, current_source: str | None) -> str:
    return (
        f"{_reference_prompt(current_source)}\n\n"
        f"Текущий источник: {current_source or 'неизвестно'}\n"
        f"Текст:\n{chunk_text}"
    )


def _ollama_chat_url() -> str:
    return f"{settings.reference_llm_base_url.rstrip('/')}/api/chat"


def _structured_reference_items(chunk_text: str, current_source: str | None) -> list[object]:
    payload = {
        "model": settings.reference_llm_model,
        "messages": [{"role": "user", "content": _reference_message(chunk_text, current_source)}],
        "stream": False,
        "think": settings.reference_llm_reasoning,
        "format": _reference_output_schema(current_source),
        "options": {
            "num_ctx": 2048,
            "num_predict": 512,
            "temperature": 0,
            "top_p": 0.1,
            "top_k": 1,
        },
    }
    response = httpx.post(_ollama_chat_url(), json=payload, timeout=120.0)
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return _extract_json_array(content if isinstance(content, str) else str(content))


def _fallback_reference_items(chunk_text: str, current_source: str | None) -> list[object]:
    llm = _reference_llm()
    response = llm.invoke(_reference_message(chunk_text, current_source))
    content = response.content if isinstance(response.content, str) else str(response.content)
    return _extract_json_array(content)


@lru_cache(maxsize=2048)
def _llm_extract_cached(chunk_text: str, current_source: str | None) -> tuple[str, ...]:
    LOGGER.info(
        "LLM extracting references: model=%s source=%s text_len=%s",
        settings.reference_llm_model,
        current_source,
        len(chunk_text),
    )
    try:
        data = _structured_reference_items(chunk_text, current_source)
    except Exception as exc:
        LOGGER.warning("Structured Ollama reference extraction failed, falling back to ChatOllama: %s", exc)
        data = _fallback_reference_items(chunk_text, current_source)

    references: list[str] = []
    for item in data:
        _append_unique(references, _coerce_llm_item(item, current_source=current_source))

    LOGGER.info("LLM extracted %s normalized references: %s", len(references), references)
    return tuple(references)


def _filter_self_references(
    references: list[str], current_source: str | None, current_article: str | None
) -> list[str]:
    if not current_source:
        return references

    current_number = _extract_article_number(current_article or "")
    filtered: list[str] = []
    for reference in references:
        if reference == current_source:
            continue
        if current_number and reference == f"{current_source} ст.{current_number}":
            continue
        filtered.append(reference)
    return filtered


def extract_references(
    text: str,
    current_source: str | None = None,
    current_article: str | None = None,
    extraction_mode: str = "llm",
) -> list[str]:
    extraction_mode = _validate_extraction_mode(extraction_mode)

    references: list[str] = []
    for value in _rule_based_references(text, current_source=current_source):
        _append_unique(references, value)

    if extraction_mode in {"morphology", "hybrid"}:
        LOGGER.info("Morphology extracting references: source=%s text_len=%s", current_source, len(text))
        for value in _morphology_references(text, current_source=current_source):
            _append_unique(references, value)

    if extraction_mode in {"llm", "hybrid"}:
        try:
            for value in _llm_extract_cached(text, current_source):
                _append_unique(references, value)
        except Exception as exc:
            LOGGER.warning("LLM reference extraction failed: %s", exc)

    return _filter_self_references(references, current_source=current_source, current_article=current_article)


def parse_reference(reference: str) -> tuple[str | None, str | None]:
    value = reference.strip()
    match = re.fullmatch(r"(.+?)\s+ст\.(\d+(?:\.\d+)?)", value)
    if match:
        return match.group(1), match.group(2)
    if value in get_known_sources():
        return value, None
    return None, None


def chunk_document(
    text: str,
    source: str,
    is_general: bool,
    limit: int | None = None,
    extraction_mode: str = "llm",
) -> list[dict]:
    extraction_mode = _validate_extraction_mode(extraction_mode)
    normalized_text = _normalize_whitespace(text)
    chunks: list[dict] = []
    chunk_index = 0

    for article, chapter, body_text in _iter_article_sections(normalized_text):
        for part in _split_long_section(article=article, chapter=chapter, body_text=body_text):
            references = extract_references(
                part["text"],
                current_source=source,
                current_article=part["article"],
                extraction_mode=extraction_mode,
            )
            article_number = _extract_article_number(part["article"])
            chunk_id = f"{source}::{article_number or part['article']}::{chunk_index}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "text": part["text"],
                    "source": source,
                    "article": part["article"],
                    "article_number": article_number,
                    "chapter": part["chapter"],
                    "is_general": is_general,
                    "references": references,
                }
            )
            chunk_index += 1
            if limit is not None and len(chunks) >= limit:
                return chunks

    return chunks
