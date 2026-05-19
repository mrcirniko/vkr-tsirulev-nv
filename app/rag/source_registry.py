from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PART_RE = re.compile(r"\bчаст[ьи]\s+(?:\d+|[ivxlcdm]+|первая|вторая|третья|четвертая|пятая)\b", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")

LOGGER = logging.getLogger("rag.source_registry")


@dataclass(frozen=True)
class SourceInfo:
    source: str
    group: str
    path: str  # legacy field, always "" now (source data lives in S3)
    aliases: tuple[str, ...]
    family_aliases: tuple[str, ...]
    article_min: float | None
    article_max: float | None
    same_doc_markers: tuple[str, ...]


GENERAL_GROUP = "general"
PRIMAL_GROUP = "primal"
SECONDARY_GROUP = "secondary"
SPECIFIC_GROUPS = {PRIMAL_GROUP, SECONDARY_GROUP}


def _normalize_spaces(value: str) -> str:
    return SPACE_RE.sub(" ", value.replace("_", " ").strip())


def canonical_source_name(filename: str) -> str:
    return _normalize_spaces(Path(filename).stem)


def _clean_alias(value: str) -> str:
    value = _normalize_spaces(value).lower()
    value = re.sub(r"[\(\)\[\]\{\}\.,:;\"'«»]", " ", value)
    return _normalize_spaces(value)


def _family_name(source: str) -> str:
    return _normalize_spaces(PART_RE.sub("", source)).strip()


def _generic_aliases(source: str) -> tuple[str, ...]:
    variants = set()
    base = _clean_alias(source)
    if base:
        variants.add(base)
    if "российской федерации" in base:
        variants.add(_normalize_spaces(base.replace("российской федерации", "рф")))
    if "российская федерация" in base:
        variants.add(_normalize_spaces(base.replace("российская федерация", "рф")))
    if base.startswith("фз "):
        variants.add(base[3:].strip())
    if base.startswith("федеральный закон "):
        variants.add(base[len("федеральный закон ") :].strip())
    if " кодекс российской федерации" in base:
        variants.add(base.replace(" кодекс российской федерации", " кодекс рф"))
    if " гражданский кодекс рф" in base:
        variants.add(base.replace("гражданский кодекс рф", "гк рф"))
    if " гражданский кодекс российской федерации" in base:
        variants.add(base.replace("гражданский кодекс российской федерации", "гк рф"))
    if base.startswith("гражданский кодекс рф"):
        variants.add(base.replace("гражданский кодекс рф", "гк рф", 1))
    return tuple(sorted(alias for alias in variants if alias))


def _family_aliases(source: str) -> tuple[str, ...]:
    family = _family_name(source)
    return _generic_aliases(family)


def _same_doc_markers(source: str) -> tuple[str, ...]:
    lowered = _clean_alias(source)
    markers = []
    if "кодекс" in lowered:
        markers.extend(["настоящего кодекса", "настоящим кодексом", "настоящем кодексе"])
    if "закон" in lowered or lowered.startswith("фз "):
        markers.extend(["настоящего закона", "настоящим законом", "настоящем законе"])
    return tuple(dict.fromkeys(markers))


def _group_from_collection(collection: str | None) -> str | None:
    """Reverse-map a Qdrant collection name -> the abstract group label.

    Settings can rename collections, so we ask config rather than hard-coding.
    Returns None for unknown collections (e.g. an old indexed run in a
    collection name that no longer exists in settings).
    """
    if not collection:
        return None
    try:
        from config import settings
    except Exception:
        return None
    if collection == settings.qdrant_collection_general:
        return GENERAL_GROUP
    if collection == settings.qdrant_collection_primal:
        return PRIMAL_GROUP
    if collection == settings.qdrant_collection_secondary:
        return SECONDARY_GROUP
    return None


@lru_cache(maxsize=1)
def get_source_registry() -> tuple[SourceInfo, ...]:
    """Build the registry from the npa_sources DB table.

    Only sources that have been indexed at least once are included — the
    `group` is derived from `last_indexed_collection`. Aliases / family /
    same-doc markers are computed from the source name string. Article
    ranges are intentionally dropped: they used to require reading raw text
    from disk (now lives in S3) and were only used to disambiguate
    multi-part codes — admins now type unique source names per part anyway.

    Returns an empty tuple if the DB is unreachable; callers (chunker,
    retriever) handle that gracefully — reference extraction simply skips
    morphology-based passes for unknown sources.
    """
    try:
        from sqlalchemy import select

        from db.models import NpaSource
        from db.session import SessionLocal
    except Exception:
        LOGGER.debug("DB stack unavailable; returning empty source registry")
        return ()

    registry: list[SourceInfo] = []
    seen: set[str] = set()
    try:
        with SessionLocal() as session:
            stmt = (
                select(NpaSource)
                .where(NpaSource.last_indexed_collection.is_not(None))
                .order_by(NpaSource.last_indexed_at.desc().nullslast())
            )
            for npa in session.scalars(stmt).all():
                source = (npa.source_name or "").strip()
                if not source or source in seen:
                    continue
                group = _group_from_collection(npa.last_indexed_collection)
                if group is None:
                    continue
                seen.add(source)
                registry.append(
                    SourceInfo(
                        source=source,
                        group=group,
                        path="",
                        aliases=_generic_aliases(source),
                        family_aliases=_family_aliases(source),
                        article_min=None,
                        article_max=None,
                        same_doc_markers=_same_doc_markers(source),
                    )
                )
    except Exception:
        LOGGER.exception("Failed to read npa_sources for registry; returning empty")
        return ()
    return tuple(registry)


def clear_source_registry_cache() -> None:
    """Drop the cached registry so the next call rebuilds from DB.

    Called by admin/crud.py after any change to npa_sources so chunker /
    retriever pick up new sources without an app restart. Also clears
    chunker's lemma-keyed alias/marker caches because they're derived
    from the registry.
    """
    get_source_registry.cache_clear()
    get_known_sources.cache_clear()
    source_by_name.cache_clear()
    grouped_family_sources.cache_clear()
    # Lazy import to avoid a circular dependency: chunker imports from this
    # module at top level.
    try:
        from rag import chunker

        chunker._normalized_aliases.cache_clear()
        chunker._normalized_same_doc_markers.cache_clear()
    except Exception:
        LOGGER.debug("Failed to clear chunker caches; non-fatal")


@lru_cache(maxsize=1)
def get_known_sources() -> tuple[str, ...]:
    return tuple(item.source for item in get_source_registry())


@lru_cache(maxsize=1)
def source_by_name() -> dict[str, SourceInfo]:
    return {item.source: item for item in get_source_registry()}


@lru_cache(maxsize=1)
def grouped_family_sources() -> dict[str, tuple[SourceInfo, ...]]:
    grouped: dict[str, list[SourceInfo]] = {}
    for item in get_source_registry():
        for family_alias in item.family_aliases:
            grouped.setdefault(family_alias, []).append(item)
    return {key: tuple(value) for key, value in grouped.items()}


def resolve_source_by_alias(
    alias: str, article_number: str | None = None, current_source: str | None = None
) -> str | None:
    alias_clean = _clean_alias(alias)
    if _is_civil_code_alias(alias_clean):
        return _secondary_source_by_alias("гк рф") or _source_by_clean_name("гк рф")

    exact_matches = [item for item in get_source_registry() if alias_clean in item.aliases]
    if len(exact_matches) == 1:
        return exact_matches[0].source
    if len(exact_matches) > 1:
        return _resolve_by_article_range(exact_matches, article_number, current_source=current_source)

    family_matches = grouped_family_sources().get(alias_clean, ())
    if family_matches:
        return _resolve_by_article_range(list(family_matches), article_number, current_source=current_source)
    return None


def _source_by_clean_name(clean_name: str) -> str | None:
    for item in get_source_registry():
        if _clean_alias(item.source) == clean_name:
            return item.source
    return None


def _secondary_source_by_alias(alias_clean: str) -> str | None:
    for item in get_source_registry():
        if item.group == SECONDARY_GROUP and alias_clean in item.aliases:
            return item.source
    return None


def _is_civil_code_alias(alias_clean: str) -> bool:
    return alias_clean == "гк рф" or alias_clean.startswith("гк рф ") or "гражданский кодекс" in alias_clean


def _resolve_by_article_range(
    candidates: list[SourceInfo], article_number: str | None, current_source: str | None = None
) -> str | None:
    if not candidates:
        return None
    if article_number is not None:
        try:
            article_value = float(article_number)
        except ValueError:
            article_value = None
        if article_value is not None:
            ranged = [
                item
                for item in candidates
                if item.article_min is not None
                and item.article_max is not None
                and item.article_min <= article_value <= item.article_max
            ]
            if len(ranged) == 1:
                return ranged[0].source
            if len(ranged) > 1 and current_source:
                for item in ranged:
                    if item.source == current_source:
                        return item.source
                return ranged[0].source
    if current_source:
        current = source_by_name().get(current_source)
        if current:
            current_family = set(current.family_aliases)
            for item in candidates:
                if current_family.intersection(item.family_aliases):
                    return item.source
    return candidates[0].source


def resolve_same_doc_reference(current_source: str, article_number: str | None) -> str | None:
    if _is_civil_code_alias(_clean_alias(current_source)):
        return _secondary_source_by_alias("гк рф") or current_source

    current = source_by_name().get(current_source)
    if not current:
        return current_source
    family_candidates: list[SourceInfo] = []
    for family_alias in current.family_aliases:
        family_candidates.extend(grouped_family_sources().get(family_alias, ()))
    unique_candidates = list({item.source: item for item in family_candidates}.values()) or [current]
    return _resolve_by_article_range(unique_candidates, article_number, current_source=current_source)


def infer_specific_source_filter(text: str) -> str | None:
    """Heuristic disabled — lexical match against PRIMAL source names was unreliable.
    Stub kept so callers don't need to change if the approach is restored.
    """
    return None
