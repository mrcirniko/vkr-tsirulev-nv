"""Unit tests for rag.source_registry pure helpers and resolution logic.

The registry is normally built from the `npa_sources` DB table. Here we
patch `get_source_registry` to inject hand-crafted SourceInfo records and
exercise the alias / family / article-range resolution logic without any
DB or Qdrant. All other lru_cache-decorated helpers are cleared in the
fixture so each test starts from a clean slate.
"""

from __future__ import annotations

import pytest

from rag import source_registry as sr
from rag.source_registry import (
    GENERAL_GROUP,
    PRIMAL_GROUP,
    SECONDARY_GROUP,
    SourceInfo,
    canonical_source_name,
    infer_specific_source_filter,
    resolve_same_doc_reference,
    resolve_source_by_alias,
)


def _make_source(
    name: str,
    group: str = PRIMAL_GROUP,
    *,
    article_min: float | None = None,
    article_max: float | None = None,
) -> SourceInfo:
    return SourceInfo(
        source=name,
        group=group,
        path="",
        aliases=sr._generic_aliases(name),
        family_aliases=sr._family_aliases(name),
        article_min=article_min,
        article_max=article_max,
        same_doc_markers=sr._same_doc_markers(name),
    )


def _safe_cache_clear(*funcs):
    """Call cache_clear on lru_cache-wrapped funcs; tolerate plain callables
    (the case during teardown after monkeypatch.setattr replaced the original)."""
    for fn in funcs:
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()


@pytest.fixture
def patched_registry(monkeypatch):
    """Inject a fixed registry and clear every lru_cache that depends on it."""

    def _install(sources: list[SourceInfo]) -> list[SourceInfo]:
        _safe_cache_clear(
            sr.get_source_registry,
            sr.get_known_sources,
            sr.source_by_name,
            sr.grouped_family_sources,
        )
        monkeypatch.setattr(sr, "get_source_registry", lambda: tuple(sources))
        # Dependent caches read `get_source_registry` through the module
        # binding — flushing them is enough; the patched function gets picked
        # up on the next call.
        _safe_cache_clear(
            sr.get_known_sources,
            sr.source_by_name,
            sr.grouped_family_sources,
        )
        return sources

    yield _install
    # After yield monkeypatch hasn't undone the setattr yet — sr.get_source_registry
    # is still the lambda. Use _safe_cache_clear so we don't crash on the lambda.
    _safe_cache_clear(
        sr.get_source_registry,
        sr.get_known_sources,
        sr.source_by_name,
        sr.grouped_family_sources,
    )


# ---------------------------------------------------------------- pure helpers


def test_canonical_source_name_strips_extension_and_normalizes():
    assert canonical_source_name("ГК_РФ_часть_первая.txt") == "ГК РФ часть первая"
    assert canonical_source_name("/path/to/Закон  о  защите.rtf") == "Закон о защите"


def test_normalize_spaces_collapses_underscores_and_runs():
    assert sr._normalize_spaces("a__b   c") == "a b c"
    assert sr._normalize_spaces("  leading_trailing  ") == "leading trailing"


def test_clean_alias_strips_punctuation_and_lowercases():
    assert sr._clean_alias("ГК (РФ),  часть 1.") == "гк рф часть 1"
    assert sr._clean_alias('"Закон о защите"') == "закон о защите"


def test_family_name_strips_part_marker():
    """`_family_name` must drop "часть N" so different parts of the same code
    cluster under one family alias for cross-part references."""
    assert sr._family_name("ГК РФ часть первая") == "ГК РФ"
    assert sr._family_name("ГК РФ часть 2") == "ГК РФ"
    assert sr._family_name("ГК РФ") == "ГК РФ"


def test_generic_aliases_expands_rf_variants():
    aliases = sr._generic_aliases("Гражданский кодекс Российской Федерации")
    assert "гражданский кодекс российской федерации" in aliases
    assert "гражданский кодекс рф" in aliases


def test_generic_aliases_expands_gk_rf_when_base_starts_with_it():
    """`гк рф` shorthand only appears when the source name explicitly starts
    with 'Гражданский кодекс РФ' — see implementation: the `startswith` check
    runs against the original base, not the rewritten one."""
    aliases = sr._generic_aliases("Гражданский кодекс РФ")
    assert "гк рф" in aliases


def test_generic_aliases_handles_fz_prefix():
    aliases = sr._generic_aliases("ФЗ О защите прав потребителей")
    assert "фз о защите прав потребителей" in aliases
    assert "о защите прав потребителей" in aliases


def test_same_doc_markers_for_codex():
    """Markers are derived from the cleaned name string. The abbreviation
    'ГК РФ' alone doesn't contain the substring 'кодекс', so we test with
    the full name where the substring is present."""
    markers = sr._same_doc_markers("Гражданский кодекс РФ")
    assert "настоящего кодекса" in markers
    assert "настоящим кодексом" in markers


def test_same_doc_markers_empty_for_abbreviated_name():
    """Abbreviation-only names yield no markers — the function relies on
    substring matches against 'кодекс' / 'закон' in the cleaned form."""
    assert sr._same_doc_markers("ГК РФ") == ()


def test_same_doc_markers_for_fz():
    markers = sr._same_doc_markers("ФЗ Об ипотеке")
    assert "настоящего закона" in markers
    # The "ипотеке" string also contains "кодекс"? No — but it contains
    # "закон". Check no kodeks markers leaked in.
    assert all("кодекс" not in m for m in markers)


def test_same_doc_markers_dedupes_for_codex_with_law_in_name():
    """A source like 'Жилищный кодекс' contains "кодекс" and may also match
    "закон" via substring — markers must be deduplicated."""
    markers = sr._same_doc_markers("Жилищный кодекс РФ")
    assert len(markers) == len(set(markers))


def test_is_civil_code_alias_matches_variants():
    assert sr._is_civil_code_alias("гк рф") is True
    assert sr._is_civil_code_alias("гк рф часть 1") is True
    assert sr._is_civil_code_alias("гражданский кодекс") is True
    assert sr._is_civil_code_alias("ук рф") is False
    assert sr._is_civil_code_alias("закон о защите") is False


# ---------------------------------------------------------------- resolve_source_by_alias


def test_resolve_source_by_alias_civil_code_special_case(patched_registry):
    patched_registry(
        [
            _make_source("ГК РФ", group=SECONDARY_GROUP),
            _make_source("ГК РФ часть первая", group=PRIMAL_GROUP),
        ]
    )
    # "гк рф" must always resolve to the SECONDARY canonical entry, not to
    # any PRIMAL part-specific source — see _is_civil_code_alias / docstring.
    assert resolve_source_by_alias("ГК РФ") == "ГК РФ"
    assert resolve_source_by_alias("Гражданский кодекс РФ") == "ГК РФ"


def test_resolve_source_by_alias_unique_match(patched_registry):
    patched_registry([_make_source("ФЗ Об ипотеке")])
    assert resolve_source_by_alias("ФЗ Об ипотеке") == "ФЗ Об ипотеке"
    # Stripped FZ prefix variant — generated as an alias.
    assert resolve_source_by_alias("Об ипотеке") == "ФЗ Об ипотеке"


def test_resolve_source_by_alias_unknown_returns_none(patched_registry):
    patched_registry([_make_source("ФЗ Об ипотеке")])
    assert resolve_source_by_alias("Налоговый кодекс") is None


def test_resolve_source_by_alias_disambiguates_by_article_range(patched_registry):
    """Two sources sharing a family alias get disambiguated by article range —
    used when a doc has multiple parts (e.g. ГК РФ ч. 1: статьи 1–453)."""
    patched_registry(
        [
            _make_source("ТК РФ часть первая", article_min=1, article_max=200),
            _make_source("ТК РФ часть вторая", article_min=201, article_max=400),
        ]
    )
    # Article 50 → part one, article 250 → part two
    assert resolve_source_by_alias("ТК РФ", article_number="50") == "ТК РФ часть первая"
    assert resolve_source_by_alias("ТК РФ", article_number="250") == "ТК РФ часть вторая"


# ---------------------------------------------------------------- resolve_same_doc_reference


def test_resolve_same_doc_reference_falls_through_for_unknown_source(patched_registry):
    patched_registry([_make_source("ФЗ Об ипотеке")])
    # Unknown current_source — function should return it unchanged so the
    # reference is still recorded with whatever name was passed.
    assert resolve_same_doc_reference("Совсем другой закон", "5") == "Совсем другой закон"


def test_resolve_same_doc_reference_civil_code_special(patched_registry):
    patched_registry(
        [
            _make_source("ГК РФ", group=SECONDARY_GROUP),
            _make_source("ГК РФ часть первая", group=PRIMAL_GROUP),
        ]
    )
    # Self-reference inside any ГК part must collapse to the canonical ГК РФ.
    assert resolve_same_doc_reference("ГК РФ часть первая", "10") == "ГК РФ"


def test_resolve_same_doc_reference_resolves_within_family(patched_registry):
    patched_registry(
        [
            _make_source("ТК РФ часть первая", article_min=1, article_max=200),
            _make_source("ТК РФ часть вторая", article_min=201, article_max=400),
        ]
    )
    # "статьей 250 настоящего Кодекса" inside part-one body must resolve to
    # part-two by article range.
    assert resolve_same_doc_reference("ТК РФ часть первая", "250") == "ТК РФ часть вторая"


# ---------------------------------------------------------------- infer_specific_source_filter


def test_infer_specific_source_filter_picks_best_word_overlap(patched_registry):
    """Word overlap is computed by exact match (no morphology/lemmatization),
    so the query has to contain a word form that's present in the source name."""
    patched_registry(
        [
            _make_source("ФЗ Об ипотеке", group=PRIMAL_GROUP),
            _make_source("ФЗ О долевом строительстве", group=PRIMAL_GROUP),
            _make_source("ГК РФ", group=SECONDARY_GROUP),  # secondary excluded
        ]
    )
    # Query contains "ипотеке" — exact match with the same word form in source.
    assert infer_specific_source_filter("Договор об ипотеке") == "ФЗ Об ипотеке"
    # No matching word in any primal source → None.
    assert infer_specific_source_filter("Совсем другой текст") is None


def test_infer_specific_source_filter_ignores_secondary_group(patched_registry):
    """Secondary group is for cross-references; it must NOT be the target of
    a primal filter inference."""
    patched_registry(
        [
            _make_source("ФЗ Об ипотеке", group=PRIMAL_GROUP),
            # Secondary source whose words also appear in the query — should
            # still be skipped.
            _make_source("Закон об ипотеке", group=SECONDARY_GROUP),
        ]
    )
    assert infer_specific_source_filter("договор об ипотеке") == "ФЗ Об ипотеке"


def test_infer_specific_source_filter_empty_text_returns_none(patched_registry):
    patched_registry([_make_source("ФЗ Об ипотеке")])
    assert infer_specific_source_filter("") is None
    assert infer_specific_source_filter("   ") is None


# ---------------------------------------------------------------- group resolution


def test_group_from_collection_maps_known_collections(monkeypatch):
    from config import settings

    object.__setattr__(settings, "qdrant_collection_general", "test_general")
    object.__setattr__(settings, "qdrant_collection_primal", "test_primal")
    object.__setattr__(settings, "qdrant_collection_secondary", "test_secondary")
    try:
        assert sr._group_from_collection("test_general") == GENERAL_GROUP
        assert sr._group_from_collection("test_primal") == PRIMAL_GROUP
        assert sr._group_from_collection("test_secondary") == SECONDARY_GROUP
        assert sr._group_from_collection("unknown_collection") is None
        assert sr._group_from_collection(None) is None
        assert sr._group_from_collection("") is None
    finally:
        # Restore real defaults for downstream tests.
        object.__setattr__(settings, "qdrant_collection_general", "npa_general")
        object.__setattr__(settings, "qdrant_collection_primal", "npa_primal")
        object.__setattr__(settings, "qdrant_collection_secondary", "npa_secondary")
