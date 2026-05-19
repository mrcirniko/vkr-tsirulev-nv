"""Tests for agent.deal_types — TTL cache, S3→disk fallback, validation."""

from __future__ import annotations

import json

import pytest

from agent import deal_types


@pytest.fixture(autouse=True)
def _reset_cache():
    """Drop the module-level cache before/after every test so cases are
    independent. The cache is a global tuple; bleeding state between tests
    masks bugs."""
    deal_types._cache = None
    yield
    deal_types._cache = None


# ---------------------------------------------------------------- disk fallback


def test_load_from_disk_returns_real_catalog():
    """The bundled JSON file shipped in the repo is the canonical fallback;
    if it goes missing or becomes malformed, the entire graph breaks."""
    data = deal_types._load_from_disk()
    assert isinstance(data, dict)
    assert data, "bundled catalog must not be empty"
    for key, value in data.items():
        assert isinstance(key, str)
        assert isinstance(value, str)


def test_falls_back_to_disk_when_s3_unavailable(monkeypatch):
    monkeypatch.setattr(deal_types, "_load_from_s3", lambda: None)
    data = deal_types.get_supported_deal_type_descriptions()
    # Should match the bundled file exactly.
    assert data == deal_types._load_from_disk()


# ---------------------------------------------------------------- S3 priority


def test_s3_takes_priority_over_disk(monkeypatch):
    s3_payload = {"Договор оказания услуг": "S3-only description"}
    monkeypatch.setattr(deal_types, "_load_from_s3", lambda: s3_payload)
    # Disk loader must NOT be touched when S3 wins.
    monkeypatch.setattr(
        deal_types,
        "_load_from_disk",
        lambda: pytest.fail("disk fallback must not run when S3 returned data"),
    )
    assert deal_types.get_supported_deal_type_descriptions() == s3_payload


# ---------------------------------------------------------------- caching


def test_cache_returns_same_value_on_subsequent_calls(monkeypatch):
    counter = {"calls": 0}

    def _fake_s3():
        counter["calls"] += 1
        return {"Договор подряда": "x"}

    monkeypatch.setattr(deal_types, "_load_from_s3", _fake_s3)
    first = deal_types.get_supported_deal_type_descriptions()
    second = deal_types.get_supported_deal_type_descriptions()
    assert first == second
    # Within TTL, S3 should be hit at most once. Implementation reads the
    # cached dict on the second call.
    assert counter["calls"] == 1


def test_invalidate_cache_forces_reload(monkeypatch):
    payloads = iter([{"a": "1"}, {"b": "2"}])
    monkeypatch.setattr(deal_types, "_load_from_s3", lambda: next(payloads))
    first = deal_types.get_supported_deal_type_descriptions()
    deal_types.invalidate_cache()
    second = deal_types.get_supported_deal_type_descriptions()
    assert first == {"a": "1"}
    assert second == {"b": "2"}


# ---------------------------------------------------------------- validation


def test_s3_loader_rejects_malformed_json(monkeypatch):
    """`_load_from_s3` swallows JSONDecodeError and returns None so the
    caller drops to disk."""

    def _fake_download(_key):
        return "this is not json {"

    fake_storage = type("M", (), {"download_text": staticmethod(_fake_download)})

    import sys

    monkeypatch.setitem(sys.modules, "admin.storage", fake_storage)
    assert deal_types._load_from_s3() is None


def test_s3_loader_rejects_wrong_shape(monkeypatch):
    """A JSON list at the top level is valid JSON but wrong type — must be
    rejected so the dict assumption downstream holds."""

    def _fake_download(_key):
        return json.dumps(["not", "a", "dict"])

    fake_storage = type("M", (), {"download_text": staticmethod(_fake_download)})

    import sys

    monkeypatch.setitem(sys.modules, "admin.storage", fake_storage)
    assert deal_types._load_from_s3() is None


def test_s3_loader_rejects_non_string_values(monkeypatch):
    def _fake_download(_key):
        return json.dumps({"deal": 42})  # value is int, not str

    fake_storage = type("M", (), {"download_text": staticmethod(_fake_download)})

    import sys

    monkeypatch.setitem(sys.modules, "admin.storage", fake_storage)
    assert deal_types._load_from_s3() is None


# ---------------------------------------------------------------- public API shape


def test_get_supported_deal_types_returns_tuple_of_keys(monkeypatch):
    payload = {"Купля-продажа": "x", "Подряд": "y"}
    monkeypatch.setattr(deal_types, "_load_from_s3", lambda: payload)
    result = deal_types.get_supported_deal_types()
    assert isinstance(result, tuple)
    assert set(result) == set(payload.keys())
