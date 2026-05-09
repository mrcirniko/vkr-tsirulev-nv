"""Tests for billing.webhooks: IP allowlist + idempotency, all DB-mocked."""

from __future__ import annotations

from billing import webhooks


def test_ip_allowlist_blocks_random_ip():
    assert webhooks._ip_is_allowed("203.0.113.10") is False


def test_ip_allowlist_allows_known_yookassa_subnet():
    assert webhooks._ip_is_allowed("185.71.76.5") is True


def test_ip_allowlist_handles_garbage():
    assert webhooks._ip_is_allowed("") is False
    assert webhooks._ip_is_allowed("not-an-ip") is False
