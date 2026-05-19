"""Unit tests for billing.webhooks IP-allowlist and X-Forwarded-For parsing.

Pure-logic tests — no FastAPI app needed. We construct lightweight stand-ins
that satisfy the duck-typed Request interface used by `_client_ip` (only
.headers and .client attributes are read).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from billing import webhooks


@dataclass
class _FakeClient:
    host: str | None


class _FakeRequest:
    def __init__(self, *, headers: dict | None = None, client_host: str | None = "1.2.3.4") -> None:
        self.headers = headers or {}
        self.client = _FakeClient(client_host) if client_host is not None else None


# ---------------------------------------------------------------- _ip_is_allowed


@pytest.mark.parametrize(
    "ip",
    [
        # 185.71.76.0/27 covers .0..31
        "185.71.76.0",
        "185.71.76.31",
        # 185.71.77.0/27
        "185.71.77.5",
        # /25 networks
        "77.75.153.10",
        "77.75.154.200",
        # /32 single hosts
        "77.75.156.11",
        "77.75.156.35",
    ],
)
def test_ip_is_allowed_yookassa_ranges(ip):
    assert webhooks._ip_is_allowed(ip) is True


@pytest.mark.parametrize(
    "ip",
    [
        "8.8.8.8",
        "127.0.0.1",
        "10.0.0.1",
        # one off the /27 boundary
        "185.71.76.32",
        "185.71.76.255",
        # outside the /32
        "77.75.156.12",
    ],
)
def test_ip_is_allowed_rejects_outside_ranges(ip):
    assert webhooks._ip_is_allowed(ip) is False


def test_ip_is_allowed_ipv6_yookassa_range():
    # 2a02:5180::/32 — first address inside the prefix
    assert webhooks._ip_is_allowed("2a02:5180::1") is True


def test_ip_is_allowed_ipv6_outside_range():
    assert webhooks._ip_is_allowed("2a01::1") is False


@pytest.mark.parametrize("bad", ["", "not-an-ip", "999.999.999.999", "::zz"])
def test_ip_is_allowed_handles_invalid_input(bad):
    assert webhooks._ip_is_allowed(bad) is False


# ---------------------------------------------------------------- _client_ip


def test_client_ip_prefers_first_x_forwarded_for():
    request = _FakeRequest(
        headers={"x-forwarded-for": "203.0.113.5, 10.0.0.1, 192.168.1.1"},
        client_host="127.0.0.1",
    )
    # Even though the direct client is localhost, X-Forwarded-For wins —
    # this matches the production behavior behind nginx.
    assert webhooks._client_ip(request) == "203.0.113.5"


def test_client_ip_strips_whitespace_in_xff():
    request = _FakeRequest(headers={"x-forwarded-for": "   185.71.76.5   ,  bla"})
    assert webhooks._client_ip(request) == "185.71.76.5"


def test_client_ip_falls_back_to_request_client():
    request = _FakeRequest(client_host="185.71.76.10")
    assert webhooks._client_ip(request) == "185.71.76.10"


def test_client_ip_returns_empty_when_no_client_and_no_xff():
    request = _FakeRequest(client_host=None)
    assert webhooks._client_ip(request) == ""


def test_client_ip_handles_client_with_none_host():
    request = _FakeRequest(client_host=None)
    request.client = _FakeClient(host=None)
    assert webhooks._client_ip(request) == ""


# ---------------------------------------------------------------- network coverage


def test_yookassa_nets_constant_includes_ipv6():
    """Sanity check on the constant tuple — guards against accidental regression
    if someone reformats the list and drops the v6 prefix."""
    families = {net.version for net in webhooks._YOOKASSA_NETS}
    assert 4 in families
    assert 6 in families
