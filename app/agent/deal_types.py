"""Lazy access to the supported-deal-types catalog.

The catalog lives in MinIO/S3 (`config/supported_deal_types.json` in the
admin-npa bucket) and is editable from the admin panel. Both the FastAPI
app and the langgraph_dev container read it through this module.

Two-layer load: try S3 first, fall back to the bundled JSON file shipped
in the repo. Results are memoized for `_TTL_SECONDS` so a freshly-saved
catalog propagates to the running graph within a few seconds without a
container restart.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

LOGGER = logging.getLogger("agent.deal_types")

_TTL_SECONDS = 5.0
_FALLBACK_PATH = Path(__file__).with_name("supported_deal_types.json")
_S3_KEY = "config/supported_deal_types.json"

_lock = threading.Lock()
_cache: tuple[float, dict[str, str]] | None = None


def _load_from_s3() -> dict[str, str] | None:
    try:
        from admin.storage import download_text

        text = download_text(_S3_KEY)
    except Exception as exc:
        LOGGER.debug("Deal types not available from S3: %s", exc)
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        LOGGER.exception("Deal types JSON in S3 is malformed; falling back to disk")
        return None
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        LOGGER.warning("Deal types JSON in S3 has wrong shape; falling back to disk")
        return None
    return data


def _load_from_disk() -> dict[str, str]:
    with _FALLBACK_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def get_supported_deal_type_descriptions() -> dict[str, str]:
    """Return {name: description} for the supported deal types.

    Memoized; pass the cached value through if it's still fresh. On cache
    miss, prefers S3, falls back to the bundled file.
    """
    global _cache
    now = time.monotonic()
    if _cache is not None and (now - _cache[0]) < _TTL_SECONDS:
        return _cache[1]
    with _lock:
        # Re-check inside the lock — another thread may have just refreshed.
        if _cache is not None and (now - _cache[0]) < _TTL_SECONDS:
            return _cache[1]
        data = _load_from_s3()
        if data is None:
            data = _load_from_disk()
        _cache = (now, data)
        return data


def get_supported_deal_types() -> tuple[str, ...]:
    return tuple(get_supported_deal_type_descriptions().keys())


def invalidate_cache() -> None:
    """Drop the in-memory cache. Next read goes to S3.

    Called from the admin PUT handler to force the local process to pick
    up the freshly-saved catalog. langgraph_dev runs in a separate
    container and refreshes via TTL on its own.
    """
    global _cache
    _cache = None


def seed_s3_if_missing() -> None:
    """One-shot: copy the bundled JSON into S3 on first startup.

    Lets you boot a fresh stack without manually uploading the catalog.
    Idempotent — does nothing if the object already exists or if S3 is
    unreachable.
    """
    try:
        from admin.storage import download_text, upload_text
    except Exception:
        return
    try:
        download_text(_S3_KEY)
        return  # already present
    except Exception:
        LOGGER.debug("S3 deal-types catalog absent or unreachable; will attempt to seed", exc_info=True)
    try:
        text = _FALLBACK_PATH.read_text(encoding="utf-8")
        upload_text(_S3_KEY, text, content_type="application/json; charset=utf-8")
        LOGGER.info("Seeded supported_deal_types.json into S3")
    except Exception:
        LOGGER.warning("Failed to seed supported_deal_types.json into S3", exc_info=True)
