"""Real-time fanout for the FastAPI app via WebSockets.

Each authenticated browser holds a single WS to /api/ws. Server-side
events about the user's cases (new message, status change, contract
version, title update, etc.) are broadcast to every WS belonging to
that user.

This is intentionally tiny — single process, in-memory registry.
For multi-worker deployments a Redis pub/sub fanout would be needed.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any
from uuid import UUID

from fastapi import WebSocket

LOGGER = logging.getLogger("app.realtime")


class WSManager:
    def __init__(self) -> None:
        self._user_to_sockets: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def register(self, user_id: str | UUID, ws: WebSocket) -> None:
        user_key = str(user_id)
        async with self._lock:
            self._user_to_sockets[user_key].add(ws)
        LOGGER.info("WS registered user_id=%s total=%d", user_key, len(self._user_to_sockets[user_key]))

    async def unregister(self, user_id: str | UUID, ws: WebSocket) -> None:
        user_key = str(user_id)
        async with self._lock:
            sockets = self._user_to_sockets.get(user_key)
            if sockets and ws in sockets:
                sockets.discard(ws)
                if not sockets:
                    self._user_to_sockets.pop(user_key, None)
        LOGGER.info("WS unregistered user_id=%s", user_key)

    async def broadcast_to_user(self, user_id: str | UUID, event: dict[str, Any]) -> None:
        user_key = str(user_id)
        async with self._lock:
            sockets = list(self._user_to_sockets.get(user_key, ()))
        if not sockets:
            return
        for ws in sockets:
            try:
                await ws.send_json(event)
            except Exception as exc:
                LOGGER.warning("WS send failed for user=%s: %s; will be cleaned on next disconnect", user_key, exc)


WS_MANAGER = WSManager()


def _serialize(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    return value


def _normalize(event: dict[str, Any]) -> dict[str, Any]:
    return {k: _serialize(v) for k, v in event.items()}


async def emit_to_user(user_id: str | UUID, event: dict[str, Any]) -> None:
    await WS_MANAGER.broadcast_to_user(user_id, _normalize(event))


def emit_to_user_threadsafe(loop: asyncio.AbstractEventLoop, user_id: str | UUID, event: dict[str, Any]) -> None:
    """Schedule an emit from a non-async context (background thread, etc.).

    Currently unused but kept around because asyncio.to_thread callbacks
    occasionally need to push WS events without re-entering the loop.
    """
    asyncio.run_coroutine_threadsafe(emit_to_user(user_id, event), loop)
