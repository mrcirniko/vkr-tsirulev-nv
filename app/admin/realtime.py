"""Admin-side WebSocket fanout.

Mirrors `app.realtime.WS_MANAGER` but keys by `admin_id` so user-channel
events don't leak into the admin panel and vice versa. One panel = one
admin = one or two WS connections (fresh page reload while the previous
WS is still closing).
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any
from uuid import UUID

from fastapi import WebSocket

LOGGER = logging.getLogger("app.admin.realtime")


class _AdminWSManager:
    def __init__(self) -> None:
        self._sockets: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def register(self, admin_id: str | UUID, ws: WebSocket) -> None:
        key = str(admin_id)
        async with self._lock:
            self._sockets[key].add(ws)
        LOGGER.info("Admin WS registered admin_id=%s total=%d", key, len(self._sockets[key]))

    async def unregister(self, admin_id: str | UUID, ws: WebSocket) -> None:
        key = str(admin_id)
        async with self._lock:
            sockets = self._sockets.get(key)
            if sockets and ws in sockets:
                sockets.discard(ws)
                if not sockets:
                    self._sockets.pop(key, None)
        LOGGER.info("Admin WS unregistered admin_id=%s", key)

    async def broadcast(self, admin_id: str | UUID, event: dict[str, Any]) -> None:
        key = str(admin_id)
        async with self._lock:
            sockets = list(self._sockets.get(key, ()))
        for ws in sockets:
            try:
                await ws.send_json(event)
            except Exception as exc:
                LOGGER.warning("Admin WS send failed admin=%s: %s", key, exc)


ADMIN_WS = _AdminWSManager()


def _serialize(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    return value


async def emit_admin_event(admin_id: str | UUID, event: dict[str, Any]) -> None:
    payload = {k: _serialize(v) for k, v in event.items()}
    await ADMIN_WS.broadcast(admin_id, payload)
