from __future__ import annotations

import logging

from config import settings

from rag.embedding_provider import preload_embedding_provider

LOGGER = logging.getLogger("rag.preload")


def preload_startup_models() -> None:
    if not settings.preload_embeddings_on_startup:
        return
    try:
        preload_embedding_provider()
    except Exception as exc:
        LOGGER.warning("Failed to preload embeddings on startup: %s", exc)
