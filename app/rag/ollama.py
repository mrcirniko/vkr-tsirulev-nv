from __future__ import annotations

import logging

import httpx
from config import settings

LOGGER = logging.getLogger("rag.ollama")


def unload_ollama_model(model: str | None = None) -> None:
    """Ask Ollama to unload a model from RAM/VRAM without deleting it from disk."""
    model_name = model or settings.llm_model
    if not model_name:
        return

    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_name,
        "prompt": "",
        "stream": False,
        "keep_alive": 0,
    }
    try:
        response = httpx.post(url, json=payload, timeout=15.0)
        response.raise_for_status()
        LOGGER.info("Requested Ollama unload for model=%s", model_name)
    except Exception as exc:
        LOGGER.warning("Failed to unload Ollama model=%s: %s", model_name, exc)
