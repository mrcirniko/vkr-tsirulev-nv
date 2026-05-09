"""Lenient JSON extractor used by every LLM-driven graph node.

Lives on its own (no LangChain / Ollama imports) so tests can exercise it
without bootstrapping the LLM stack — the same trick we use for `gate.py`
and `routing.py`.
"""

from __future__ import annotations

import json
import re

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def coerce_json(text: str) -> dict:
    """Parse a JSON object out of an LLM reply.

    Even with Ollama's `format='json'`, models occasionally wrap their reply
    in a ```json fence``` or prepend a chatty sentence. This helper:

      1. Strips a markdown fence if present.
      2. Tries `json.loads` on the cleaned string.
      3. As a last resort, extracts the first `{...}` block via regex.

    Raises `json.JSONDecodeError` if no parsable object is found.
    """
    cleaned = (text or "").strip()
    fence = _JSON_FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    obj_match = _JSON_OBJECT_RE.search(cleaned)
    if obj_match:
        return json.loads(obj_match.group(0))
    raise json.JSONDecodeError("no JSON object found", cleaned, 0)
