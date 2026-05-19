"""Filesystem path helpers for the legacy local DOCX storage.

Production stores contract DOCX files in S3/MinIO; these helpers exist to
serve the legacy `data/contracts/<case_id>/v<n>.docx` layout that survived
from the pre-S3 era. `_is_within_contracts` is a path-traversal guard:
every resolved path must sit under CONTRACTS_DIR before we hand it to
FileResponse.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOGGER = logging.getLogger("cases.docx_paths")

APP_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_DIR = (APP_ROOT / "data" / "contracts").resolve()
LEGACY_MARKERS = ("/data/contracts/", "\\data\\contracts\\")


def _is_within_contracts(path: Path) -> bool:
    try:
        path.relative_to(CONTRACTS_DIR)
        return True
    except ValueError:
        return False


def canonical_docx_path(case_id: str, version_number: int) -> Path | None:
    candidate = (CONTRACTS_DIR / str(case_id) / f"v{version_number}.docx").resolve()
    if not _is_within_contracts(candidate):
        LOGGER.warning("Refused canonical path outside CONTRACTS_DIR: %s", candidate)
        return None
    return candidate


def resolve_docx_path(raw_path: str) -> Path | None:
    try:
        path = Path(raw_path).resolve()
    except OSError:
        LOGGER.warning("Invalid DOCX path: %s", raw_path)
        return None
    if path.exists() and _is_within_contracts(path):
        return path

    normalized = raw_path.replace("\\", "/")
    for marker in LEGACY_MARKERS:
        marker_normalized = marker.replace("\\", "/")
        if marker_normalized in normalized:
            suffix = normalized.split(marker_normalized, 1)[1].lstrip("/")
            candidate = (CONTRACTS_DIR / suffix).resolve()
            if candidate.exists() and _is_within_contracts(candidate):
                LOGGER.info("Resolved legacy DOCX path %s -> %s", raw_path, candidate)
                return candidate
    LOGGER.warning("DOCX path rejected or missing: %s", raw_path)
    return None
