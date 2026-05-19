"""Background jobs for admin NPA workflow.

Two long-running operations are exposed:

- run_chunking_job(): re-chunk an existing NPA without LLM reference
  extraction. Emits progress via WS so the admin
  page can show a spinner.
- run_indexing_job(): chunk (with optional LLM ref extraction) + embed +
  upsert into Qdrant. The slow path. Emits per-batch progress.

A single per-admin asyncio.Lock serializes jobs for the same admin so two
concurrent indexings don't fight over GPU/Ollama. We keep it lightweight
(in-memory dict) — fine for one admin operator on a single worker.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from admin import crud as admin_crud
from admin import storage as admin_storage
from admin.realtime import emit_admin_event
from db.models import NpaStatus
from rag.indexer import (
    GENERAL_GROUP,
    PRIMAL_GROUP,
    SECONDARY_GROUP,
    chunk_text,
    index_chunks_for_source,
)

LOGGER = logging.getLogger("app.admin.tasks")

# Global lock — indexing and chunking share GPU/Ollama, so they're serialized.
_JOB_LOCK = asyncio.Lock()

# Latest progress event per npa_id; used by ADMIN_WS on reconnect to rebuild progressMap.
_LIVE_PROGRESS: dict[str, dict[str, Any]] = {}


def snapshot_live_progress() -> list[dict[str, Any]]:
    """Return a copy of in-flight progress events, one per active npa_id."""
    return [dict(event) for event in _LIVE_PROGRESS.values()]


def _collection_for_group(source_group: str) -> str:
    from config import settings

    if source_group == GENERAL_GROUP:
        return settings.qdrant_collection_general
    if source_group == PRIMAL_GROUP:
        return settings.qdrant_collection_primal
    if source_group == SECONDARY_GROUP:
        return settings.qdrant_collection_secondary
    raise ValueError(f"Unsupported source_group: {source_group!r}")


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


_TERMINAL_STATUSES = {"indexed", "ready", "failed"}
_IN_FLIGHT_STATUSES = {"indexing", "chunking"}


def _track_for_snapshot(event_type: str, payload: dict[str, Any]) -> None:
    """Maintain `_LIVE_PROGRESS` so reconnecting admin clients see in-flight
    jobs immediately. Keeps the latest progress event per npa_id; clears on
    terminal status. Status-only events (indexing/chunking/etc.) seed the
    entry if no progress has arrived yet so the UI shows *something*."""
    npa_id = payload.get("npa_id")
    if not npa_id:
        return
    key = str(npa_id)
    event = {"type": event_type, **payload}
    if event_type == "admin_npa_progress":
        _LIVE_PROGRESS[key] = event
        return
    if event_type == "admin_npa_status":
        status = payload.get("status")
        if status in _TERMINAL_STATUSES:
            _LIVE_PROGRESS.pop(key, None)
        elif status in _IN_FLIGHT_STATUSES and key not in _LIVE_PROGRESS:
            _LIVE_PROGRESS[key] = event


async def _emit(admin_id: str, event_type: str, **payload: Any) -> None:
    _track_for_snapshot(event_type, payload)
    await emit_admin_event(admin_id, {"type": event_type, **payload})


async def run_chunking_job(
    *,
    admin_id: str,
    npa_id: str,
    source_group: str,
) -> None:
    """Re-chunk a single NPA without LLM extraction (fast)."""
    if _JOB_LOCK.locked():
        await _emit(admin_id, "admin_npa_busy", npa_id=npa_id)
    async with _JOB_LOCK:
        try:
            npa = admin_crud.get_npa_source(npa_id)
            if npa is None:
                await _emit(admin_id, "admin_npa_failed", npa_id=npa_id, error="not_found")
                return

            admin_crud.update_npa_source(npa_id, status=NpaStatus.CHUNKING, clear_error=True)
            await _emit(admin_id, "admin_npa_status", npa_id=npa_id, status="chunking")

            raw_text = await asyncio.to_thread(admin_storage.download_text, npa.raw_txt_s3_key)
            chunks = await asyncio.to_thread(
                chunk_text,
                raw_text,
                npa.source_name,
                source_group,
                "morphology",
            )

            chunks_json = json.dumps(chunks, ensure_ascii=False, indent=2)
            chunks_key = admin_storage.chunks_json_key(npa_id)
            await asyncio.to_thread(
                admin_storage.upload_text,
                chunks_key,
                chunks_json,
                "application/json",
            )

            admin_crud.update_npa_source(
                npa_id,
                status=NpaStatus.READY,
                chunks_json_s3_key=chunks_key,
                chunks_count=len(chunks),
            )
            await _emit(
                admin_id,
                "admin_npa_status",
                npa_id=npa_id,
                status="ready",
                chunks_count=len(chunks),
            )
        except Exception as exc:
            LOGGER.exception("Chunking job failed for npa_id=%s", npa_id)
            try:
                admin_crud.update_npa_source(npa_id, status=NpaStatus.FAILED, error_text=str(exc))
            except Exception:
                LOGGER.exception("Also failed to mark npa as FAILED")
            await _emit(admin_id, "admin_npa_failed", npa_id=npa_id, error=str(exc))


async def run_indexing_job(
    *,
    admin_id: str,
    npa_id: str,
    source_group: str,
    recreate_collection: bool,
    delete_existing_for_source: bool = True,
) -> None:
    """Embed + upsert a single NPA into the chosen Qdrant collection.

    Reads the chunks JSON produced at upload time — does NOT re-chunk.
    Reference extraction is decided once, at upload, via the
    `extract_references` flag on the upload form. The only thing this
    job rewrites in each chunk is `is_general`, which depends on the
    group choice that's only known here.
    """
    if _JOB_LOCK.locked():
        await _emit(admin_id, "admin_npa_busy", npa_id=npa_id)
    async with _JOB_LOCK:
        try:
            npa = admin_crud.get_npa_source(npa_id)
            if npa is None:
                await _emit(admin_id, "admin_npa_failed", npa_id=npa_id, error="not_found")
                return
            if not npa.chunks_json_s3_key:
                await _emit(admin_id, "admin_npa_failed", npa_id=npa_id, error="no_chunks")
                admin_crud.update_npa_source(npa_id, status=NpaStatus.FAILED, error_text="Chunks JSON отсутствует")
                return

            collection = _collection_for_group(source_group)

            admin_crud.update_npa_source(npa_id, status=NpaStatus.INDEXING, clear_error=True)
            await _emit(
                admin_id,
                "admin_npa_status",
                npa_id=npa_id,
                status="indexing",
                stage="loading",
                collection=collection,
            )

            chunks_text = await asyncio.to_thread(admin_storage.download_text, npa.chunks_json_s3_key)
            chunks = json.loads(chunks_text)
            is_general = source_group == GENERAL_GROUP
            for chunk in chunks:
                chunk["is_general"] = is_general

            await _emit(
                admin_id,
                "admin_npa_progress",
                npa_id=npa_id,
                stage="indexing",
                processed=0,
                total=len(chunks),
            )

            loop = asyncio.get_running_loop()

            def _progress(done: int, total: int) -> None:
                # Worker-thread → main-loop hop for non-blocking WS emit.
                asyncio.run_coroutine_threadsafe(
                    _emit(
                        admin_id,
                        "admin_npa_progress",
                        npa_id=npa_id,
                        stage="indexing",
                        processed=done,
                        total=total,
                    ),
                    loop,
                )

            # Per-source delete is redundant when recreate_collection wipes everything.
            effective_delete = False if recreate_collection else delete_existing_for_source
            inserted = await asyncio.to_thread(
                index_chunks_for_source,
                collection_name=collection,
                chunks=chunks,
                source_name=npa.source_name,
                recreate_collection=recreate_collection,
                delete_existing_for_source=effective_delete,
                progress_cb=_progress,
            )

            had_refs = bool((npa.conversion_options or {}).get("extract_references"))
            admin_crud.update_npa_source(
                npa_id,
                status=NpaStatus.INDEXED,
                last_indexed_collection=collection,
                last_indexed_with_refs=had_refs,
            )
            await _emit(
                admin_id,
                "admin_npa_status",
                npa_id=npa_id,
                status="indexed",
                collection=collection,
                indexed_count=inserted,
                with_refs=had_refs,
                indexed_at=_utcnow_iso(),
            )
        except Exception as exc:
            LOGGER.exception("Indexing job failed for npa_id=%s", npa_id)
            try:
                admin_crud.update_npa_source(npa_id, status=NpaStatus.FAILED, error_text=str(exc))
            except Exception:
                LOGGER.exception("Also failed to mark npa as FAILED")
            await _emit(admin_id, "admin_npa_failed", npa_id=npa_id, error=str(exc))
