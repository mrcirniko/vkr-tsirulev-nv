from __future__ import annotations

import logging
from functools import lru_cache

from config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from rag.chunker import parse_reference
from rag.embedding_provider import clear_embedding_provider, get_embedding_provider
from rag.ollama import unload_ollama_model

DEFAULT_TOP_K = 7
SOURCE_EXPANSION_LIMIT = 3
REFERENCE_COLLECTIONS = (settings.qdrant_collection_secondary,)
LOGGER = logging.getLogger("rag.retriever")


def _prepare_embedding_memory() -> None:
    if settings.memory_swap_mode:
        unload_ollama_model(settings.llm_model)


def _release_embedding_memory() -> None:
    if settings.memory_swap_mode:
        clear_embedding_provider()
        LOGGER.info("Released cached embedding provider after retrieval")


def _client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


def _payload_from_point(point, include_score: bool = True) -> dict:
    payload = dict(point.payload or {})
    if include_score and hasattr(point, "score"):
        payload["score"] = point.score
    if hasattr(point, "id"):
        payload.setdefault("point_id", str(point.id))
    return payload


def _normalize_search_result(result):
    if isinstance(result, tuple):
        return result[0]
    if hasattr(result, "points"):
        return result.points
    return result


def _chunk_key(payload: dict) -> tuple[str, str, str, str]:
    return (
        str(payload.get("point_id") or ""),
        str(payload.get("chunk_id") or ""),
        str(payload.get("source") or ""),
        str(payload.get("article") or ""),
    )


def _fallback_chunk_key(payload: dict) -> tuple[str, str, str, str]:
    return (
        str(payload.get("source") or ""),
        str(payload.get("article") or ""),
        str(payload.get("chunk_index") or ""),
        (str(payload.get("text") or "")[:160]),
    )


@lru_cache(maxsize=1)
def _list_general_chunks() -> list[dict]:
    client = _client()
    chunks: list[dict] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection_general,
            with_payload=True,
            with_vectors=False,
            limit=128,
            offset=offset,
        )
        chunks.extend(dict(point.payload or {}) for point in points)
        if offset is None:
            break
    return chunks


@lru_cache(maxsize=1)
def _reranker():
    if not settings.reranker_enabled:
        return None
    from sentence_transformers import CrossEncoder

    requested_device = settings.reranker_device or None
    LOGGER.info(
        "Loading reranker model=%s device=%s",
        settings.reranker_model,
        requested_device or "auto",
    )
    try:
        return (
            CrossEncoder(settings.reranker_model, device=requested_device)
            if requested_device
            else CrossEncoder(settings.reranker_model)
        )
    except Exception as exc:
        # Common case: requested CUDA but the embedder already filled VRAM.
        # Fall back to CPU rather than crashing the whole retrieval pipeline.
        message = str(exc).lower()
        if requested_device and ("cuda" in requested_device.lower() or "out of memory" in message):
            LOGGER.warning(
                "Reranker failed to load on %s (%s) — falling back to CPU",
                requested_device,
                exc,
            )
            return CrossEncoder(settings.reranker_model, device="cpu")
        LOGGER.exception("Reranker load failed; disabling reranker for this session")
        return None


def _chunk_rerank_text(chunk: dict) -> str:
    # Mirror the indexer's heading prefix so the cross-encoder sees the same
    # "ГК РФ\nСтатья 549. ..." context that the bi-encoder was trained on.
    heading_parts = [
        str(chunk.get(field) or "").strip() for field in ("source", "article") if str(chunk.get(field) or "").strip()
    ]
    body = (chunk.get("text") or "").strip()
    if not heading_parts:
        return body
    return "\n".join(heading_parts + ([body] if body else []))


def _rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    model = _reranker()
    if model is None or not chunks:
        return sorted(chunks, key=lambda item: float(item.get("score") or 0.0), reverse=True)[:top_k]

    pairs = [[query, _chunk_rerank_text(chunk)] for chunk in chunks]
    scores = model.predict(pairs, batch_size=settings.reranker_batch_size)
    scored_chunks: list[dict] = []
    for chunk, score in zip(chunks, scores, strict=True):
        updated = dict(chunk)
        updated["rerank_score"] = float(score)
        scored_chunks.append(updated)
    return sorted(scored_chunks, key=lambda item: item.get("rerank_score", 0.0), reverse=True)[:top_k]


def _merge_points(points) -> list[dict]:
    merged: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for point in points:
        payload = _payload_from_point(point)
        key = _chunk_key(payload)
        if key in seen:
            continue
        seen.add(key)
        merged.append(payload)
    return merged


def retrieve_general(
    query: str | None = None, top_k: int | None = None, candidate_top_k: int | None = None
) -> list[dict]:
    if not query:
        return _list_general_chunks()

    _prepare_embedding_memory()
    final_top_k = top_k or settings.retrieval_general_top_k
    search_top_k = candidate_top_k or max(settings.retrieval_candidate_top_k, final_top_k)
    try:
        candidates = _merge_points(
            _search_collection(
                collection_name=settings.qdrant_collection_general,
                query=query,
                top_k=search_top_k,
            )
        )
        result = _rerank(query, candidates, final_top_k)
    finally:
        _release_embedding_memory()
    LOGGER.info(
        "General retrieval query=%r candidates=%s returned=%s sources=%s",
        query,
        len(candidates),
        len(result),
        [item.get("source") for item in result[:10]],
    )
    return result


def retrieve_secondary(query: str, top_k: int | None = None, candidate_top_k: int | None = None) -> list[dict]:
    if not query:
        return []

    _prepare_embedding_memory()
    final_top_k = top_k or settings.retrieval_secondary_enrichment_top_k
    search_top_k = candidate_top_k or max(settings.retrieval_candidate_top_k, final_top_k)
    try:
        candidates = _merge_points(
            _search_collection(
                collection_name=settings.qdrant_collection_secondary,
                query=query,
                top_k=search_top_k,
            )
        )
        result = _rerank(query, candidates, final_top_k)
    finally:
        _release_embedding_memory()
    LOGGER.info(
        "Secondary retrieval query=%r candidates=%s returned=%s sources=%s",
        query,
        len(candidates),
        len(result),
        [item.get("source") for item in result[:10]],
    )
    return result


def _search_collection(
    collection_name: str,
    query: str,
    source_filter: str | None = None,
    top_k: int = DEFAULT_TOP_K,
):
    client = _client()
    query_vector = get_embedding_provider().embed_query(query)

    query_filter = None
    if source_filter:
        query_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_filter))])

    if hasattr(client, "search"):
        result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            limit=top_k,
        )
        return _normalize_search_result(result)

    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
        limit=top_k,
    )
    return _normalize_search_result(result)


def _search_primal(query: str, source_filter: str | None = None, top_k: int = DEFAULT_TOP_K):
    return _search_collection(
        collection_name=settings.qdrant_collection_primal,
        query=query,
        source_filter=source_filter,
        top_k=top_k,
    )


def _fetch_reference_chunks_from_collection(
    collection_name: str, source: str, article_number: str | None
) -> list[dict]:
    client = _client()
    must = [FieldCondition(key="source", match=MatchValue(value=source))]
    limit = SOURCE_EXPANSION_LIMIT
    if article_number:
        must.append(FieldCondition(key="article_number", match=MatchValue(value=article_number)))
        limit = 1

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(must=must),
        with_payload=True,
        with_vectors=False,
        limit=limit,
    )
    return [dict(point.payload or {}) for point in points]


def _fetch_reference_chunks(source: str, article_number: str | None) -> list[dict]:
    chunks: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for collection_name in REFERENCE_COLLECTIONS:
        for payload in _fetch_reference_chunks_from_collection(collection_name, source, article_number):
            key = _fallback_chunk_key(payload)
            if key in seen:
                continue
            seen.add(key)
            chunks.append(payload)
    return chunks


def retrieve_specific(
    query: str,
    source_filter: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_top_k: int | None = None,
) -> list[dict]:
    _prepare_embedding_memory()
    search_top_k = candidate_top_k or max(settings.retrieval_candidate_top_k, top_k)
    try:
        primary_points = list(_search_primal(query=query, source_filter=source_filter, top_k=search_top_k))
        if source_filter and settings.retrieval_soft_source_filter:
            primary_points.extend(_search_primal(query=query, source_filter=None, top_k=search_top_k))

        primary = _merge_points(primary_points)
        ranked_primary = _rerank(query, primary, top_k)
        dedup: dict[tuple[str, str, str, str], dict] = {}

        for payload in ranked_primary:
            dedup[_fallback_chunk_key(payload)] = payload

        for payload in ranked_primary:
            for reference in payload.get("references", []):
                source, article_number = parse_reference(reference)
                if not source:
                    continue
                for ref_payload in _fetch_reference_chunks(source=source, article_number=article_number):
                    key = _fallback_chunk_key(ref_payload)
                    if key in dedup:
                        continue
                    ref_payload.setdefault("score", None)
                    dedup[key] = ref_payload

        primary_keys = [_fallback_chunk_key(payload) for payload in ranked_primary]
        ordered = [dedup[key] for key in primary_keys if key in dedup]
        for key, payload in dedup.items():
            if key not in primary_keys:
                ordered.append(payload)
    finally:
        _release_embedding_memory()
    LOGGER.info(
        "Specific retrieval query=%r source_filter=%s candidates=%s primary=%s returned=%s sources=%s",
        query,
        source_filter,
        len(primary),
        len(ranked_primary),
        len(ordered),
        [item.get("source") for item in ordered[:10]],
    )
    return ordered
