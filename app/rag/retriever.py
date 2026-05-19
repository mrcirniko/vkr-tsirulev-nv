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


def _reranker_model_kwargs() -> dict:
    """Translate settings.reranker_precision into model_kwargs={'torch_dtype': ...}.

    Returns an empty dict for "auto" (let transformers pick — usually fp32).
    Unknown values log a warning and fall through to auto.
    """
    precision = settings.reranker_precision
    if not precision or precision == "auto":
        return {}
    try:
        import torch
    except Exception as exc:
        LOGGER.warning("torch unavailable, ignoring RERANKER_PRECISION=%s: %s", precision, exc)
        return {}
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "full": torch.float32,
    }
    dtype = dtype_map.get(precision)
    if dtype is None:
        LOGGER.warning("Unknown RERANKER_PRECISION=%s, ignoring", precision)
        return {}
    return {"torch_dtype": dtype}


def _log_reranker_actual(model) -> None:
    """Log the device and dtype the cross-encoder actually loaded with —
    `device=auto` in the request log doesn't tell you what was picked."""
    try:
        params = next(model.model.parameters())
        LOGGER.info("Reranker loaded actual_device=%s actual_dtype=%s", params.device, params.dtype)
    except Exception:
        LOGGER.debug("Failed to probe reranker device/dtype", exc_info=True)


def _is_qwen3_reranker(model_id: str) -> bool:
    """Detect Qwen3-Reranker by its HF model id. We can't load it via the
    sentence-transformers CrossEncoder shim (CausalLM head, no score.weight),
    so we route to the custom path defined below."""
    return "qwen3-reranker" in (model_id or "").lower()


class _Qwen3Reranker:
    """Custom inference for Qwen3-Reranker following the "Using Transformers"
    recipe from the model card. Mimics `CrossEncoder.predict(pairs, batch_size=...)`
    so the rest of the retrieval pipeline doesn't care which backend is in use.

    Scoring: each (query, doc) pair is wrapped in a chat-template prompt
    that asks the model to answer "yes" or "no". We take the final-position
    logits, compute log_softmax over [no, yes] and return P(yes) as the
    relevance score (in [0, 1]).
    """

    _PREFIX = (
        '<|im_start|>system\nJudge whether the Document meets the requirements '
        'based on the Query and the Instruct provided. Note that the answer can '
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model_id: str, device: str | None = None, dtype=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs: dict = {}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        # Direct device load avoids transient CPU→GPU duplication that doubles peak RAM.
        if device:
            load_kwargs["device_map"] = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
        if device and next(self.model.parameters()).device.type != device.split(":")[0]:
            self.model = self.model.to(device)

        self.token_yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_tokens = self.tokenizer.encode(self._PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self._SUFFIX, add_special_tokens=False)
        self.task = settings.reranker_task_instruction
        self.max_length = max(512, int(settings.reranker_max_length))

    def _format(self, query: str, doc: str) -> str:
        return f"<Instruct>: {self.task}\n<Query>: {query}\n<Document>: {doc}"

    def predict(self, pairs: list[list[str]] | list[tuple[str, str]], batch_size: int = 8, **_kwargs):
        """Returns a list of float scores in the order of `pairs`. Signature
        mirrors sentence-transformers CrossEncoder.predict so the existing
        `_rerank()` call site works unchanged."""
        import torch

        if not pairs:
            return []
        scores: list[float] = []
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        for start in range(0, len(pairs), max(1, batch_size)):
            batch = pairs[start : start + batch_size]
            prompts = [self._format(q, d) for q, d in batch]
            inputs = self.tokenizer(
                prompts,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=budget,
            )
            for i, ids in enumerate(inputs["input_ids"]):
                inputs["input_ids"][i] = self.prefix_tokens + ids + self.suffix_tokens
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]
            yes_logits = logits[:, self.token_yes_id]
            no_logits = logits[:, self.token_no_id]
            stacked = torch.stack([no_logits, yes_logits], dim=1)
            probs = torch.nn.functional.log_softmax(stacked, dim=1)
            scores.extend(probs[:, 1].exp().cpu().tolist())
        return scores


@lru_cache(maxsize=1)
def _reranker():
    if not settings.reranker_enabled:
        return None

    requested_device = settings.reranker_device or None
    model_kwargs = _reranker_model_kwargs()
    LOGGER.info(
        "Loading reranker model=%s device=%s precision=%s",
        settings.reranker_model,
        requested_device or "auto",
        settings.reranker_precision or "auto",
    )

    # Qwen3-Reranker is a CausalLM; sentence-transformers CrossEncoder would attach a random
    # score head and produce garbage. Route to a custom yes/no-logit path per the model card.
    if _is_qwen3_reranker(settings.reranker_model):
        try:
            dtype = model_kwargs.get("torch_dtype") if model_kwargs else None
            model = _Qwen3Reranker(settings.reranker_model, device=requested_device, dtype=dtype)
            _log_reranker_actual(model)
            return model
        except Exception:
            LOGGER.exception("Qwen3-Reranker load failed; disabling reranker for this session")
            return None

    from sentence_transformers import CrossEncoder

    try:
        kwargs: dict = {}
        if requested_device:
            kwargs["device"] = requested_device
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        model = CrossEncoder(settings.reranker_model, **kwargs)
        _log_reranker_actual(model)
        return model
    except Exception as exc:
        # CUDA OOM fallback to CPU — preserves precision (bf16/fp16 work on CPU too, just slower).
        message = str(exc).lower()
        if requested_device and ("cuda" in requested_device.lower() or "out of memory" in message):
            LOGGER.warning(
                "Reranker failed to load on %s (%s) — falling back to CPU",
                requested_device,
                exc,
            )
            cpu_kwargs: dict = {"device": "cpu"}
            if model_kwargs:
                cpu_kwargs["model_kwargs"] = model_kwargs
            model = CrossEncoder(settings.reranker_model, **cpu_kwargs)
            _log_reranker_actual(model)
            return model
        LOGGER.exception("Reranker load failed; disabling reranker for this session")
        return None


def _chunk_rerank_text(chunk: dict) -> str:
    # Mirror indexer's heading prefix so the cross-encoder sees the same context as the bi-encoder.
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


def search_primal_bi_encoder(
    query: str, source_filter: str | None = None, top_k: int = DEFAULT_TOP_K
) -> list[dict]:
    """Bi-encoder-only PRIMAL search — no reranker, no reference expansion.

    Used by the iterative retrieval flow in agent/nodes.py: each loop
    iteration wants raw similarity results to pass to the strict-judge LLM,
    and final reranking happens once at the end (over the full accumulated
    set, including reference-expanded SECONDARY chunks).
    """
    _prepare_embedding_memory()
    try:
        return _merge_points(_search_primal(query=query, source_filter=source_filter, top_k=top_k))
    finally:
        _release_embedding_memory()


def fetch_reference_chunks_by_payload(source: str, article_number: str | None) -> list[dict]:
    """Public alias for `_fetch_reference_chunks` — scrolls SECONDARY by
    (source, article_number) without similarity. Exposed for the iterative
    retrieval flow's reference-expansion step.
    """
    return _fetch_reference_chunks(source=source, article_number=article_number)


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Public alias for `_rerank` — cross-encoder pass over a list of chunks.
    Used by the iterative retrieval flow's final rerank step over the
    union (approved + reference-expanded)."""
    return _rerank(query, chunks, top_k)


def _fetch_reference_chunks_from_collection(
    collection_name: str, source: str, article_number: str | None
) -> list[dict]:
    client = _client()
    must = [FieldCondition(key="source", match=MatchValue(value=source))]
    if article_number:
        must.append(FieldCondition(key="article_number", match=MatchValue(value=article_number)))
        # Long articles split across chunks — pull all without abusing the index.
        limit = settings.retrieval_reference_article_chunk_cap
    else:
        # Bare-source reference: representative sample, not a deep dive.
        limit = SOURCE_EXPANSION_LIMIT

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


def _collect_next_hop_chunks(
    seed_chunks: list[dict],
    seen_keys: set[tuple[str, str, str, str]],
    chunk_cap: int,
) -> list[dict]:
    """Walk the `references` payload on each seed and return new chunks.

    Bounds fan-out via `chunk_cap` so a high-degree node in the citation
    graph doesn't dominate the candidate set. Mutates `seen_keys` so the
    caller's running dedup stays consistent across hops.
    """
    collected: list[dict] = []
    for seed in seed_chunks:
        if len(collected) >= chunk_cap:
            break
        for reference in seed.get("references") or []:
            if len(collected) >= chunk_cap:
                break
            source, article_number = parse_reference(reference)
            if not source:
                continue
            for payload in _fetch_reference_chunks(source=source, article_number=article_number):
                key = _fallback_chunk_key(payload)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                collected.append(payload)
                if len(collected) >= chunk_cap:
                    break
    return collected


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
        primary = _merge_points(primary_points)
        ranked_primary = _rerank(query, primary, top_k)

        # Reference-graph expansion: walk citations for up to max_hops, per-hop cap on fan-out.
        seen_keys: set[tuple[str, str, str, str]] = {_fallback_chunk_key(p) for p in ranked_primary}
        all_references: list[dict] = []
        seed = ranked_primary
        for _hop in range(max(0, settings.retrieval_reference_max_hops)):
            hop_chunks = _collect_next_hop_chunks(
                seed_chunks=seed,
                seen_keys=seen_keys,
                chunk_cap=settings.retrieval_reference_hop_chunk_cap,
            )
            if not hop_chunks:
                break
            for chunk in hop_chunks:
                chunk.setdefault("score", None)
            all_references.extend(hop_chunks)
            seed = hop_chunks

        # Joint rerank so a referenced statute can outrank a weak primary tail entry.
        union = list(ranked_primary) + all_references
        final_top_k = max(top_k, settings.retrieval_specific_expanded_top_k)
        ordered = _rerank(query, union, final_top_k)
    finally:
        _release_embedding_memory()
    LOGGER.info(
        "Specific retrieval query=%r source_filter=%s candidates=%s primary=%s references=%s returned=%s sources=%s",
        query,
        source_filter,
        len(primary),
        len(ranked_primary),
        len(all_references),
        len(ordered),
        [item.get("source") for item in ordered[:10]],
    )
    return ordered
