from __future__ import annotations

import argparse
import json
import logging
import math
import uuid
from pathlib import Path

from config import settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag.chunker import chunk_document
from rag.embedding_provider import get_embedding_provider
from rag.source_registry import canonical_source_name

BATCH_SIZE = 100
EMBED_SUBCHUNK_MAX_CHARS = 1400
EMBED_SUBCHUNK_MAX_COUNT = 3
PROGRESS_TICK_INTERVAL = 0.15
GENERAL_GROUP = "general"
PRIMAL_GROUP = "primal"
SECONDARY_GROUP = "secondary"
SPECIFIC_GROUP = "specific"
LOGGER = logging.getLogger("rag.indexer")


def _chunk_heading(chunk: dict) -> str:
    parts = [
        part.strip() for part in (chunk.get("source") or "", chunk.get("article") or "") if part and str(part).strip()
    ]
    return "\n".join(parts)


def _split_text_for_embedding(
    text: str,
    max_chars: int = EMBED_SUBCHUNK_MAX_CHARS,
    max_count: int = EMBED_SUBCHUNK_MAX_COUNT,
) -> list[str]:
    text = text.strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [text]

    subchunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 2 > max_chars:
            subchunks.append("\n\n".join(current).strip())
            if len(subchunks) >= max_count:
                return subchunks
            current = []
            current_len = 0

        if len(paragraph) > max_chars:
            start = 0
            while start < len(paragraph):
                remaining_slots = max_count - len(subchunks)
                if remaining_slots <= 0:
                    return subchunks
                piece = paragraph[start : start + max_chars].strip()
                if piece:
                    if current:
                        subchunks.append("\n\n".join(current).strip())
                        if len(subchunks) >= max_count:
                            return subchunks
                        current = []
                        current_len = 0
                    subchunks.append(piece)
                    if len(subchunks) >= max_count:
                        return subchunks
                start += max_chars
            continue

        current.append(paragraph)
        current_len += len(paragraph) + 2

    if current and len(subchunks) < max_count:
        subchunks.append("\n\n".join(current).strip())

    return subchunks[:max_count]


def _texts_for_chunk_embedding(chunk: dict) -> list[str]:
    heading = _chunk_heading(chunk)
    body = (chunk.get("text") or "").strip()
    body_subchunks = _split_text_for_embedding(body)
    texts: list[str] = []
    for subchunk in body_subchunks:
        payload = f"{heading}\n{subchunk}".strip() if heading else subchunk
        texts.append(payload)
    return texts or ([heading] if heading else [""])


def _average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("No vectors to average")
    if len(vectors) == 1:
        return vectors[0]

    size = len(vectors[0])
    acc = [0.0] * size
    for vector in vectors:
        if len(vector) != size:
            raise ValueError("Inconsistent vector sizes")
        for idx, value in enumerate(vector):
            acc[idx] += float(value)

    count = float(len(vectors))
    averaged = [value / count for value in acc]
    norm = math.sqrt(sum(value * value for value in averaged))
    if norm > 0:
        averaged = [value / norm for value in averaged]
    return averaged


class NormIndexer:
    def __init__(self, qdrant_url: str | None = None) -> None:
        self.client = QdrantClient(url=qdrant_url or settings.qdrant_url)
        self.embeddings = get_embedding_provider()

    def ensure_collection(self, collection_name: str) -> None:
        existing = {item.name for item in self.client.get_collections().collections}
        if collection_name in existing:
            LOGGER.info("Collection already exists: %s", collection_name)
            return
        LOGGER.info(
            "Creating collection: %s (embedding_backend=%s, size=%s)",
            collection_name,
            settings.embedding_backend,
            settings.embedding_dimensions,
        )
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=settings.embedding_dimensions, distance=Distance.COSINE),
        )

    def recreate_collection(self, collection_name: str) -> None:
        existing = {item.name for item in self.client.get_collections().collections}
        if collection_name in existing:
            LOGGER.info("Deleting collection before recreate: %s", collection_name)
            self.client.delete_collection(collection_name=collection_name)
        self.ensure_collection(collection_name)

    def _embed_chunk(self, chunk: dict) -> list[float]:
        texts = _texts_for_chunk_embedding(chunk)
        LOGGER.info(
            "Embedding chunk via %s subchunks: source=%s article=%s text_len=%s backend=%s model=%s",
            len(texts),
            chunk.get("source"),
            chunk.get("article"),
            len(chunk.get("text") or ""),
            settings.embedding_backend,
            settings.embed_model,
        )
        vectors = self.embeddings.embed_documents(texts)
        return _average_vectors(vectors)

    def index_chunks(
        self,
        collection_name: str,
        chunks: list[dict],
        progress_cb=None,
    ) -> int:
        if not chunks:
            LOGGER.info("No chunks to index for collection: %s", collection_name)
            return 0

        import time

        total = len(chunks)
        # Throttle progress emissions: WS events arriving faster than the
        # browser repaints (~16ms) result in only the LAST one being painted,
        # so the bar appeared to jump straight to 100%. 150ms gives ~6 visible
        # frames per second — smooth enough for the eye, sparse enough for the
        # WS channel. The 0% (initial) and 100% (final) ticks are always sent.
        last_emit = 0.0

        def _tick(done: int, *, force: bool = False) -> None:
            nonlocal last_emit
            if progress_cb is None:
                return
            now = time.monotonic()
            if not force and (now - last_emit) < PROGRESS_TICK_INTERVAL:
                return
            last_emit = now
            try:
                progress_cb(done, total)
            except Exception:
                LOGGER.exception("progress_cb failed; continuing indexing")

        # Initial 0/total tick so the UI bar appears immediately at "starting".
        _tick(0, force=True)

        inserted = 0
        for start in range(0, total, BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            LOGGER.info("Embedding batch for %s: start=%s size=%s", collection_name, start, len(batch))
            vectors: list[list[float]] = []
            for offset, chunk in enumerate(batch):
                vectors.append(self._embed_chunk(chunk))
                _tick(start + offset + 1)
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=chunk,
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(collection_name=collection_name, points=points)
            inserted += len(points)
            LOGGER.info("Uploaded batch to %s, total_inserted=%s", collection_name, inserted)
            _tick(inserted, force=True)
        return inserted

    def delete_by_source(self, collection_name: str, source_name: str) -> None:
        """Drop every point in the collection whose payload.source matches.

        Called before re-indexing a single NPA so the new points don't pile
        up alongside the old ones (they would, since point IDs are random
        UUIDs — there's no natural upsert key).
        """
        existing = {item.name for item in self.client.get_collections().collections}
        if collection_name not in existing:
            return
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(
                    filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_name))])
                ),
            )
            LOGGER.info("Deleted points for source=%s in collection=%s", source_name, collection_name)
        except Exception:
            LOGGER.exception("delete_by_source failed for source=%s collection=%s", source_name, collection_name)


def resolve_source(filename: str) -> str:
    return canonical_source_name(filename)


def _specific_directories(root: Path) -> dict[str, Path]:
    specific_root = root / SPECIFIC_GROUP
    primal_dir = specific_root / PRIMAL_GROUP
    secondary_dir = specific_root / SECONDARY_GROUP
    directories: dict[str, Path] = {}

    if primal_dir.exists():
        directories[PRIMAL_GROUP] = primal_dir
    elif specific_root.exists():
        directories[PRIMAL_GROUP] = specific_root

    if secondary_dir.exists():
        directories[SECONDARY_GROUP] = secondary_dir

    return directories


def _directory_for_group(root: Path, source_group: str) -> Path:
    if source_group == GENERAL_GROUP:
        return root / GENERAL_GROUP
    if source_group == SPECIFIC_GROUP:
        return _specific_directories(root).get(PRIMAL_GROUP, root / SPECIFIC_GROUP)
    directories = _specific_directories(root)
    if source_group in directories:
        return directories[source_group]
    raise ValueError(f"Unsupported source group: {source_group}")


def _count_txt_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob("*.txt"))


def _corpus_directories(root: Path) -> tuple[Path, Path, Path]:
    general_dir = root / GENERAL_GROUP
    specific_dirs = _specific_directories(root)
    primal_dir = specific_dirs.get(PRIMAL_GROUP, root / SPECIFIC_GROUP)
    secondary_dir = specific_dirs.get(SECONDARY_GROUP, root / SPECIFIC_GROUP / SECONDARY_GROUP)
    return general_dir, primal_dir, secondary_dir


def load_chunks_from_dir(
    directory: Path,
    is_general: bool,
    limit: int | None = None,
    extraction_mode: str = "llm",
) -> list[dict]:
    LOGGER.info("Loading .txt files from %s", directory)
    chunks: list[dict] = []
    if not directory.exists():
        LOGGER.info("Directory does not exist, skipping: %s", directory)
        return chunks

    for path in sorted(directory.rglob("*.txt")):
        if limit is not None and len(chunks) >= limit:
            LOGGER.info("Chunk limit reached before reading next file: limit=%s", limit)
            break
        LOGGER.info("Reading file: %s", path)
        text = path.read_text(encoding="utf-8")
        source = resolve_source(path.name)
        remaining = None if limit is None else max(limit - len(chunks), 0)
        if remaining == 0:
            LOGGER.info("Chunk limit reached before chunking file body: limit=%s", limit)
            break
        file_chunks = chunk_document(
            text=text,
            source=source,
            is_general=is_general,
            limit=remaining,
            extraction_mode=extraction_mode,
        )
        LOGGER.info("Chunked file %s into %s chunks", path.name, len(file_chunks))
        chunks.extend(file_chunks)
    LOGGER.info("Directory %s produced %s chunks", directory, len(chunks))
    return chunks


def index_documents(data_dir: str, recreate: bool = False, extraction_mode: str = "llm") -> dict[str, int | str]:
    root = Path(data_dir)
    general_dir, primal_dir, secondary_dir = _corpus_directories(root)

    indexer = NormIndexer()
    prepare = indexer.recreate_collection if recreate else indexer.ensure_collection
    prepare(settings.qdrant_collection_general)
    prepare(settings.qdrant_collection_primal)
    prepare(settings.qdrant_collection_secondary)

    general_chunks = load_chunks_from_dir(general_dir, is_general=True, extraction_mode=extraction_mode)
    primal_chunks = load_chunks_from_dir(primal_dir, is_general=False, extraction_mode=extraction_mode)
    secondary_chunks = load_chunks_from_dir(secondary_dir, is_general=False, extraction_mode=extraction_mode)

    general_count = indexer.index_chunks(settings.qdrant_collection_general, general_chunks)
    primal_count = indexer.index_chunks(settings.qdrant_collection_primal, primal_chunks)
    secondary_count = indexer.index_chunks(settings.qdrant_collection_secondary, secondary_chunks)

    return {
        "general_files": _count_txt_files(general_dir),
        "primal_files": _count_txt_files(primal_dir),
        "secondary_files": _count_txt_files(secondary_dir),
        "general_chunks": general_count,
        "primal_chunks": primal_count,
        "secondary_chunks": secondary_count,
        "reference_extraction_mode": extraction_mode,
    }


def export_chunks(data_dir: str, output_path: str, extraction_mode: str = "llm") -> dict[str, int | str]:
    root = Path(data_dir)
    general_dir, primal_dir, secondary_dir = _corpus_directories(root)

    payload = {
        "metadata": {
            "data_dir": str(root),
            "reference_extraction_mode": extraction_mode,
        },
        GENERAL_GROUP: load_chunks_from_dir(general_dir, is_general=True, extraction_mode=extraction_mode),
        PRIMAL_GROUP: load_chunks_from_dir(primal_dir, is_general=False, extraction_mode=extraction_mode),
        SECONDARY_GROUP: load_chunks_from_dir(secondary_dir, is_general=False, extraction_mode=extraction_mode),
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_path": str(path),
        "general_files": _count_txt_files(general_dir),
        "primal_files": _count_txt_files(primal_dir),
        "secondary_files": _count_txt_files(secondary_dir),
        "general_chunks": len(payload[GENERAL_GROUP]),
        "primal_chunks": len(payload[PRIMAL_GROUP]),
        "secondary_chunks": len(payload[SECONDARY_GROUP]),
        "reference_extraction_mode": extraction_mode,
    }


def index_chunks_file(chunks_path: str, recreate: bool = False) -> dict[str, int | str]:
    path = Path(chunks_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    indexer = NormIndexer()
    prepare = indexer.recreate_collection if recreate else indexer.ensure_collection
    prepare(settings.qdrant_collection_general)
    prepare(settings.qdrant_collection_primal)
    prepare(settings.qdrant_collection_secondary)

    general_chunks = payload.get(GENERAL_GROUP) or []
    primal_chunks = payload.get(PRIMAL_GROUP) or []
    secondary_chunks = payload.get(SECONDARY_GROUP) or []

    general_count = indexer.index_chunks(settings.qdrant_collection_general, general_chunks)
    primal_count = indexer.index_chunks(settings.qdrant_collection_primal, primal_chunks)
    secondary_count = indexer.index_chunks(settings.qdrant_collection_secondary, secondary_chunks)

    return {
        "chunks_path": str(path),
        "general_chunks": general_count,
        "primal_chunks": primal_count,
        "secondary_chunks": secondary_count,
        "reference_extraction_mode": (payload.get("metadata") or {}).get("reference_extraction_mode", "unknown"),
    }


def chunk_text(
    text: str,
    source_name: str,
    source_group: str,
    extraction_mode: str = "morphology",
) -> list[dict]:
    """Chunk a single in-memory text without touching Qdrant.

    Used by the admin UI: convert TXT -> chunks -> JSON, all in
    one upload request, so the operator can download and review the JSON
    before any embedding work happens.
    """
    is_general = source_group == GENERAL_GROUP
    return chunk_document(
        text=text,
        source=source_name,
        is_general=is_general,
        extraction_mode=extraction_mode,
    )


def index_chunks_for_source(
    *,
    collection_name: str,
    chunks: list[dict],
    source_name: str,
    recreate_collection: bool = False,
    delete_existing_for_source: bool = True,
    progress_cb=None,
) -> int:
    """Embed `chunks` and upsert into `collection_name`.

    Combines collection lifecycle + per-source cleanup so the admin route
    just hands chunks over without juggling Qdrant directly.

    - recreate_collection=True wipes the entire collection (kills other
      sources of the same group). Use with care; UI must confirm.
    - delete_existing_for_source=True (default) drops only this source's
      old points first so re-indexing one NPA is safe and idempotent.
    """
    indexer = NormIndexer()
    if recreate_collection:
        indexer.recreate_collection(collection_name)
    else:
        indexer.ensure_collection(collection_name)
        if delete_existing_for_source:
            indexer.delete_by_source(collection_name, source_name)

    return indexer.index_chunks(collection_name, chunks, progress_cb=progress_cb)


def preview_chunks(data_dir: str, source_group: str, limit: int = 3, extraction_mode: str = "llm") -> list[dict]:
    root = Path(data_dir)
    normalized_group = PRIMAL_GROUP if source_group == SPECIFIC_GROUP else source_group
    is_general = normalized_group == GENERAL_GROUP
    directory = _directory_for_group(root, normalized_group)
    chunks = load_chunks_from_dir(directory, is_general=is_general, limit=limit, extraction_mode=extraction_mode)
    LOGGER.info("Preview created %s chunks total", len(chunks))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Index normative documents into Qdrant")
    parser.add_argument("data_dir", nargs="?", default="data/npa")
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collections before indexing")
    parser.add_argument(
        "--preview",
        choices=[GENERAL_GROUP, PRIMAL_GROUP, SECONDARY_GROUP, SPECIFIC_GROUP],
        help="Preview chunking without indexing",
    )
    parser.add_argument("--limit", type=int, default=3, help="Preview chunk limit")
    parser.add_argument("--reference-extraction-mode", choices=["llm", "morphology", "hybrid"], default="llm")
    args = parser.parse_args()

    if args.preview:
        preview_chunks(args.data_dir, args.preview, limit=args.limit, extraction_mode=args.reference_extraction_mode)
        return

    index_documents(args.data_dir, recreate=args.recreate, extraction_mode=args.reference_extraction_mode)


if __name__ == "__main__":
    main()
