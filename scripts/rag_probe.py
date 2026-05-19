"""Bi-encoder-only sanity check against the RAG indices.

Bypasses everything except the embedder + Qdrant similarity search — no
reranker, no reference-expansion, no LLM filter. Useful for answering
"is the embedder/index even returning the right ballpark?" without the
agent's noise.

Usage:
    python scripts/rag_probe.py "существенные условия договора"
    python scripts/rag_probe.py "..." --top-k 10
    python scripts/rag_probe.py "..." --collection primal
    python scripts/rag_probe.py "..." --collection all --show-text
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
APP_DIR = BASE_DIR if (BASE_DIR / "config.py").exists() else BASE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from config import settings  # noqa: E402
from rag.retriever import _payload_from_point, _search_collection  # noqa: E402


def _collection_names() -> dict[str, str]:
    return {
        "general": settings.qdrant_collection_general,
        "primal": settings.qdrant_collection_primal,
        "secondary": settings.qdrant_collection_secondary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure bi-encoder probe against Qdrant")
    parser.add_argument("query", help="Поисковый запрос (заключи в кавычки)")
    parser.add_argument("--top-k", type=int, default=5, help="Сколько чанков на коллекцию (default: 5)")
    parser.add_argument(
        "--collection",
        choices=["general", "primal", "secondary", "all"],
        default="all",
        help="Какую коллекцию опросить (default: все)",
    )
    parser.add_argument(
        "--show-text", action="store_true", help="Печатать первые 300 символов текста чанка"
    )
    parser.add_argument(
        "--source-filter",
        default=None,
        help="Точное имя источника для payload-фильтра (только primal/secondary)",
    )
    args = parser.parse_args()

    cols = _collection_names()
    targets = [args.collection] if args.collection != "all" else ["general", "primal", "secondary"]

    print(
        f"\nEmbedder: backend={settings.embedding_backend} model={settings.embed_model}  "
        f"dim={settings.embedding_dimensions}\n"
        f"Query:    {args.query!r}\n"
    )

    for name in targets:
        collection = cols[name]
        print(f"=== {name.upper()} ({collection}) ===")
        try:
            points = _search_collection(
                collection_name=collection,
                query=args.query,
                source_filter=source_filter,
                top_k=args.top_k,
            )
        except Exception as exc:
            print(f"  [error] {exc}")
            continue

        if not points:
            print("  (no results)")
            print()
            continue

        for point in points:
            payload = _payload_from_point(point)
            score = payload.get("score")
            score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
            source = payload.get("source") or "-"
            article = payload.get("article") or "-"
            chunk_id = payload.get("chunk_id") or "?"
            print(f"  [{score_str}] {source}  /  {article}  (chunk_id={chunk_id})")
            if args.show_text:
                text = (payload.get("text") or "").strip().replace("\n", " ")
                if len(text) > 300:
                    text = text[:300].rstrip() + "..."
                print(f"      {text}")
        print()


if __name__ == "__main__":
    main()
