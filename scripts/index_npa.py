# from __future__ import annotations

# import argparse
# import json
# import logging
# import os
# import sys
# from pathlib import Path

# from qdrant_client import QdrantClient

# BASE_DIR = Path(__file__).resolve().parents[1]
# APP_DIR = BASE_DIR if (BASE_DIR / "config.py").exists() else BASE_DIR / "app"
# if str(APP_DIR) not in sys.path:
#     sys.path.insert(0, str(APP_DIR))

# # Indexing can generate a large number of internal LLM calls while extracting references.
# # Disable LangSmith tracing here so routine reindexing does not consume tracing quota.
# os.environ["LANGSMITH_TRACING"] = "false"
# os.environ["LANGCHAIN_TRACING_V2"] = "false"
# os.environ.pop("LANGSMITH_API_KEY", None)
# os.environ.pop("LANGCHAIN_API_KEY", None)

# from config import settings
# from rag.indexer import (
#     GENERAL_GROUP,
#     PRIMAL_GROUP,
#     SECONDARY_GROUP,
#     SPECIFIC_GROUP,
#     export_chunks,
#     index_chunks_file,
#     index_documents,
#     preview_chunks,
# )
# from rag.retriever import retrieve_general, retrieve_specific

# DEFAULT_DATA_DIR = BASE_DIR / "data" / "npa"
# LOGGER = logging.getLogger("index_npa")
# REFERENCE_EXTRACTION_MODES = ["llm", "morphology", "hybrid"]
# PREVIEW_GROUPS = [GENERAL_GROUP, PRIMAL_GROUP, SECONDARY_GROUP, SPECIFIC_GROUP]


# def configure_logging(verbose: bool) -> None:
#     level = logging.INFO if verbose else logging.WARNING
#     logging.basicConfig(
#         level=level,
#         format="[%(levelname)s] %(name)s: %(message)s",
#         stream=sys.stderr,
#         force=True,
#     )


# def _collection_points_count(client: QdrantClient, collection_name: str) -> int:
#     try:
#         info = client.get_collection(collection_name)
#         return int(getattr(info, "points_count", 0) or 0)
#     except Exception:
#         return 0


# def _ensure_reindex_confirmation(force_recreate: bool, auto_yes: bool) -> bool:
#     client = QdrantClient(url=settings.qdrant_url)
#     general_count = _collection_points_count(client, settings.qdrant_collection_general)
#     primal_count = _collection_points_count(client, settings.qdrant_collection_primal)
#     secondary_count = _collection_points_count(client, settings.qdrant_collection_secondary)
#     total = general_count + primal_count + secondary_count
#     if total == 0:
#         return force_recreate
#     if force_recreate or auto_yes:
#         return True

#     print(
#         "Коллекции уже содержат данные: "
#         f"{settings.qdrant_collection_general}={general_count}, "
#         f"{settings.qdrant_collection_primal}={primal_count}, "
#         f"{settings.qdrant_collection_secondary}={secondary_count}."
#     )
#     answer = input("Переиндексировать и очистить все коллекции? [y/N]: ").strip().lower()
#     return answer in {"y", "yes", "д", "да"}


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Utilities for normative base indexing and retrieval")
#     parser.add_argument("--verbose", action="store_true", help="Enable progress logging to stderr")
#     subparsers = parser.add_subparsers(dest="command", required=True)

#     index_parser = subparsers.add_parser("index", help="Index all normative documents into Qdrant")
#     index_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
#     index_parser.add_argument("--recreate", action="store_true")
#     index_parser.add_argument("--yes", action="store_true", help="Skip confirmation before collection cleanup")
#     index_parser.add_argument("--reference-extraction-mode", choices=REFERENCE_EXTRACTION_MODES, default="llm")

#     export_parser = subparsers.add_parser("export-chunks", help="Chunk normative documents into a JSON file without embedding or Qdrant writes")
#     export_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
#     export_parser.add_argument("--output", default=str(BASE_DIR / "data" / "chunks" / "npa_chunks.json"))
#     export_parser.add_argument("--reference-extraction-mode", choices=REFERENCE_EXTRACTION_MODES, default="llm")

#     index_chunks_parser = subparsers.add_parser("index-chunks", help="Index a previously exported chunks JSON file into Qdrant")
#     index_chunks_parser.add_argument("--chunks-path", default=str(BASE_DIR / "data" / "chunks" / "npa_chunks.json"))
#     index_chunks_parser.add_argument("--recreate", action="store_true")
#     index_chunks_parser.add_argument("--yes", action="store_true", help="Skip confirmation before collection cleanup")

#     preview_parser = subparsers.add_parser("preview", help="Preview chunking without Qdrant")
#     preview_parser.add_argument("group", choices=PREVIEW_GROUPS)
#     preview_parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
#     preview_parser.add_argument("--limit", type=int, default=3, help="How many chunks to create in total")
#     preview_parser.add_argument("--only-with-references", action="store_true")
#     preview_parser.add_argument("--reference-extraction-mode", choices=REFERENCE_EXTRACTION_MODES, default="llm")

#     q_general = subparsers.add_parser("query-general", help="Search general chunks in Qdrant, or read all when query is omitted")
#     q_general.add_argument("query", nargs="?")
#     q_general.add_argument("--limit", type=int, default=5)

#     q_specific = subparsers.add_parser("query-specific", help="Search primal chunks in Qdrant and expand references into primal and secondary chunks")
#     q_specific.add_argument("query")
#     q_specific.add_argument("--source-filter")
#     q_specific.add_argument("--top-k", type=int, default=7)

#     args = parser.parse_args()
#     configure_logging(args.verbose)

#     if args.command == "index":
#         recreate = _ensure_reindex_confirmation(args.recreate, args.yes)
#         LOGGER.info(
#             "Starting indexing: data_dir=%s recreate=%s extraction_mode=%s",
#             args.data_dir,
#             recreate,
#             args.reference_extraction_mode,
#         )
#         stats = index_documents(
#             args.data_dir,
#             recreate=recreate,
#             extraction_mode=args.reference_extraction_mode,
#         )
#         LOGGER.info("Indexing finished: %s", stats)
#         print(json.dumps(stats, ensure_ascii=False, indent=2))
#         return

#     if args.command == "export-chunks":
#         LOGGER.info(
#             "Exporting chunks: data_dir=%s output=%s extraction_mode=%s",
#             args.data_dir,
#             args.output,
#             args.reference_extraction_mode,
#         )
#         stats = export_chunks(
#             args.data_dir,
#             output_path=args.output,
#             extraction_mode=args.reference_extraction_mode,
#         )
#         LOGGER.info("Chunk export finished: %s", stats)
#         print(json.dumps(stats, ensure_ascii=False, indent=2))
#         return

#     if args.command == "index-chunks":
#         recreate = _ensure_reindex_confirmation(args.recreate, args.yes)
#         LOGGER.info("Indexing exported chunks: chunks_path=%s recreate=%s", args.chunks_path, recreate)
#         stats = index_chunks_file(args.chunks_path, recreate=recreate)
#         LOGGER.info("Chunk indexing finished: %s", stats)
#         print(json.dumps(stats, ensure_ascii=False, indent=2))
#         return

#     if args.command == "preview":
#         LOGGER.info(
#             "Preview started: group=%s data_dir=%s chunk_limit=%s extraction_mode=%s",
#             args.group,
#             args.data_dir,
#             args.limit,
#             args.reference_extraction_mode,
#         )
#         chunks = preview_chunks(
#             args.data_dir,
#             args.group,
#             limit=args.limit,
#             extraction_mode=args.reference_extraction_mode,
#         )
#         LOGGER.info("Preview created %s chunks before filtering", len(chunks))
#         if args.only_with_references:
#             chunks = [chunk for chunk in chunks if chunk.get("references")]
#             LOGGER.info("After --only-with-references: %s chunks", len(chunks))
#         print(json.dumps(chunks, ensure_ascii=False, indent=2))
#         return

#     if args.command == "query-general":
#         LOGGER.info("Querying general collection query=%r limit=%s", args.query, args.limit)
#         print(json.dumps(retrieve_general(args.query, top_k=args.limit)[: args.limit], ensure_ascii=False, indent=2))
#         return

#     if args.command == "query-specific":
#         LOGGER.info(
#             "Querying primal collection with reference expansion query=%r source_filter=%r top_k=%s",
#             args.query,
#             args.source_filter,
#             args.top_k,
#         )
#         result = retrieve_specific(args.query, source_filter=args.source_filter, top_k=args.top_k)
#         LOGGER.info("Query returned %s chunks", len(result))
#         print(json.dumps(result, ensure_ascii=False, indent=2))
#         return


# if __name__ == "__main__":
#     main()
