from __future__ import annotations

import gc
import logging
from functools import lru_cache
from typing import Protocol

from config import settings
from langchain_ollama import OllamaEmbeddings

LOGGER = logging.getLogger("rag.embedding_provider")


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...
    def dimensions(self) -> int: ...


class OllamaEmbeddingProvider:
    def __init__(self) -> None:
        self._client = OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._client.embed_query(query)

    def dimensions(self) -> int:
        return settings.embedding_dimensions


class SentenceTransformerEmbeddingProvider:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        init_kwargs: dict[str, object] = {}
        if settings.embedding_device:
            init_kwargs["device"] = settings.embedding_device

        model_kwargs: dict[str, object] = {}
        config_kwargs: dict[str, object] = {}
        if settings.embedding_trust_remote_code:
            model_kwargs["trust_remote_code"] = True
            config_kwargs["trust_remote_code"] = True

        precision = settings.embedding_precision
        if precision and precision != "auto":
            try:
                import torch

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
                if precision in dtype_map:
                    model_kwargs["torch_dtype"] = dtype_map[precision]
                else:
                    LOGGER.warning("Unknown EMBEDDING_PRECISION=%s, ignoring", precision)
            except Exception as exc:
                LOGGER.warning("Failed to apply EMBEDDING_PRECISION=%s: %s", precision, exc)

        self._model = SentenceTransformer(
            settings.embed_model,
            model_kwargs=model_kwargs,
            config_kwargs=config_kwargs,
            **init_kwargs,
        )
        if settings.embedding_max_tokens > 0:
            self._model.max_seq_length = settings.embedding_max_tokens
        LOGGER.info(
            "Initialized SentenceTransformer embeddings model=%s requested_device=%s actual_device=%s",
            settings.embed_model,
            settings.embedding_device or "auto",
            getattr(self._model, "device", "unknown"),
        )

    def _normalize_inputs(self, texts: list[str]) -> list[str]:
        normalized: list[str] = []
        for text in texts:
            cleaned = (text or "").replace("\x00", " ").strip()
            normalized.append(cleaned)
        return normalized or [""]

    def _encode(self, texts: list[str], prompt: str | None = None) -> list[list[float]]:
        encoded = self._model.encode(
            self._normalize_inputs(texts),
            prompt=prompt,
            batch_size=settings.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return encoded.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, query: str) -> list[float]:
        prompt = None
        if not settings.embedding_query_instruction_disabled and settings.embedding_query_instruction:
            # E5/Instructor-style asymmetric prefix. SentenceTransformer.encode
            # prepends `prompt` to the actual text before tokenization, so the
            # final input becomes:
            #   Instruct: <task>\nQuery: <user query>
            # Documents in the index are encoded WITHOUT this prefix, which is
            # exactly what instruction-tuned encoders like Giga-Embeddings expect.
            prompt = f"Instruct: {settings.embedding_query_instruction.strip()}\nQuery: "
        return self._encode([(query or "").strip()], prompt=prompt)[0]

    def dimensions(self) -> int:
        dimension = self._model.get_sentence_embedding_dimension()
        if not dimension:
            raise ValueError("SentenceTransformer embedding dimension is unavailable")
        return int(dimension)


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    backend = settings.embedding_backend.strip().lower()
    if backend == "ollama":
        return OllamaEmbeddingProvider()
    if backend in {"sentence_transformers", "sentence-transformer", "huggingface", "hf"}:
        return SentenceTransformerEmbeddingProvider()
    raise ValueError(f"Unsupported embedding backend: {settings.embedding_backend}")


def clear_embedding_provider() -> None:
    """Release the cached embedding model and CUDA allocator cache."""
    get_embedding_provider.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        LOGGER.debug("Failed to clear CUDA embedding cache: %s", exc)


def preload_embedding_provider() -> None:
    provider = get_embedding_provider()
    dimensions = provider.dimensions()
    if settings.embedding_query_instruction_disabled:
        instruction_status = "disabled"
    elif settings.embedding_query_instruction:
        truncated = settings.embedding_query_instruction.strip()
        if len(truncated) > 100:
            truncated = truncated[:97] + "..."
        instruction_status = f"enabled: {truncated!r}"
    else:
        instruction_status = "none"
    actual_dtype: object = "n/a"
    try:
        if isinstance(provider, SentenceTransformerEmbeddingProvider):
            params = provider._model.parameters()
            actual_dtype = next(params).dtype
    except Exception:
        LOGGER.debug("Failed to probe embedding model dtype", exc_info=True)
    LOGGER.info(
        "Preloaded embedding provider backend=%s model=%s dimensions=%s dtype=%s query_instruction=%s",
        settings.embedding_backend,
        settings.embed_model,
        dimensions,
        actual_dtype,
        instruction_status,
    )
