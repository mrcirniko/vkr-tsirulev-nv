from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger("app.config")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_embedding_dimensions() -> int:
    explicit = os.getenv("EMBEDDING_DIMENSIONS")
    if explicit:
        return int(explicit)

    backend = os.getenv("EMBEDDING_BACKEND", "ollama").strip().lower()
    model = os.getenv("EMBED_MODEL", "nomic-embed-text")
    if (
        backend in {"sentence_transformers", "sentence-transformer", "huggingface", "hf"}
        and model == "ai-sage/Giga-Embeddings-instruct"
    ):
        return 2048
    if (
        backend in {"sentence_transformers", "sentence-transformer", "huggingface", "hf"}
        and model == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        return 384
    return 768


@dataclass(frozen=True)
class Settings:
    postgres_user: str = os.getenv("POSTGRES_USER", "user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "password")
    postgres_db: str = os.getenv("POSTGRES_DB", "contracts_db")
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@postgres:5432/contracts_db",
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:14b")
    llm_reasoning: bool = os.getenv("LLM_REASONING", "false").strip().lower() in {"1", "true", "yes", "on"}
    llm_num_ctx: int = int(os.getenv("LLM_NUM_CTX", "131072"))
    llm_num_predict: int = int(os.getenv("LLM_NUM_PREDICT", "8192"))
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "0")
    memory_swap_mode: bool = os.getenv("MEMORY_SWAP_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
    reference_llm_model: str = os.getenv("REFERENCE_LLM_MODEL", "qwen2.5:0.5b")
    reference_llm_base_url: str = os.getenv(
        "REFERENCE_LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    )
    reference_llm_reasoning: bool = os.getenv("REFERENCE_LLM_REASONING", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "ollama")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    embedding_dimensions: int = _default_embedding_dimensions()
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "")
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    embedding_max_tokens: int = int(os.getenv("EMBEDDING_MAX_TOKENS", "4096"))
    # Asymmetric retrieval prefix for the query side.
    # Giga-Embeddings-instruct (ai-sage) is an instruction-tuned encoder:
    # queries should be wrapped as `Instruct: <task>\nQuery: <text>` while
    # documents are encoded plain. The default below is a legal-RAG-specific
    # task instruction; override via env if you swap the corpus. An empty or
    # unset env var falls back to this default (a literal "" disables the
    # instruction by setting EMBEDDING_QUERY_INSTRUCTION_DISABLED=1 instead).
    embedding_query_instruction: str = (
        os.getenv("EMBEDDING_QUERY_INSTRUCTION", "").strip()
        or "Дано краткое описание сделки на русском языке. Найди статьи и положения "
        "российского гражданского законодательства, регулирующие условия такой сделки."
    )
    # Escape hatch: set EMBEDDING_QUERY_INSTRUCTION_DISABLED=1 to drop the
    # instruction wrapping entirely (compare with raw plain queries).
    embedding_query_instruction_disabled: bool = os.getenv(
        "EMBEDDING_QUERY_INSTRUCTION_DISABLED", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    embedding_trust_remote_code: bool = os.getenv("EMBEDDING_TRUST_REMOTE_CODE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Override the precision the SentenceTransformer is loaded in. "auto" lets
    # the model card decide (typically FP16/BF16 for modern checkpoints, FP32
    # for older ones). Set "float16" / "bfloat16" / "float32" explicitly when
    # tuning VRAM (e.g. fitting Giga + reranker on a 16 GB card).
    embedding_precision: str = os.getenv("EMBEDDING_PRECISION", "auto").strip().lower()
    preload_embeddings_on_startup: bool = os.getenv("PRELOAD_EMBEDDINGS_ON_STARTUP", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    qdrant_collection_general: str = os.getenv("QDRANT_COLLECTION_GENERAL", "npa_general")
    qdrant_collection_primal: str = os.getenv("QDRANT_COLLECTION_PRIMAL", "npa_primal")
    qdrant_collection_secondary: str = os.getenv("QDRANT_COLLECTION_SECONDARY", "npa_secondary")
    qdrant_collection_specific: str = os.getenv(
        "QDRANT_COLLECTION_SPECIFIC", os.getenv("QDRANT_COLLECTION_PRIMAL", "npa_primal")
    )

    retrieval_candidate_top_k: int = int(os.getenv("RETRIEVAL_CANDIDATE_TOP_K", "40"))
    retrieval_context_top_k: int = int(os.getenv("RETRIEVAL_CONTEXT_TOP_K", "7"))
    retrieval_general_top_k: int = int(os.getenv("RETRIEVAL_GENERAL_TOP_K", "8"))
    retrieval_secondary_enrichment_top_k: int = int(os.getenv("RETRIEVAL_SECONDARY_ENRICHMENT_TOP_K", "5"))
    recommendation_enrichment_max_items: int = int(os.getenv("RECOMMENDATION_ENRICHMENT_MAX_ITEMS", "6"))
    retrieval_soft_source_filter: bool = os.getenv("RETRIEVAL_SOFT_SOURCE_FILTER", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_device: str = os.getenv("RERANKER_DEVICE", os.getenv("EMBEDDING_DEVICE", ""))
    reranker_batch_size: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))

    langgraph_api_url: str = os.getenv("LANGGRAPH_API_URL", "http://langgraph_dev:2024")
    langgraph_assistant_id: str = os.getenv("LANGGRAPH_ASSISTANT_ID", "contract_agent")
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "3"))
    max_classification_clarifications: int = int(os.getenv("MAX_CLASSIFICATION_CLARIFICATIONS", "5"))
    # Hard timeout for a single LangGraph run (seconds). After this we cancel
    # the run and mark the assistant message as ERROR.
    run_timeout_seconds: int = int(os.getenv("RUN_TIMEOUT_SECONDS", "300"))

    s3_enabled: bool = os.getenv("S3_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
    s3_public_endpoint_url: str = os.getenv("S3_PUBLIC_ENDPOINT_URL", "http://localhost:9000")
    s3_bucket: str = os.getenv("S3_BUCKET", "contracts")
    s3_access_key_id: str = os.getenv("S3_ACCESS_KEY_ID", "minioadmin")
    s3_secret_access_key: str = os.getenv("S3_SECRET_ACCESS_KEY", "minioadmin")
    s3_region: str = os.getenv("S3_REGION", "us-east-1")
    s3_presign_expires_seconds: int = int(os.getenv("S3_PRESIGN_EXPIRES_SECONDS", "3600"))

    env: str = os.getenv("ENV", "dev").strip().lower()
    log_level: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    session_secret: str = os.getenv("SESSION_SECRET", "")
    session_https_only: bool = os.getenv("SESSION_HTTPS_ONLY", "false").strip().lower() in {"1", "true", "yes", "on"}
    session_same_site: str = os.getenv("SESSION_SAME_SITE", "lax").strip().lower()
    frontend_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(_split_csv(os.getenv("FRONTEND_ORIGINS", "http://localhost:3000")))
    )
    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    google_redirect_uri: str = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:3000/api/auth/google/callback")
    # Yandex OAuth for end-users (optional; disabled when creds are empty).
    yandex_client_id: str = os.getenv("YANDEX_CLIENT_ID", "").strip()
    yandex_client_secret: str = os.getenv("YANDEX_CLIENT_SECRET", "").strip()
    yandex_redirect_uri: str = os.getenv("YANDEX_REDIRECT_URI", "http://localhost:3000/api/auth/yandex/callback")
    seed_admin_email: str = os.getenv("SEED_ADMIN_EMAIL", "admin@local")
    # Owner credentials — the top-level operator with exclusive rights to manage
    # admins. Auth is verified directly against these env vars (never stored in
    # admin_users). Leave both empty to run without an owner account.
    # OWNER_USERNAME / OWNER_PASSWORD take precedence; ADMIN_USERNAME /
    # ADMIN_PASSWORD are accepted as a backward-compatible fallback.
    owner_username: str = os.getenv("OWNER_USERNAME", "").strip() or os.getenv("ADMIN_USERNAME", "").strip()
    owner_password: str = os.getenv("OWNER_PASSWORD", "") or os.getenv("ADMIN_PASSWORD", "")
    admin_npa_bucket: str = os.getenv("ADMIN_NPA_BUCKET", "admin-npa")
    admin_npa_max_upload_mb: int = int(os.getenv("ADMIN_NPA_MAX_UPLOAD_MB", "50"))

    # Billing (YooKassa). Empty creds disable billing entirely: /api/billing/*
    # routes return 503 and the frontend hides upgrade CTAs. Test/prod is
    # decided by the shop_id you registered, not by a flag.
    yookassa_shop_id: str = os.getenv("YOOKASSA_SHOP_ID", "").strip()
    yookassa_secret_key: str = os.getenv("YOOKASSA_SECRET_KEY", "").strip()
    yookassa_test_mode: bool = os.getenv("YOOKASSA_TEST_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
    yookassa_return_url: str = os.getenv(
        "YOOKASSA_RETURN_URL",
        "http://localhost:3000/billing/return",
    )

    @property
    def billing_enabled(self) -> bool:
        return bool(self.yookassa_shop_id and self.yookassa_secret_key)

    @property
    def owner_enabled(self) -> bool:
        return bool(self.owner_username and self.owner_password)

    @property
    def yandex_configured(self) -> bool:
        return bool(self.yandex_client_id and self.yandex_client_secret)


def _validate(s: Settings) -> Settings:
    is_prod = s.env == "production"
    insecure_defaults = {
        "POSTGRES_PASSWORD": (s.postgres_password, {"password", ""}),
        "S3_SECRET_ACCESS_KEY": (s.s3_secret_access_key, {"minioadmin", ""}),
        "SESSION_SECRET": (s.session_secret, {""}),
        "GOOGLE_CLIENT_ID": (s.google_client_id, {""}),
        "GOOGLE_CLIENT_SECRET": (s.google_client_secret, {""}),
    }
    for name, (value, bad) in insecure_defaults.items():
        if value in bad:
            msg = f"{name} is empty or uses insecure default"
            if is_prod:
                raise RuntimeError(msg)
            LOGGER.warning("config: %s (allowed only in dev)", msg)
    if not s.session_secret:
        # Per-process random secret for dev; sessions invalidate on restart.
        object.__setattr__(s, "session_secret", secrets.token_urlsafe(48))
        LOGGER.warning("config: SESSION_SECRET not set; generated ephemeral secret")
    return s


settings = _validate(Settings())
