from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path
import os
import gc


PROJECT_ROOT = Path(__file__).parents[1]

# Laptop defaults are the field defaults below. A non-laptop profile only
# overrides values the operator did not set explicitly, so environment
# variables always win over the profile.
_PROFILE_OVERRIDES: dict[str, dict[str, object]] = {
    "laptop": {},
    "server": {
        "embedding_batch_size": 128,
        "enable_reranker": True,
        "retrieval_ranking": "rrf",
        "top_k_dense": 20,
        "top_k_sparse": 20,
        "top_k_reranked": 5,
        "max_retries": 2,
        "max_context_chars": 20000,
        "max_sub_questions": 4,
        "max_concurrent_queries": 4,
        "max_output_tokens": 1024,
        "request_timeout_seconds": 60.0,
    },
}


class Settings(BaseSettings):
    """Runtime configuration. Environment variables use upper-case field names."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    profile: str = "laptop"
    llm_provider: str = "groq"

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    ragas_judge_model: str = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
    groq_temperature: float = 0.0
    groq_seed: int = 42
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_model: str = "qwen2.5:1.5b-instruct"

    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_system")
    index_version: str = "v1"

    # Small, reliable default for an 8 GB CPU-only development machine.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_local_files_only: bool = True
    embedding_batch_size: int = 64
    enable_reranker: bool = False

    # How chunks are ordered: "bm25" or "rrf" (dense and BM25 fused by rank). On 300
    # labeled questions with all-MiniLM-L6-v2, hit@3 was 0.683 for BM25 and 0.617 for
    # RRF. Dense search runs either way, because grading needs its cosine score.
    retrieval_ranking: str = "bm25"

    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_reranked: int = 3
    max_retries: int = 1
    max_question_chars: int = 1000
    max_context_chars: int = 7000
    max_sub_questions: int = 2
    max_concurrent_queries: int = 1
    max_output_tokens: int = 768
    request_timeout_seconds: float = 45.0

    # Provider resilience. LLM_PROVIDER's model is always tried first, then
    # LLM_FALLBACK_CHAIN in order, written as "provider:model,provider:model".
    # The attempt timeout must leave room inside the request timeout for the
    # next provider, otherwise one hung provider uses up the whole request.
    llm_fallback_chain: str = "groq:openai/gpt-oss-20b"
    llm_attempt_timeout_seconds: float = 12.0
    llm_connect_timeout_seconds: float = 5.0
    llm_min_call_seconds: float = 2.0
    llm_max_attempts_per_provider: int = 2
    llm_max_calls_per_query: int = 6
    breaker_failure_threshold: int = 3
    breaker_reset_seconds: float = 30.0

    # A shared daily budget, kept under the provider's limit. 0 disables a limit.
    llm_daily_token_budget: int = 180_000
    llm_daily_request_budget: int = 900
    llm_quota_path: str = ".cache/quota"

    # A retrieved query counts as covered when its best cosine similarity reaches
    # this. Calibrated for all-MiniLM-L6-v2 with scripts/calibrate_evidence.py: 0.40
    # covers 99% of 300 corpus questions and 5% of 40 off-topic ones. It detects
    # off-topic questions; it does not show that the right passage was retrieved.
    evidence_min_dense_score: float = 0.40
    # Cited sentences must share this share of their content words with the
    # chunks they cite, or they count as unsupported.
    citation_min_support: float = 0.2

    warmup_on_startup: bool = True
    metrics_enabled: bool = True

    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_sample_rate: float = 1.0
    # Off by default: LLM entries then hold lengths instead of prompts and passages.
    langfuse_capture_content: bool = False
    # On by default: the query entry shows the question and answer next to their
    # lengths, even with capture_content off. Turn off for shared demos.
    langfuse_capture_query_text: bool = True

    api_key: str = ""
    require_auth: bool = False
    auth_tenant_id: str = "tenant-demo"
    auth_roles: str = "public"

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"groq", "openai", "ollama"}:
            raise ValueError("LLM_PROVIDER must be groq, openai, or ollama")
        return value

    @field_validator("retrieval_ranking")
    @classmethod
    def validate_retrieval_ranking(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"bm25", "rrf"}:
            raise ValueError("RETRIEVAL_RANKING must be bm25 or rrf")
        return value

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in _PROFILE_OVERRIDES:
            raise ValueError(f"PROFILE must be one of {sorted(_PROFILE_OVERRIDES)}")
        return value

    @model_validator(mode="after")
    def apply_profile_overrides(self) -> "Settings":
        """Apply profile defaults to every field the operator did not set."""
        for key, value in _PROFILE_OVERRIDES[self.profile].items():
            if key not in self.model_fields_set:
                setattr(self, key, value)
        return self


settings = Settings()

# HuggingFaceEmbedding drops its local_files_only argument, so the library's own
# offline mode is the only setting that actually stops network lookups. This module
# must stay free of Hugging Face imports for the setting to apply before they load.
if settings.embedding_local_files_only:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


@lru_cache(maxsize=1)
def get_embed_model():
    """
    Return a cached HuggingFaceEmbedding for the project's embedding model.

    The loader is hardened against the intermittent sentence-transformers
    "Cannot copy out of meta tensor" failure (parameters left on the meta
    device by a concurrent/duplicate model instantiation) by forcing an
    explicit, stable CPU-load path and retrying construction. The successful
    instance is cached by the decorator.
    """

    import torch
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    cache_folder = str(PROJECT_ROOT / ".cache" / "huggingface")

    def _build():
        return HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            cache_folder=cache_folder,
            embed_batch_size=settings.embedding_batch_size,
            show_progress_bar=True,
            device="cpu",
            model_kwargs={
                "low_cpu_mem_usage": False,
                "use_safetensors": True,
                "dtype": torch.float32,
            },
        )

    last_error = None
    for _ in range(3):
        try:
            return _build()
        except Exception as exc:
            last_error = exc
            gc.collect()
    raise last_error
