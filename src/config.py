from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
import os
import gc
import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class Settings(BaseSettings):
    load_dotenv()

    groq_api_key: str = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL")
    ragas_judge_model: str = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
    groq_temperature: float = 0.0
    groq_seed: int = 42

    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "rag_system")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_local_files_only: bool = True

    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_reranked: int = 5
    max_retries: int = 2

    class Config:
        env_file = ".env"


settings = Settings()


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

    cache_folder = str(Path(__file__).parents[2] / ".cache" / "huggingface")

    def _build() -> HuggingFaceEmbedding:
        return HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            cache_folder=cache_folder,
            embed_batch_size=128,
            show_progress_bar=True,
            device="cpu",
            local_files_only=settings.embedding_local_files_only,
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
