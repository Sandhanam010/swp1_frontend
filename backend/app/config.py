"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """All configuration is read from .env (or real env vars)."""

    # ── Gemini (Embeddings) ──
    gemini_api_key: str = Field(..., description="Google Gemini API key for embeddings")

    # ── Local LLM ──
    model_path: str = Field(..., description="Absolute path to the GGUF model file")

    # ── ChromaDB ──
    chroma_persist_dir: str = Field(default="./chroma_db")

    # ── Data ──
    data_dir: str = Field(default="./data")

    # ── LLM Parameters ──
    context_window: int = Field(default=4096)
    max_new_tokens: int = Field(default=512)

    # ── Retrieval ──
    top_k: int = Field(default=3)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton – import this everywhere
settings = Settings()
