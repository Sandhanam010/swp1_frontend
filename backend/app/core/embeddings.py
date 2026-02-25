"""Gemini embedding model wrapper for LlamaIndex."""

from llama_index.embeddings.gemini import GeminiEmbedding
from app.config import settings


def get_embed_model() -> GeminiEmbedding:
    """Return a configured Gemini embedding model instance."""
    return GeminiEmbedding(
        api_key=settings.gemini_api_key,
        model_name="models/embedding-001",
    )
