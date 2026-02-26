"""Gemini embedding model wrapper for LlamaIndex.

Uses the new `google-genai` SDK (not the deprecated google-generativeai)
with `gemini-embedding-001` — the currently available embedding model.
"""

from typing import List

from google import genai
from llama_index.core.base.embeddings.base import BaseEmbedding

from app.config import settings


class GeminiEmbedding(BaseEmbedding):
    """Custom LlamaIndex embedding using the google-genai SDK."""

    model_name: str = "gemini-embedding-001"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def _get_query_embedding(self, query: str) -> List[float]:
        """Embed a single query string."""
        result = self._client.models.embed_content(
            model=self.model_name,
            contents=query,
        )
        return result.embeddings[0].values

    def _get_text_embedding(self, text: str) -> List[float]:
        """Embed a single document text."""
        result = self._client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        return result.embeddings[0].values

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        embeddings = []
        for text in texts:
            result = self._client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


def get_embed_model() -> GeminiEmbedding:
    """Return a configured Gemini embedding model instance."""
    return GeminiEmbedding()
