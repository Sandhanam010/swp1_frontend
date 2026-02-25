"""Mistral-7B local LLM via llama-cpp-python, wrapped for LlamaIndex."""

from llama_index.llms.llama_cpp import LlamaCPP
from app.config import settings


def get_llm() -> LlamaCPP:
    """Return a configured LlamaCPP instance pointing at the local GGUF model."""
    return LlamaCPP(
        model_path=settings.model_path,
        temperature=0.1,
        max_new_tokens=settings.max_new_tokens,
        context_window=settings.context_window,
        # Generate kwargs passed to llama-cpp
        generate_kwargs={"top_p": 0.9, "top_k": 40},
        # Model kwargs passed to llama_cpp.Llama()
        model_kwargs={"n_gpu_layers": 0},  # CPU-only; set >0 if GPU available
        verbose=False,
    )
