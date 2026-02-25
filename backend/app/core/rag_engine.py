"""RAG Engine — LlamaIndex orchestration over ChromaDB.

Provides:
  - query(user_query) → answer string + source metadata
  - ingest(file_path)  → load, chunk, and store a document
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from llama_index.core import (
    Settings as LlamaSettings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore

from app.config import settings
from app.core.embeddings import get_embed_model
from app.core.llm import get_llm
from app.core.prompts import QA_PROMPT

logger = logging.getLogger(__name__)


class RAGEngine:
    """Singleton-style RAG engine wrapping index + query engine."""

    def __init__(self) -> None:
        self._index: Optional[VectorStoreIndex] = None
        self._chroma_collection = None

    # ──────────────────────────────────────────────
    # Startup
    # ──────────────────────────────────────────────
    def initialize(self) -> None:
        """Performs heavy one-time setup: LLM, embeddings, ChromaDB, index."""
        logger.info("Initializing RAG engine …")

        # 1. Configure global LlamaIndex settings
        embed_model = get_embed_model()
        llm = get_llm()
        LlamaSettings.embed_model = embed_model
        LlamaSettings.llm = llm
        LlamaSettings.chunk_size = 512
        LlamaSettings.chunk_overlap = 50

        # 2. Persistent ChromaDB client + collection
        persist_dir = Path(settings.chroma_persist_dir).resolve()
        persist_dir.mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        self._chroma_collection = chroma_client.get_or_create_collection(
            name="academic_docs"
        )

        # 3. Build (or load) the VectorStoreIndex
        vector_store = ChromaVectorStore(chroma_collection=self._chroma_collection)
        storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_ctx,
        )

        doc_count = self._chroma_collection.count()
        logger.info("RAG engine ready  •  %d chunks in ChromaDB", doc_count)

    # ──────────────────────────────────────────────
    # Query
    # ──────────────────────────────────────────────
    def query(self, user_query: str) -> dict:
        """Run retrieval + synthesis and return answer + sources."""
        if self._index is None:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")

        query_engine = self._index.as_query_engine(
            similarity_top_k=settings.top_k,
            text_qa_template=QA_PROMPT,
        )

        response = query_engine.query(user_query)

        # Extract source metadata
        sources = []
        for node in response.source_nodes:
            sources.append(
                {
                    "text": node.node.get_content()[:200],  # preview
                    "score": round(node.score, 4) if node.score else None,
                    "metadata": node.node.metadata,
                }
            )

        return {"answer": str(response), "sources": sources}

    # ──────────────────────────────────────────────
    # Ingest
    # ──────────────────────────────────────────────
    def ingest(self, file_path: str) -> int:
        """Read a single file, chunk it, and insert into ChromaDB.

        Returns the number of chunks created.
        """
        if self._index is None:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info("Ingesting %s …", path.name)

        # Load documents
        reader = SimpleDirectoryReader(input_files=[str(path)])
        documents = reader.load_data()

        # Chunk
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)

        # Insert into the existing index
        self._index.insert_nodes(nodes)

        logger.info("Ingested %d chunks from %s", len(nodes), path.name)
        return len(nodes)

    def ingest_directory(self, dir_path: str) -> int:
        """Ingest all supported files from a directory.

        Returns total number of chunks created.
        """
        if self._index is None:
            raise RuntimeError("RAG engine not initialized. Call initialize() first.")

        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        reader = SimpleDirectoryReader(input_dir=str(path), recursive=True)
        documents = reader.load_data()

        if not documents:
            logger.warning("No documents found in %s", dir_path)
            return 0

        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        self._index.insert_nodes(nodes)

        logger.info("Ingested %d chunks from directory %s", len(nodes), dir_path)
        return len(nodes)


# Module-level singleton
rag_engine = RAGEngine()
