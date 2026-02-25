"""Document ingestion helpers.

Thin wrappers around rag_engine.ingest / ingest_directory used by the API
route and by any future CLI scripts.
"""

import shutil
from pathlib import Path

from app.config import settings
from app.core.rag_engine import rag_engine


def save_upload(filename: str, content: bytes) -> Path:
    """Persist an uploaded file to the data directory and return its path."""
    data_dir = Path(settings.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    dest = data_dir / filename
    dest.write_bytes(content)
    return dest


def ingest_uploaded_file(filename: str, content: bytes) -> dict:
    """Save an uploaded file and ingest it into ChromaDB.

    Returns a summary dict with filename, chunk count, and status.
    """
    file_path = save_upload(filename, content)
    chunk_count = rag_engine.ingest(str(file_path))
    return {
        "status": "ok",
        "filename": filename,
        "chunks": chunk_count,
    }


def ingest_all_from_data_dir() -> dict:
    """Bulk-ingest every file already in the data directory."""
    data_dir = Path(settings.data_dir).resolve()
    if not data_dir.is_dir():
        return {"status": "error", "message": f"Data directory not found: {data_dir}"}

    chunk_count = rag_engine.ingest_directory(str(data_dir))
    return {
        "status": "ok",
        "directory": str(data_dir),
        "chunks": chunk_count,
    }
