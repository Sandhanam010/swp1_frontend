"""API routes — POST /api/ask  +  POST /api/ingest."""

import logging

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.core.rag_engine import rag_engine
from app.services.ingest import ingest_uploaded_file, ingest_all_from_data_dir

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["rag"])


# ────────────────────────────────────────────────
#  Schemas
# ────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Student query")


class SourceNode(BaseModel):
    text: str
    score: float | None = None
    metadata: dict = {}


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceNode] = []


class IngestResponse(BaseModel):
    status: str
    filename: str | None = None
    directory: str | None = None
    chunks: int = 0
    message: str | None = None


# ────────────────────────────────────────────────
#  POST /api/ask
# ────────────────────────────────────────────────
@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest):
    """Run RAG query: retrieve context from ChromaDB → synthesize via Mistral-7B."""
    try:
        result = rag_engine.query(body.query)
        return AskResponse(
            answer=result["answer"],
            sources=[SourceNode(**s) for s in result["sources"]],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")


# ────────────────────────────────────────────────
#  POST /api/ingest
# ────────────────────────────────────────────────
@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF/TXT, chunk it, embed it, and store in ChromaDB."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed = {".pdf", ".txt", ".md", ".docx"}
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    try:
        content = await file.read()
        result = ingest_uploaded_file(file.filename, content)
        return IngestResponse(**result)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")


# ────────────────────────────────────────────────
#  POST /api/ingest/bulk
# ────────────────────────────────────────────────
@router.post("/ingest/bulk", response_model=IngestResponse)
async def ingest_bulk():
    """Ingest all files already present in the data/ directory."""
    try:
        result = ingest_all_from_data_dir()
        return IngestResponse(**result)
    except Exception as exc:
        logger.exception("Bulk ingestion failed")
        raise HTTPException(status_code=500, detail=f"Bulk ingestion error: {exc}")
