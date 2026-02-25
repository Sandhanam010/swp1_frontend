"""FastAPI application — Academic AI Tutor RAG Backend.

Endpoints:
  POST /api/ask          → Query the RAG pipeline
  POST /api/ingest       → Upload & ingest a single document
  POST /api/ingest/bulk  → Ingest all files from data/ directory
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.rag_engine import rag_engine
from app.api.routes import router

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG engine once on startup."""
    logger.info("Starting RAG engine initialization …")
    rag_engine.initialize()
    logger.info("RAG engine is ready.")
    yield
    logger.info("Shutting down.")


# ── App ──
app = FastAPI(
    title="SWP1 Academic AI Tutor API",
    description="Headless RAG backend — LlamaIndex + ChromaDB + Mistral-7B",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (permissive for dev) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──
app.include_router(router)


# ── Health check ──
@app.get("/health")
async def health():
    return {"status": "ok"}
