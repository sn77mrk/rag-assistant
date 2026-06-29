"""
RAG-Based AI Assistant — FastAPI entry point.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Interactive docs available at:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import config
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG-Based AI Assistant",
    description=(
        "Document question-answering system powered by semantic retrieval and LLM generation. "
        "Supports multi-step agent-like workflows: QA, research, cross-document comparison, "
        "and structured extraction. Built with LangChain, ChromaDB, and OpenAI."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    logger.info("Starting RAG Assistant")
    logger.info("  LLM model:         %s", config.GROQ_MODEL)
    logger.info("  Embedding backend: %s", config.EMBEDDING_BACKEND)
    logger.info("  Vector store:      %s", config.CHROMA_PERSIST_DIR)
    logger.info("  Chunk size:        %d tokens", config.CHUNK_SIZE)
    logger.info("  Retrieval top-k:   %d", config.RETRIEVAL_TOP_K)


@app.get("/", tags=["root"])
def root():
    return {
        "service": "RAG-Based AI Assistant",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health":   "/api/v1/health",
            "ingest":   "POST /api/v1/ingest",
            "upload":   "POST /api/v1/ingest/upload",
            "ask":      "POST /api/v1/ask",
            "compare":  "POST /api/v1/compare",
            "extract":  "POST /api/v1/extract",
            "evaluate": "POST /api/v1/evaluate",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT, reload=True)
