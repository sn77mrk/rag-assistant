"""
FastAPI route handlers for the RAG assistant.

Endpoints:
  GET  /health           — service health and status
  GET  /store/info       — vector store statistics
  POST /ingest           — ingest a document into the knowledge base
  POST /ask              — ask a question (QA, research, or deep_answer workflow)
  POST /compare          — compare two documents
  POST /extract          — extract structured fields from documents
  POST /evaluate         — evaluate a query-answer pair
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
import shutil
import tempfile

from config import config
from ingestion import DocumentLoader, TextCleaner, Chunker
from retrieval import VectorStore, Retriever
from generation import LLMClient
from agent import RAGWorkflow
from evaluation import RAGEvaluator

from .schemas import (
    AskRequest,
    AskResponse,
    CompareRequest,
    EvaluateRequest,
    EvaluateResponse,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    StoreInfoResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Shared singletons (initialized once at startup) ──────────────────────────
_store: VectorStore = None
_retriever: Retriever = None
_llm: LLMClient = None
_workflow: RAGWorkflow = None
_evaluator: RAGEvaluator = None


def get_components():
    """Lazy initialization of all components (avoids loading models at import time)."""
    global _store, _retriever, _llm, _workflow, _evaluator

    if _store is None:
        _store = VectorStore()
        _retriever = Retriever(_store)
        _llm = LLMClient()
        _workflow = RAGWorkflow(_retriever, _llm)
        _evaluator = RAGEvaluator(llm_client=None)  # heuristic by default

    return _store, _retriever, _llm, _workflow, _evaluator


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Check service health. Returns model and store info."""
    store, *_ = get_components()
    return HealthResponse(
        status="ok",
        model=config.GROQ_MODEL,
        embedding_backend=config.EMBEDDING_BACKEND,
        documents_in_store=store.count(),
    )


@router.get("/store/info", response_model=StoreInfoResponse, tags=["system"])
def store_info():
    """Return vector store statistics."""
    store, *_ = get_components()
    return StoreInfoResponse(
        total_documents=store.count(),
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_backend=config.EMBEDDING_BACKEND,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


# ── Ingestion ─────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, tags=["documents"])
def ingest(request: IngestRequest):
    """
    Ingest a document or directory into the knowledge base.

    Pipeline:
      load → clean → chunk → embed → store in ChromaDB
    """
    store, *_ = get_components()

    path = Path(request.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {request.path}")

    loader = DocumentLoader()
    cleaner = TextCleaner()
    chunker = Chunker()

    raw_docs = loader.load(str(path))
    if not raw_docs:
        raise HTTPException(status_code=422, detail="No text could be extracted from the provided path.")

    cleaned_docs = cleaner.clean_documents(raw_docs)
    chunks = chunker.split_documents(cleaned_docs)

    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no chunks after processing.")

    store.add_documents(chunks)

    return IngestResponse(
        status="ok",
        documents_loaded=len(raw_docs),
        chunks_indexed=len(chunks),
        total_in_store=store.count(),
    )


@router.post("/ingest/upload", response_model=IngestResponse, tags=["documents"])
async def ingest_upload(file: UploadFile = File(...)):
    """
    Upload and ingest a file directly via multipart form.

    Saves the file to a temp directory, then runs the same ingestion pipeline.
    """
    store, *_ = get_components()

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    loader = DocumentLoader()
    cleaner = TextCleaner()
    chunker = Chunker()

    raw_docs = loader.load(tmp_path)
    cleaned_docs = cleaner.clean_documents(raw_docs)
    chunks = chunker.split_documents(cleaned_docs)

    if not chunks:
        raise HTTPException(status_code=422, detail="Uploaded file produced no usable content.")

    store.add_documents(chunks)

    return IngestResponse(
        status="ok",
        documents_loaded=len(raw_docs),
        chunks_indexed=len(chunks),
        total_in_store=store.count(),
    )


# ── QA and Workflows ──────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(request: AskRequest):
    """
    Ask a question. Supports three workflow modes:
      - ask:         standard retrieve → answer
      - research:    retrieve → summarize (broader context)
      - deep_answer: retrieve with scores → filter → answer
    """
    _, _, _, workflow, _ = get_components()

    if workflow is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    wf = request.workflow.lower()
    if wf == "research":
        result = workflow.research(request.question, source=request.source)
    elif wf == "deep_answer":
        result = workflow.deep_answer(request.question)
    else:
        result = workflow.ask(request.question, source=request.source)

    if result.error:
        logger.error("Workflow error: %s", result.error)

    return AskResponse(
        workflow=result.workflow,
        question=request.question,
        answer=result.answer,
        sources=result.sources,
        steps=result.steps,
        error=result.error,
    )


@router.post("/compare", response_model=AskResponse, tags=["qa"])
def compare(request: CompareRequest):
    """Compare a topic across two documents."""
    _, _, _, workflow, _ = get_components()
    result = workflow.compare(request.question, request.source_a, request.source_b)

    return AskResponse(
        workflow=result.workflow,
        question=request.question,
        answer=result.answer,
        sources=result.sources,
        steps=result.steps,
        error=result.error,
    )


@router.post("/extract", response_model=ExtractResponse, tags=["qa"])
def extract(request: ExtractRequest):
    """Extract structured fields from documents as JSON."""
    _, _, _, workflow, _ = get_components()
    result = workflow.extract(
        fields=request.fields,
        query=request.query,
        source=request.source,
    )

    return ExtractResponse(
        workflow=result.workflow,
        fields_requested=request.fields,
        extracted=result.answer if isinstance(result.answer, dict) else {"raw": result.answer},
        sources=result.sources,
        error=result.error,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

@router.post("/evaluate", response_model=EvaluateResponse, tags=["evaluation"])
def evaluate(request: EvaluateRequest):
    """
    Evaluate a query-answer pair for retrieval relevance and answer grounding.

    By default uses heuristic evaluation (no API cost).
    Set use_llm=true to use LLM-assisted scoring.
    """
    store, retriever, llm, _, evaluator = get_components()

    # Retrieve context for the query
    docs = retriever.retrieve(request.query)

    if request.use_llm:
        evaluator_to_use = RAGEvaluator(llm_client=llm)
    else:
        evaluator_to_use = evaluator

    result = evaluator_to_use.evaluate(
        query=request.query,
        answer=request.answer,
        context_docs=docs,
    )

    return EvaluateResponse(
        query=result.query,
        retrieval_relevance_score=result.retrieval_relevance_score,
        answer_grounding_score=result.answer_grounding_score,
        hallucination_risk=result.hallucination_risk,
        context_coverage=result.context_coverage,
        notes=result.notes,
    )
