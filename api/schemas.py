"""
Pydantic schemas for API request/response validation.

Why strict schemas:
  FastAPI uses Pydantic to validate every request before it reaches the handler.
  This prevents malformed input from reaching the ML pipeline (empty queries,
  oversized requests, invalid field types) and produces clear error messages.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request schemas ───────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Request to ingest a document from a file path."""
    path: str = Field(..., description="Path to a file or directory to ingest.")


class AskRequest(BaseModel):
    """Standard QA request."""
    question: str = Field(..., min_length=1, max_length=2000)
    source: Optional[str] = Field(None, description="Restrict to a specific document source.")
    workflow: str = Field("ask", description="Workflow type: ask | research | deep_answer")

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class CompareRequest(BaseModel):
    """Cross-document comparison request."""
    question: str = Field(..., min_length=1, max_length=2000)
    source_a: str = Field(..., description="Path or name of document A.")
    source_b: str = Field(..., description="Path or name of document B.")


class ExtractRequest(BaseModel):
    """Structured extraction request."""
    fields: List[str] = Field(..., min_length=1, description="Fields to extract.")
    query: Optional[str] = None
    source: Optional[str] = None


class EvaluateRequest(BaseModel):
    """Evaluation request for a query-answer pair."""
    query: str
    answer: str
    use_llm: bool = Field(False, description="Use LLM-assisted evaluation (costs tokens).")


# ── Response schemas ──────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str
    documents_loaded: int
    chunks_indexed: int
    total_in_store: int


class AskResponse(BaseModel):
    workflow: str
    question: str
    answer: str
    sources: List[str]
    steps: List[Dict[str, Any]]
    error: Optional[str] = None


class ExtractResponse(BaseModel):
    workflow: str
    fields_requested: List[str]
    extracted: Dict[str, Any]
    sources: List[str]
    error: Optional[str] = None


class EvaluateResponse(BaseModel):
    query: str
    retrieval_relevance_score: float
    answer_grounding_score: float
    hallucination_risk: str
    context_coverage: str
    notes: str


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_backend: str
    documents_in_store: int


class StoreInfoResponse(BaseModel):
    total_documents: int
    collection_name: str
    embedding_backend: str
    chunk_size: int
    chunk_overlap: int
