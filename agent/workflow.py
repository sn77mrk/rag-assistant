"""
RAG Workflow: orchestrates multi-step agent-like pipelines.

Why multi-step workflows:
  Simple RAG (retrieve → generate) handles basic QA well.
  But real use cases often require:
    - retrieving from multiple sources and combining results
    - deciding between summarization and QA
    - comparing two documents
    - extracting structured data before generating text
    - checking whether retrieved context is relevant before generating

  This class implements these patterns explicitly as named workflows,
  each composed from the tool primitives in tools.py.

  Design philosophy: explicit > implicit. Each workflow step is visible
  and traceable — unlike black-box agent chains that are hard to debug.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from generation.llm_client import LLMClient
from retrieval.retriever import Retriever
from .tools import (
    AnswerTool,
    CompareTool,
    ExtractTool,
    SearchTool,
    SummarizeTool,
    ToolResult,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Structured output of any workflow execution."""
    workflow: str
    answer: Any
    sources: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class RAGWorkflow:
    """
    Orchestrates agent-like multi-step RAG workflows.

    Available workflows:
        ask(question)               → retrieve + answer (standard QA)
        research(topic)             → retrieve + summarize (research mode)
        compare(question, src_a, src_b) → retrieve from both + compare
        extract(fields, source)     → retrieve + structured extraction
        deep_answer(question)       → retrieve + check relevance + answer
    """

    def __init__(self, retriever: Retriever, llm_client: LLMClient):
        self.retriever = retriever
        self.llm = llm_client

        # Initialize tools
        self.search = SearchTool(retriever)
        self.answer_tool = AnswerTool(llm_client)
        self.summarize_tool = SummarizeTool(llm_client)
        self.compare_tool = CompareTool(llm_client)
        self.extract_tool = ExtractTool(llm_client)

    # ── Standard QA ──────────────────────────────────────────────────────────

    def ask(self, question: str, source: str = None) -> WorkflowResult:
        """
        Standard RAG workflow: retrieve → answer.

        Step 1: Search the vector store for relevant chunks.
        Step 2: Generate an LLM answer grounded in those chunks.

        Args:
            question: User's question.
            source:   Optional: restrict retrieval to a specific document.

        Returns:
            WorkflowResult with the generated answer and sources.
        """
        steps = []

        # Step 1: Retrieve
        search_result = self.search.run(question, source=source)
        steps.append({"step": "search", "n_chunks": len(search_result.output), "query": question})

        if not search_result.success:
            return WorkflowResult(workflow="ask", answer="", error=search_result.error, steps=steps)

        if not search_result.output:
            return WorkflowResult(
                workflow="ask",
                answer="No relevant documents found for this question.",
                steps=steps,
            )

        # Step 2: Generate answer
        answer_result = self.answer_tool.run(question, search_result.output)
        steps.append({"step": "answer", "success": answer_result.success})

        return WorkflowResult(
            workflow="ask",
            answer=answer_result.output,
            sources=search_result.sources,
            steps=steps,
            error=answer_result.error,
        )

    # ── Research / Summarization ──────────────────────────────────────────────

    def research(self, topic: str, focus: str = "general", source: str = None) -> WorkflowResult:
        """
        Research workflow: retrieve → summarize.

        Useful when the user wants a digest rather than a specific answer.

        Step 1: Retrieve more chunks than for QA (broader context).
        Step 2: Summarize with an optional focus area.
        """
        steps = []

        # Retrieve with larger k for broader coverage
        search_result = self.search.run(topic, k=8, source=source)
        steps.append({"step": "search", "n_chunks": len(search_result.output), "topic": topic})

        if not search_result.success or not search_result.output:
            return WorkflowResult(
                workflow="research",
                answer="No relevant content found to summarize.",
                steps=steps,
                error=search_result.error,
            )

        # Summarize
        summary_result = self.summarize_tool.run(search_result.output, focus=focus)
        steps.append({"step": "summarize", "focus": focus})

        return WorkflowResult(
            workflow="research",
            answer=summary_result.output,
            sources=search_result.sources,
            steps=steps,
            error=summary_result.error,
        )

    # ── Cross-document comparison ─────────────────────────────────────────────

    def compare(
        self,
        question: str,
        source_a: str,
        source_b: str,
    ) -> WorkflowResult:
        """
        Comparison workflow: retrieve from two sources → compare.

        Step 1: Search document A for the question topic.
        Step 2: Search document B for the question topic.
        Step 3: Generate a structured comparison.

        Args:
            question: What to compare (e.g. "refund policy").
            source_a: Path or name of document A.
            source_b: Path or name of document B.
        """
        steps = []

        # Retrieve from document A
        result_a = self.search.run(question, source=source_a)
        steps.append({"step": "search_a", "source": source_a, "n_chunks": len(result_a.output)})

        # Retrieve from document B
        result_b = self.search.run(question, source=source_b)
        steps.append({"step": "search_b", "source": source_b, "n_chunks": len(result_b.output)})

        if not result_a.output and not result_b.output:
            return WorkflowResult(
                workflow="compare",
                answer="Could not find relevant content in either document.",
                steps=steps,
            )

        # Compare
        compare_result = self.compare_tool.run(question, result_a.output, result_b.output)
        steps.append({"step": "compare"})

        return WorkflowResult(
            workflow="compare",
            answer=compare_result.output,
            sources=[source_a, source_b],
            steps=steps,
            error=compare_result.error,
        )

    # ── Structured extraction ─────────────────────────────────────────────────

    def extract(
        self,
        fields: List[str],
        query: str = None,
        source: str = None,
    ) -> WorkflowResult:
        """
        Extraction workflow: retrieve → extract structured JSON.

        Step 1: Retrieve relevant chunks (or all chunks from source).
        Step 2: Extract requested fields as a JSON object.

        Args:
            fields: Field names to extract, e.g. ["author", "date", "price"].
            query:  Optional search query to focus retrieval.
            source: Optional source filter.
        """
        steps = []

        search_query = query or " ".join(fields)
        search_result = self.search.run(search_query, source=source)
        steps.append({"step": "search", "query": search_query})

        if not search_result.output:
            return WorkflowResult(
                workflow="extract",
                answer={f: None for f in fields},
                steps=steps,
            )

        extract_result = self.extract_tool.run(search_result.output, fields)
        steps.append({"step": "extract", "fields": fields})

        return WorkflowResult(
            workflow="extract",
            answer=extract_result.output,
            sources=search_result.sources,
            steps=steps,
            error=extract_result.error,
        )

    # ── Deep answer (with relevance check) ────────────────────────────────────

    def deep_answer(self, question: str) -> WorkflowResult:
        """
        Deep answer workflow: retrieve → check relevance → answer.

        Adds a relevance check step before generating an answer.
        If retrieved chunks are not relevant enough, returns a
        "no sufficient information" response rather than hallucinating.

        Step 1: Retrieve with scores.
        Step 2: Filter out low-relevance chunks (score threshold).
        Step 3: Generate answer only from relevant chunks.
        """
        steps = []

        # Retrieve with scores
        scored_docs = self.retriever.retrieve_with_scores(question)
        steps.append({"step": "retrieve_with_scores", "n_raw": len(scored_docs)})

        # Filter by relevance score
        # ChromaDB uses L2 distance: lower = more similar
        # We keep chunks where score < threshold (adjustable)
        DISTANCE_THRESHOLD = 1.5
        relevant_docs = [
            doc for doc, score in scored_docs if score < DISTANCE_THRESHOLD
        ]
        steps.append({
            "step": "relevance_filter",
            "threshold": DISTANCE_THRESHOLD,
            "kept": len(relevant_docs),
            "dropped": len(scored_docs) - len(relevant_docs),
        })

        if not relevant_docs:
            return WorkflowResult(
                workflow="deep_answer",
                answer=(
                    "The knowledge base does not contain sufficiently relevant "
                    "information to answer this question reliably."
                ),
                steps=steps,
            )

        # Answer from filtered chunks only
        answer_result = self.answer_tool.run(question, relevant_docs)
        steps.append({"step": "answer"})

        sources = list({d.metadata.get("source", "") for d in relevant_docs})
        return WorkflowResult(
            workflow="deep_answer",
            answer=answer_result.output,
            sources=sources,
            steps=steps,
            error=answer_result.error,
        )
