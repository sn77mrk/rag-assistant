"""
RAG Evaluator: measures retrieval quality, answer relevance, and grounding.

Why evaluation matters in RAG:
  "The model output is only as good as the retrieved context."
  Without evaluation we cannot tell whether:
    - the retriever is finding genuinely relevant chunks
    - the LLM is hallucinating (generating information not in context)
    - chunk size or overlap changes improve or harm retrieval
    - prompt changes improve answer quality

  This evaluator implements lightweight, LLM-assisted evaluation metrics
  suitable for a portfolio project without a labeled golden dataset.

Evaluation dimensions:
  1. Retrieval relevance  — are retrieved chunks relevant to the query?
  2. Answer grounding     — is the answer supported by retrieved chunks?
  3. Hallucination risk   — does the answer contain info NOT in the context?
  4. Context coverage     — does context contain enough info to answer?
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a single RAG query-answer pair."""
    query: str
    answer: str
    n_chunks_retrieved: int
    retrieval_relevance_score: float   # 0.0–1.0
    answer_grounding_score: float      # 0.0–1.0
    hallucination_risk: str            # "low" | "medium" | "high"
    context_coverage: str             # "sufficient" | "partial" | "insufficient"
    chunk_sources: List[str]
    notes: str = ""


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality across four dimensions.

    Can run with or without an LLM client:
        - Without LLM: heuristic-only evaluation (keyword overlap, length checks)
        - With LLM:    LLM-assisted scoring (more accurate, costs tokens)

    Usage:
        evaluator = RAGEvaluator(llm_client=llm)
        result = evaluator.evaluate(
            query="What is the refund policy?",
            answer="You can return items within 30 days.",
            context_docs=retrieved_docs,
        )
        print(evaluator.format_report(result))
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def evaluate(
        self,
        query: str,
        answer: str,
        context_docs: List[Document],
    ) -> EvaluationResult:
        """
        Run full evaluation for a query-answer-context triple.

        Args:
            query:        The original user question.
            answer:       The LLM-generated answer.
            context_docs: Retrieved chunks used for generation.

        Returns:
            EvaluationResult with all scores and classifications.
        """
        if self.llm:
            return self._llm_evaluate(query, answer, context_docs)
        return self._heuristic_evaluate(query, answer, context_docs)

    # ── LLM-assisted evaluation ───────────────────────────────────────────────

    def _llm_evaluate(
        self,
        query: str,
        answer: str,
        context_docs: List[Document],
    ) -> EvaluationResult:
        """Use the LLM to score retrieval relevance and answer grounding."""

        context_text = "\n\n".join(d.page_content for d in context_docs)
        sources = list({d.metadata.get("source", "") for d in context_docs})

        # Prompt: retrieval relevance
        relevance_prompt = (
            f"Query: {query}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            "Rate how relevant the retrieved context is to the query on a scale 0.0–1.0. "
            "Return only the number."
        )
        relevance_score = self._parse_score(
            self.llm.chat("You are a precise evaluator.", relevance_prompt)
        )

        # Prompt: answer grounding
        grounding_prompt = (
            f"Context:\n{context_text}\n\n"
            f"Answer: {answer}\n\n"
            "Rate how well the answer is grounded in the context (0.0=not grounded, 1.0=fully grounded). "
            "Return only the number."
        )
        grounding_score = self._parse_score(
            self.llm.chat("You are a precise evaluator.", grounding_prompt)
        )

        # Derive hallucination risk from grounding score
        hallucination_risk = self._classify_hallucination(grounding_score)
        context_coverage = self._classify_coverage(relevance_score, len(context_docs))

        return EvaluationResult(
            query=query,
            answer=answer,
            n_chunks_retrieved=len(context_docs),
            retrieval_relevance_score=relevance_score,
            answer_grounding_score=grounding_score,
            hallucination_risk=hallucination_risk,
            context_coverage=context_coverage,
            chunk_sources=sources,
            notes="LLM-assisted evaluation",
        )

    # ── Heuristic evaluation (no LLM) ────────────────────────────────────────

    def _heuristic_evaluate(
        self,
        query: str,
        answer: str,
        context_docs: List[Document],
    ) -> EvaluationResult:
        """
        Lightweight evaluation using keyword overlap — no LLM call required.

        This is less accurate than LLM evaluation but useful for quick checks
        during development without burning API tokens.
        """
        context_text = " ".join(d.page_content.lower() for d in context_docs)
        sources = list({d.metadata.get("source", "") for d in context_docs})

        # Retrieval relevance: fraction of query words found in context
        query_words = set(query.lower().split())
        stop_words = {"the", "a", "an", "is", "in", "of", "to", "and", "or", "what", "how", "why"}
        meaningful_words = query_words - stop_words
        if meaningful_words:
            matched = sum(1 for w in meaningful_words if w in context_text)
            relevance_score = round(matched / len(meaningful_words), 2)
        else:
            relevance_score = 0.5

        # Grounding: fraction of answer words found in context
        answer_words = set(answer.lower().split()) - stop_words
        if answer_words:
            grounded = sum(1 for w in answer_words if w in context_text)
            grounding_score = round(grounded / len(answer_words), 2)
        else:
            grounding_score = 0.0

        hallucination_risk = self._classify_hallucination(grounding_score)
        context_coverage = self._classify_coverage(relevance_score, len(context_docs))

        return EvaluationResult(
            query=query,
            answer=answer,
            n_chunks_retrieved=len(context_docs),
            retrieval_relevance_score=relevance_score,
            answer_grounding_score=grounding_score,
            hallucination_risk=hallucination_risk,
            context_coverage=context_coverage,
            chunk_sources=sources,
            notes="Heuristic evaluation (keyword overlap)",
        )

    # ── Batch evaluation ──────────────────────────────────────────────────────

    def batch_evaluate(
        self,
        test_cases: List[Dict],
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple query-answer-context triples.

        Args:
            test_cases: List of dicts with keys: query, answer, context_docs.

        Returns:
            List of EvaluationResult objects.
        """
        results = []
        for case in test_cases:
            result = self.evaluate(
                query=case["query"],
                answer=case["answer"],
                context_docs=case["context_docs"],
            )
            results.append(result)
        return results

    def aggregate_metrics(self, results: List[EvaluationResult]) -> Dict:
        """Compute aggregate statistics across a batch of evaluations."""
        if not results:
            return {}

        n = len(results)
        avg_relevance = sum(r.retrieval_relevance_score for r in results) / n
        avg_grounding = sum(r.answer_grounding_score for r in results) / n
        avg_chunks = sum(r.n_chunks_retrieved for r in results) / n

        hallucination_counts = {"low": 0, "medium": 0, "high": 0}
        for r in results:
            hallucination_counts[r.hallucination_risk] += 1

        return {
            "n_evaluated": n,
            "avg_retrieval_relevance": round(avg_relevance, 3),
            "avg_answer_grounding": round(avg_grounding, 3),
            "avg_chunks_retrieved": round(avg_chunks, 1),
            "hallucination_risk_distribution": hallucination_counts,
        }

    # ── Reporting ─────────────────────────────────────────────────────────────

    def format_report(self, result: EvaluationResult) -> str:
        """Format a single EvaluationResult as a human-readable report."""
        lines = [
            "=" * 60,
            "RAG EVALUATION REPORT",
            "=" * 60,
            f"Query:                    {result.query}",
            f"Chunks retrieved:         {result.n_chunks_retrieved}",
            f"Retrieval relevance:      {result.retrieval_relevance_score:.2f} / 1.0",
            f"Answer grounding:         {result.answer_grounding_score:.2f} / 1.0",
            f"Hallucination risk:       {result.hallucination_risk.upper()}",
            f"Context coverage:         {result.context_coverage}",
            f"Sources:                  {', '.join(result.chunk_sources) or 'none'}",
            f"Notes:                    {result.notes}",
            "-" * 60,
            "Answer:",
            result.answer,
            "=" * 60,
        ]
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract a float score from an LLM response string."""
        import re
        matches = re.findall(r"\d+\.\d+|\d+", text.strip())
        if matches:
            score = float(matches[0])
            return min(max(score, 0.0), 1.0)
        return 0.5

    @staticmethod
    def _classify_hallucination(grounding_score: float) -> str:
        if grounding_score >= 0.7:
            return "low"
        elif grounding_score >= 0.4:
            return "medium"
        return "high"

    @staticmethod
    def _classify_coverage(relevance_score: float, n_chunks: int) -> str:
        if relevance_score >= 0.6 and n_chunks >= 2:
            return "sufficient"
        elif relevance_score >= 0.3 or n_chunks >= 1:
            return "partial"
        return "insufficient"
