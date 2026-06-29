"""
Tests for prompt builder, evaluator, and agent workflows.

LLM-dependent tests are mocked — they test prompt construction and
workflow orchestration logic without requiring an OpenAI API key.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Refunds are accepted within 30 days.",
            metadata={"source": "policy.pdf", "page": 1},
        ),
        Document(
            page_content="Original receipt required for all returns.",
            metadata={"source": "policy.pdf", "page": 2},
        ),
    ]


# ── PromptBuilder ─────────────────────────────────────────────────────────────

class TestPromptBuilder:

    def test_qa_prompt_contains_question(self, sample_docs):
        from generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        inputs = builder.build_qa_prompt("What is the refund window?", sample_docs)
        assert inputs["question"] == "What is the refund window?"
        assert "context" in inputs

    def test_qa_context_contains_source_labels(self, sample_docs):
        from generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        inputs = builder.build_qa_prompt("test", sample_docs)
        assert "Source 1" in inputs["context"]
        assert "policy.pdf" in inputs["context"]

    def test_summary_prompt_has_focus(self, sample_docs):
        from generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        inputs = builder.build_summary_prompt(sample_docs, focus="pricing")
        assert inputs["focus"] == "pricing"
        assert "context" in inputs

    def test_extract_prompt_contains_fields(self, sample_docs):
        from generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        inputs = builder.build_extract_prompt(sample_docs, ["author", "date"])
        assert "author" in inputs["fields"]
        assert "date" in inputs["fields"]

    def test_compare_prompt_has_both_contexts(self, sample_docs):
        from generation.prompt_builder import PromptBuilder
        builder = PromptBuilder()
        inputs = builder.build_compare_prompt("compare terms", sample_docs[:1], sample_docs[1:])
        assert "context_a" in inputs
        assert "context_b" in inputs


# ── RAGEvaluator ──────────────────────────────────────────────────────────────

class TestRAGEvaluator:

    def test_heuristic_evaluation_runs(self, sample_docs):
        from evaluation.evaluator import RAGEvaluator
        evaluator = RAGEvaluator(llm_client=None)
        result = evaluator.evaluate(
            query="What is the refund window?",
            answer="Refunds are accepted within 30 days.",
            context_docs=sample_docs,
        )
        assert 0.0 <= result.retrieval_relevance_score <= 1.0
        assert 0.0 <= result.answer_grounding_score <= 1.0
        assert result.hallucination_risk in ("low", "medium", "high")
        assert result.context_coverage in ("sufficient", "partial", "insufficient")

    def test_high_grounding_low_hallucination(self, sample_docs):
        from evaluation.evaluator import RAGEvaluator
        evaluator = RAGEvaluator()
        result = evaluator.evaluate(
            query="refund policy",
            answer="Refunds are accepted within 30 days. Original receipt required.",
            context_docs=sample_docs,
        )
        # Answer uses words from context, grounding should be decent
        assert result.answer_grounding_score > 0.3

    def test_aggregate_metrics(self, sample_docs):
        from evaluation.evaluator import RAGEvaluator, EvaluationResult
        evaluator = RAGEvaluator()
        results = [
            EvaluationResult(
                query="q1", answer="a1",
                n_chunks_retrieved=3,
                retrieval_relevance_score=0.8,
                answer_grounding_score=0.9,
                hallucination_risk="low",
                context_coverage="sufficient",
                chunk_sources=["doc.txt"],
            ),
            EvaluationResult(
                query="q2", answer="a2",
                n_chunks_retrieved=2,
                retrieval_relevance_score=0.4,
                answer_grounding_score=0.5,
                hallucination_risk="medium",
                context_coverage="partial",
                chunk_sources=["doc.txt"],
            ),
        ]
        metrics = evaluator.aggregate_metrics(results)
        assert metrics["n_evaluated"] == 2
        assert metrics["avg_retrieval_relevance"] == pytest.approx(0.6, abs=0.01)

    def test_format_report_contains_fields(self, sample_docs):
        from evaluation.evaluator import RAGEvaluator
        evaluator = RAGEvaluator()
        result = evaluator.evaluate("test query", "test answer", sample_docs)
        report = evaluator.format_report(result)
        assert "Query" in report
        assert "Hallucination" in report
        assert "grounding" in report


# ── RAGWorkflow (mocked LLM) ──────────────────────────────────────────────────

class TestRAGWorkflow:

    @pytest.fixture
    def mock_retriever(self, sample_docs):
        retriever = MagicMock()
        retriever.retrieve.return_value = sample_docs
        retriever.retrieve_with_scores.return_value = [(d, 0.5) for d in sample_docs]
        return retriever

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.answer.return_value = "Refunds accepted within 30 days."
        llm.summarize.return_value = "Summary: refunds within 30 days, receipt required."
        llm.compare.return_value = "Document A allows 30-day returns. Document B allows 14-day returns."
        llm.extract.return_value = {"refund_window": "30 days", "requirement": "receipt"}
        return llm

    def test_ask_workflow_returns_answer(self, mock_retriever, mock_llm):
        from agent.workflow import RAGWorkflow
        workflow = RAGWorkflow(mock_retriever, mock_llm)
        result = workflow.ask("What is the refund policy?")
        assert result.success
        assert "30 days" in result.answer
        assert result.workflow == "ask"

    def test_research_workflow_returns_summary(self, mock_retriever, mock_llm):
        from agent.workflow import RAGWorkflow
        workflow = RAGWorkflow(mock_retriever, mock_llm)
        result = workflow.research("refund policy")
        assert result.success
        assert result.workflow == "research"

    def test_extract_workflow_returns_dict(self, mock_retriever, mock_llm):
        from agent.workflow import RAGWorkflow
        workflow = RAGWorkflow(mock_retriever, mock_llm)
        result = workflow.extract(fields=["refund_window", "requirement"])
        assert isinstance(result.answer, dict)
        assert result.workflow == "extract"

    def test_compare_workflow_has_two_sources(self, mock_retriever, mock_llm):
        from agent.workflow import RAGWorkflow
        workflow = RAGWorkflow(mock_retriever, mock_llm)
        result = workflow.compare("refund policy", "doc_a.pdf", "doc_b.pdf")
        assert "doc_a.pdf" in result.sources
        assert "doc_b.pdf" in result.sources

    def test_empty_retrieval_graceful_response(self, mock_llm):
        from agent.workflow import RAGWorkflow
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        workflow = RAGWorkflow(retriever, mock_llm)
        result = workflow.ask("question with no matching docs")
        assert "No relevant documents" in result.answer

    def test_workflow_steps_are_logged(self, mock_retriever, mock_llm):
        from agent.workflow import RAGWorkflow
        workflow = RAGWorkflow(mock_retriever, mock_llm)
        result = workflow.ask("test question")
        assert len(result.steps) >= 2
        step_names = [s["step"] for s in result.steps]
        assert "search" in step_names
        assert "answer" in step_names
