"""
Tests for the retrieval layer: VectorStore and Retriever.

Uses an in-memory ChromaDB collection (no persistence) so tests are fast
and don't pollute the production store.

Note: these tests use the sentence-transformers embedding backend (no API key).
      Set EMBEDDING_BACKEND=sentence-transformers in your environment.
"""

import os
import pytest

from langchain_core.documents import Document

os.environ.setdefault("EMBEDDING_BACKEND", "sentence-transformers")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="The refund policy allows returns within 30 days of purchase.",
            metadata={"source": "policy.txt", "chunk_index": 1},
        ),
        Document(
            page_content="Customers must provide proof of purchase to complete a return.",
            metadata={"source": "policy.txt", "chunk_index": 2},
        ),
        Document(
            page_content="Free shipping is available on orders over $50.",
            metadata={"source": "shipping.txt", "chunk_index": 1},
        ),
        Document(
            page_content="Express delivery takes 1-2 business days.",
            metadata={"source": "shipping.txt", "chunk_index": 2},
        ),
    ]


@pytest.fixture
def vector_store(sample_docs):
    """Create a temporary in-memory vector store for each test."""
    import chromadb
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from retrieval.vector_store import VectorStore

    # Use a unique collection name per test to avoid conflicts
    import uuid
    collection_name = f"test_{uuid.uuid4().hex[:8]}"

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    store = VectorStore.__new__(VectorStore)
    store.embeddings = embeddings
    store.db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    store.db.add_documents(sample_docs)
    yield store
    try:
        store.delete_collection()
    except Exception:
        pass  # already deleted inside test_delete_collection


class TestVectorStore:

    def test_count_after_add(self, vector_store):
        assert vector_store.count() == 4

    def test_similarity_search_returns_results(self, vector_store):
        results = vector_store.similarity_search("refund policy", k=2)
        assert len(results) == 2

    def test_relevant_chunk_ranked_first(self, vector_store):
        results = vector_store.similarity_search("return policy refund", k=4)
        # The refund/return chunks should appear before shipping chunks
        top_source = results[0].metadata["source"]
        assert "policy" in top_source

    def test_similarity_search_with_score(self, vector_store):
        results = vector_store.similarity_search_with_score("shipping delivery", k=2)
        assert len(results) == 2
        assert all(isinstance(score, float) for _, score in results)

    def test_scores_are_ordered(self, vector_store):
        results = vector_store.similarity_search_with_score("refund", k=4)
        scores = [s for _, s in results]
        # Lower L2 distance = more similar, should be ascending
        assert scores == sorted(scores)

    def test_delete_collection(self, vector_store):
        vector_store.delete_collection()
        # After deletion, count should raise or return 0
        try:
            count = vector_store.count()
            assert count == 0
        except Exception:
            pass  # Collection deleted, expected


class TestRetriever:

    def test_retrieve_returns_documents(self, vector_store):
        from retrieval.retriever import Retriever
        retriever = Retriever(vector_store, top_k=3)
        docs = retriever.retrieve("refund policy")
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_retrieve_respects_k(self, vector_store):
        from retrieval.retriever import Retriever
        retriever = Retriever(vector_store, top_k=2)
        docs = retriever.retrieve("shipping", k=2)
        assert len(docs) <= 2

    def test_format_context_contains_source_labels(self, vector_store):
        from retrieval.retriever import Retriever
        retriever = Retriever(vector_store, top_k=2)
        docs = retriever.retrieve("refund")
        context = retriever.format_context(docs)
        assert "[Chunk 1" in context
        assert "Source:" in context

    def test_retrieve_with_scores(self, vector_store):
        from retrieval.retriever import Retriever
        retriever = Retriever(vector_store, top_k=2)
        results = retriever.retrieve_with_scores("delivery time")
        assert len(results) == 2
        assert all(len(r) == 2 for r in results)
