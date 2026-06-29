"""
Vector store: manages ChromaDB persistence and document indexing.

Why ChromaDB:
  - Local, no external service required
  - Persistent across restarts (writes to disk)
  - Native LangChain integration
  - Supports metadata filtering for multi-document setups

Why embeddings matter:
  The quality of retrieval depends entirely on how well the embedding model
  captures semantic similarity. We support two backends:
    - sentence-transformers (free, local, good quality)
    - OpenAI text-embedding-3-small (higher quality, requires API key)
"""

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import config


def _build_embeddings():
    """
    Factory: return the configured embedding model.

    Choosing the backend:
        EMBEDDING_BACKEND=sentence-transformers  → no API key, runs locally
        EMBEDDING_BACKEND=openai                 → requires OPENAI_API_KEY
    """
    if config.EMBEDDING_BACKEND == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=config.SENTENCE_TRANSFORMER_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


class VectorStore:
    """
    Wraps ChromaDB with a document-ingestion and similarity-search interface.

    Usage:
        store = VectorStore()
        store.add_documents(chunks)
        results = store.similarity_search("what is the refund policy?", k=5)
    """

    def __init__(self):
        self.embeddings = _build_embeddings()
        self.db = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )

    def add_documents(self, documents: List[Document]) -> int:
        """
        Embed and store a list of chunk Documents.

        Returns:
            Number of documents added.
        """
        if not documents:
            return 0
        self.db.add_documents(documents)
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Retrieve top-k most semantically similar chunks for a query.

        Args:
            query:  User question or search string.
            k:      Number of results to return.
            filter: Optional ChromaDB metadata filter, e.g. {"source": "report.pdf"}.

        Returns:
            List of Document chunks sorted by relevance (highest first).
        """
        k = k or config.RETRIEVAL_TOP_K
        return self.db.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
        filter: Optional[dict] = None,
    ) -> List[tuple]:
        """
        Like similarity_search, but also returns similarity scores.

        Returns:
            List of (Document, score) tuples. Lower score = more similar in
            ChromaDB's default L2 distance metric.
        """
        k = k or config.RETRIEVAL_TOP_K
        return self.db.similarity_search_with_score(query, k=k, filter=filter)

    def delete_collection(self) -> None:
        """Delete all documents from the collection (useful for testing)."""
        self.db.delete_collection()

    def count(self) -> int:
        """Return the number of documents currently in the store."""
        return self.db._collection.count()

    def as_retriever(self, k: int = None):
        """Return a LangChain-compatible retriever interface."""
        k = k or config.RETRIEVAL_TOP_K
        return self.db.as_retriever(search_kwargs={"k": k})
