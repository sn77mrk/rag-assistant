"""
Retriever: high-level interface for fetching relevant context.

Why a separate Retriever class:
  VectorStore is about storage mechanics. Retriever is about retrieval strategy.
  Separating them lets us swap retrieval strategies (simple similarity, MMR,
  hybrid search) without touching the storage layer.

  Implemented strategies:
    - similarity: plain cosine/L2 nearest-neighbor
    - mmr: Maximum Marginal Relevance — balances relevance with diversity
           to avoid returning 5 near-identical chunks
"""

from typing import List, Literal, Optional

from langchain_core.documents import Document

from config import config
from .vector_store import VectorStore


class Retriever:
    """
    Retrieves relevant document chunks for a given query.

    Attributes:
        store:    The underlying VectorStore.
        strategy: "similarity" or "mmr"
        top_k:    Number of chunks to return.
    """

    def __init__(
        self,
        store: VectorStore,
        strategy: Literal["similarity", "mmr"] = "similarity",
        top_k: int = None,
    ):
        self.store = store
        self.strategy = strategy
        self.top_k = top_k or config.RETRIEVAL_TOP_K

    def retrieve(
        self,
        query: str,
        k: int = None,
        source_filter: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:         The user's question.
            k:             Override the default top_k.
            source_filter: Restrict retrieval to a specific document source path.

        Returns:
            List of the most relevant Document chunks.
        """
        k = k or self.top_k
        filter_dict = {"source": source_filter} if source_filter else None

        if self.strategy == "mmr":
            retriever = self.store.db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 3, "filter": filter_dict},
            )
            return retriever.invoke(query)
        else:
            return self.store.similarity_search(query, k=k, filter=filter_dict)

    def retrieve_with_scores(
        self,
        query: str,
        k: int = None,
    ) -> List[tuple]:
        """
        Retrieve chunks along with their similarity scores.

        Useful for evaluation: low scores may indicate the document doesn't
        contain relevant information, which is a signal the LLM may hallucinate.

        Returns:
            List of (Document, float) tuples.
        """
        k = k or self.top_k
        return self.store.similarity_search_with_score(query, k=k)

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a single context string for the LLM prompt.

        Each chunk is labelled with its source and chunk index so the LLM can
        reference it in its answer (source citation).

        Returns:
            A formatted multi-chunk context string.
        """
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            page_str = f", page {page}" if page else ""
            parts.append(
                f"[Chunk {i} | Source: {source}{page_str}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)
