"""
Chunker: splits cleaned documents into smaller overlapping chunks for retrieval.

Why this matters:
  Embedding models have a token limit (~512 tokens for most models).
  More importantly, retrieval works best with focused, self-contained text
  fragments — not whole documents. The overlap between chunks prevents
  information at chunk boundaries from being lost.

  Chunk size vs. retrieval quality tradeoff:
    - Smaller chunks → more precise retrieval, but may miss context
    - Larger chunks → more context, but less precision
    - Overlap → prevents boundary loss, small cost in storage

  Default: 800 tokens, 100 token overlap (good starting point for most docs).
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import config


class Chunker:
    """
    Splits LangChain Documents into smaller chunks using recursive character splitting.

    RecursiveCharacterTextSplitter tries to split on paragraph boundaries first
    (\n\n), then sentence boundaries (\n), then spaces, and finally characters.
    This produces more semantically coherent chunks than simple fixed-size splitting.
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Documents into chunks.

        Metadata is preserved from the parent document. Each chunk gets
        an additional `chunk_index` field in its metadata.

        Args:
            documents: List of cleaned LangChain Documents.

        Returns:
            List of chunk Documents, each ≤ chunk_size characters.
        """
        chunks = self.splitter.split_documents(documents)

        # Annotate each chunk with its position
        source_counter: dict = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            source_counter[source] = source_counter.get(source, 0) + 1
            chunk.metadata["chunk_index"] = source_counter[source]
            chunk.metadata["chunk_size"] = self.chunk_size
            chunk.metadata["chunk_overlap"] = self.chunk_overlap

        return chunks

    def split_text(self, text: str) -> List[str]:
        """
        Convenience method: split a raw string, return list of chunk strings.
        Useful for quick experimentation without Document objects.
        """
        return self.splitter.split_text(text)
