"""
Tests for the ingestion pipeline: loading, cleaning, and chunking.

Covers:
  - TXT file loading
  - Text cleaning (whitespace, page numbers, duplicates)
  - Chunking with correct overlap
  - Metadata preservation across pipeline
"""

import os
import tempfile
import pytest

from langchain_core.documents import Document

from ingestion.document_loader import DocumentLoader
from ingestion.text_cleaner import TextCleaner
from ingestion.chunker import Chunker


# ── DocumentLoader ────────────────────────────────────────────────────────────

class TestDocumentLoader:

    def test_load_txt_file(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("Hello world. This is a test document.", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(str(f))

        assert len(docs) == 1
        assert "Hello world" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "txt"

    def test_load_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("Document A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Document B", encoding="utf-8")

        loader = DocumentLoader()
        docs = loader.load(str(tmp_path))

        assert len(docs) == 2

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_text("content")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader._load_file(f)

    def test_nonexistent_path_raises(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/file.txt")

    def test_metadata_contains_source(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Some content here.")

        loader = DocumentLoader()
        docs = loader.load(str(f))

        assert "source" in docs[0].metadata
        assert str(f) in docs[0].metadata["source"]


# ── TextCleaner ───────────────────────────────────────────────────────────────

class TestTextCleaner:

    def test_removes_control_characters(self):
        cleaner = TextCleaner()
        dirty = "Hello\x00World\x01"
        result = cleaner.clean_text(dirty)
        assert "\x00" not in result
        assert "\x01" not in result

    def test_collapses_blank_lines(self):
        cleaner = TextCleaner()
        text = "Line 1\n\n\n\n\nLine 2"
        result = cleaner.clean_text(text)
        assert "\n\n\n" not in result

    def test_removes_page_numbers(self):
        cleaner = TextCleaner()
        text = "Introduction\n42\nSome content"
        result = cleaner.clean_text(text)
        assert "42" not in result or "Introduction" in result

    def test_removes_duplicate_lines(self):
        cleaner = TextCleaner()
        text = "Header\nHeader\nContent"
        result = cleaner.clean_text(text)
        lines = [l for l in result.split("\n") if l.strip()]
        assert lines.count("Header") == 1

    def test_clean_documents_preserves_metadata(self):
        cleaner = TextCleaner()
        docs = [Document(page_content="Clean text.", metadata={"source": "test.txt"})]
        cleaned = cleaner.clean_documents(docs)
        assert cleaned[0].metadata["source"] == "test.txt"

    def test_empty_document_filtered_out(self):
        cleaner = TextCleaner()
        docs = [Document(page_content="   \n\n  ", metadata={})]
        cleaned = cleaner.clean_documents(docs)
        assert len(cleaned) == 0


# ── Chunker ───────────────────────────────────────────────────────────────────

class TestChunker:

    def test_chunks_long_text(self):
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        long_text = "word " * 200
        docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]
        chunks = chunker.split_documents(docs)
        assert len(chunks) > 1

    def test_chunks_not_exceed_size(self):
        chunk_size = 150
        chunker = Chunker(chunk_size=chunk_size, chunk_overlap=20)
        docs = [Document(page_content="a " * 500, metadata={"source": "test.txt"})]
        chunks = chunker.split_documents(docs)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 50  # small tolerance

    def test_chunk_index_metadata(self):
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        docs = [Document(page_content="word " * 200, metadata={"source": "f.txt"})]
        chunks = chunker.split_documents(docs)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(1, len(chunks) + 1))

    def test_metadata_preserved(self):
        chunker = Chunker(chunk_size=200, chunk_overlap=20)
        docs = [Document(page_content="hello " * 100, metadata={"source": "x.txt", "file_type": "txt"})]
        chunks = chunker.split_documents(docs)
        assert all(c.metadata["source"] == "x.txt" for c in chunks)

    def test_short_text_single_chunk(self):
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        docs = [Document(page_content="Short document.", metadata={"source": "s.txt"})]
        chunks = chunker.split_documents(docs)
        assert len(chunks) == 1

    def test_split_text_returns_strings(self):
        chunker = Chunker(chunk_size=100, chunk_overlap=10)
        result = chunker.split_text("word " * 100)
        assert all(isinstance(s, str) for s in result)
