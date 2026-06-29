"""
Document loader: extracts raw text from PDF, TXT, DOCX files.

Why this exists:
  Before any retrieval can happen, documents must be converted to plain text.
  Different file types require different extraction strategies. This module
  handles that complexity and returns a unified List[Document] format
  compatible with LangChain.
"""

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document


class DocumentLoader:
    """
    Loads documents from various file formats and returns LangChain Documents.

    Supported formats:
        - .txt  — direct read
        - .pdf  — pdfplumber (preserves table structure better than PyPDF2)
        - .docx — python-docx paragraph extraction

    Each returned Document carries metadata:
        source   : original file path
        file_type: extension
        page     : page number (for PDFs)
    """

    SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}

    def load(self, path: str) -> List[Document]:
        """
        Load a single file or all supported files in a directory.

        Args:
            path: Path to a file or directory.

        Returns:
            List of LangChain Document objects.

        Raises:
            ValueError: If the file type is not supported.
            FileNotFoundError: If the path does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if p.is_dir():
            docs = []
            for file in sorted(p.rglob("*")):
                if file.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    docs.extend(self._load_file(file))
            return docs

        return self._load_file(p)

    def _load_file(self, path: Path) -> List[Document]:
        ext = path.suffix.lower()
        if ext == ".txt":
            return self._load_txt(path)
        elif ext == ".pdf":
            return self._load_pdf(path)
        elif ext == ".docx":
            return self._load_docx(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _load_txt(self, path: Path) -> List[Document]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": str(path), "file_type": "txt"})]

    def _load_pdf(self, path: Path) -> List[Document]:
        import pdfplumber

        docs = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": str(path), "file_type": "pdf", "page": i + 1},
                        )
                    )
        return docs

    def _load_docx(self, path: Path) -> List[Document]:
        from docx import Document as DocxDocument

        doc = DocxDocument(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)
        return [Document(page_content=full_text, metadata={"source": str(path), "file_type": "docx"})]
