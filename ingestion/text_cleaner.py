"""
Text cleaner: normalizes raw extracted text before chunking.

Why this matters:
  Raw PDF/DOCX text often contains broken formatting, repeated whitespace,
  header/footer artifacts, and unicode noise. Cleaning before chunking
  prevents garbage from polluting retrieved context and reduces token waste.
"""

import re
from typing import List

from langchain_core.documents import Document


class TextCleaner:
    """
    Cleans raw document text extracted from PDFs, DOCX, and TXT files.

    Operations applied (in order):
        1. Remove null bytes and control characters
        2. Normalize unicode whitespace
        3. Collapse repeated blank lines (> 2 → 2)
        4. Remove lines that are purely numeric (page numbers)
        5. Strip leading/trailing whitespace per line
        6. Remove duplicate consecutive lines (common in PDFs with headers)
    """

    def clean_documents(self, documents: List[Document]) -> List[Document]:
        """Apply cleaning to a list of LangChain Documents, preserving metadata."""
        cleaned = []
        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)
            if cleaned_text.strip():
                cleaned.append(Document(page_content=cleaned_text, metadata=doc.metadata))
        return cleaned

    def clean_text(self, text: str) -> str:
        """Clean a single string of extracted text."""
        # 1. Remove null bytes and non-printable control chars (except \n \t)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 2. Normalize unicode whitespace (non-breaking space, etc.)
        text = text.replace(" ", " ").replace("’", "'").replace("“", '"').replace("”", '"')

        # 3. Normalize Windows line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 4. Strip trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]

        # 5. Remove lines that are purely numeric (page numbers)
        lines = [line for line in lines if not re.fullmatch(r"\s*\d+\s*", line)]

        # 6. Remove duplicate consecutive lines (repeated headers/footers)
        deduped = []
        prev = None
        for line in lines:
            if line != prev:
                deduped.append(line)
            prev = line
        lines = deduped

        # 7. Collapse more than 2 consecutive blank lines to 2
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
