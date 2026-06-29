"""
Prompt builder: constructs structured prompts for different RAG use cases.

Why prompt structure matters:
  In RAG systems, the LLM output is only as good as the context it receives
  and how the prompt directs the model to use that context.

  Key design decisions:
    - Explicit grounding instruction: "answer ONLY based on the provided context"
      reduces hallucination risk
    - Source citation instruction: helps with factual accountability
    - Separate system and user prompts: allows model to maintain consistent behavior
    - Task-specific prompt templates: QA vs. summarization need different framing
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# ── Base system message ──────────────────────────────────────────────────────

SYSTEM_BASE = (
    "You are a precise document assistant. "
    "You answer questions strictly based on the provided context. "
    "Do NOT use general knowledge or information not present in the context. "
    "If the context does not contain enough information to answer, say: "
    "'I cannot find a sufficient answer in the provided documents.' "
    "When possible, mention which chunk or source your answer comes from."
)

# ── Prompt templates ─────────────────────────────────────────────────────────

QA_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_BASE),
    ("human", (
        "Context from retrieved documents:\n\n"
        "{context}\n\n"
        "---\n\n"
        "Question: {question}\n\n"
        "Provide a clear, factual answer based only on the context above."
    )),
])

SUMMARIZE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a professional summarizer. "
        "Summarize the provided document context into a clear, structured summary. "
        "Cover the main points, key facts, and conclusions."
    )),
    ("human", (
        "Document content:\n\n"
        "{context}\n\n"
        "---\n\n"
        "Produce a concise summary. "
        "If there is a specific focus requested, apply it: {focus}"
    )),
])

COMPARE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an analytical assistant. "
        "Compare the provided document sections carefully. "
        "Use only the provided content — do not invent information."
    )),
    ("human", (
        "Document A context:\n\n{context_a}\n\n"
        "---\n\n"
        "Document B context:\n\n{context_b}\n\n"
        "---\n\n"
        "Comparison request: {question}\n\n"
        "Provide a structured comparison with clear differences and similarities."
    )),
])

EXTRACT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a structured data extractor. "
        "Extract the requested information from the provided context. "
        "Return the result as a JSON object. "
        "Only include fields that are explicitly present in the context. "
        "Use null for missing fields."
    )),
    ("human", (
        "Context:\n\n{context}\n\n"
        "---\n\n"
        "Extract the following fields as JSON: {fields}\n\n"
        "Return only valid JSON."
    )),
])


# ── Builder class ────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Constructs prompts for different RAG workflow types.

    Why this is separate from the LLM client:
        Prompt structure is a research variable. Swapping or tuning prompts
        should not require touching LLM configuration. Separation also makes
        prompt A/B testing straightforward.
    """

    def build_qa_prompt(self, question: str, context_docs: List[Document]) -> dict:
        """Build prompt inputs for standard question-answering."""
        context = self._format_context(context_docs)
        return {"question": question, "context": context}

    def build_summary_prompt(self, context_docs: List[Document], focus: str = "general") -> dict:
        """Build prompt inputs for document summarization."""
        context = self._format_context(context_docs)
        return {"context": context, "focus": focus}

    def build_compare_prompt(
        self,
        question: str,
        docs_a: List[Document],
        docs_b: List[Document],
    ) -> dict:
        """Build prompt inputs for cross-document comparison."""
        return {
            "question": question,
            "context_a": self._format_context(docs_a),
            "context_b": self._format_context(docs_b),
        }

    def build_extract_prompt(self, context_docs: List[Document], fields: List[str]) -> dict:
        """Build prompt inputs for structured information extraction."""
        context = self._format_context(context_docs)
        return {"context": context, "fields": ", ".join(fields)}

    def _format_context(self, docs: List[Document]) -> str:
        """Format document chunks with source labels for the LLM."""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            label = f"[Source {i}: {source}" + (f", page {page}" if page else "") + "]"
            parts.append(f"{label}\n{doc.page_content}")
        return "\n\n".join(parts)
