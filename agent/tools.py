"""
Agent tools: callable units of work that the workflow can route to.

Why tool-based design:
  Decomposing agent behavior into discrete tools mirrors production agent systems
  (LangChain agents, OpenAI function calling). Each tool has a clear interface,
  can be tested independently, and can be composed into multi-step workflows.

  Tool routing logic lives in workflow.py — this file only defines what each
  tool does, not when to use it.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


@dataclass
class ToolResult:
    """Structured result returned by any tool."""
    tool_name: str
    output: Any
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class SearchTool:
    """
    Tool: retrieve relevant document chunks for a query.

    This is the core retrieval step in every RAG workflow.
    """

    name = "search"
    description = "Search the document knowledge base for relevant chunks."

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, query: str, k: int = None, source: str = None) -> ToolResult:
        try:
            docs = self.retriever.retrieve(query, k=k, source_filter=source)
            sources = list({d.metadata.get("source", "") for d in docs})
            return ToolResult(
                tool_name=self.name,
                output=docs,
                sources=sources,
                metadata={"query": query, "n_results": len(docs)},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, output=[], error=str(e))


class AnswerTool:
    """
    Tool: generate a grounded answer from retrieved context.
    """

    name = "answer"
    description = "Generate an LLM answer grounded in retrieved document context."

    def __init__(self, llm_client):
        self.llm = llm_client

    def run(self, question: str, context_docs: List[Document]) -> ToolResult:
        try:
            answer = self.llm.answer(question, context_docs)
            sources = list({d.metadata.get("source", "") for d in context_docs})
            return ToolResult(
                tool_name=self.name,
                output=answer,
                sources=sources,
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, output="", error=str(e))


class SummarizeTool:
    """
    Tool: summarize document chunks, optionally with a focus area.
    """

    name = "summarize"
    description = "Summarize retrieved document sections."

    def __init__(self, llm_client):
        self.llm = llm_client

    def run(self, context_docs: List[Document], focus: str = "general") -> ToolResult:
        try:
            summary = self.llm.summarize(context_docs, focus=focus)
            sources = list({d.metadata.get("source", "") for d in context_docs})
            return ToolResult(
                tool_name=self.name,
                output=summary,
                sources=sources,
                metadata={"focus": focus},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, output="", error=str(e))


class CompareTool:
    """
    Tool: compare two document sets on a specific question.
    """

    name = "compare"
    description = "Compare two sets of retrieved document chunks."

    def __init__(self, llm_client):
        self.llm = llm_client

    def run(
        self,
        question: str,
        docs_a: List[Document],
        docs_b: List[Document],
    ) -> ToolResult:
        try:
            comparison = self.llm.compare(question, docs_a, docs_b)
            sources_a = list({d.metadata.get("source", "") for d in docs_a})
            sources_b = list({d.metadata.get("source", "") for d in docs_b})
            return ToolResult(
                tool_name=self.name,
                output=comparison,
                sources=sources_a + sources_b,
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, output="", error=str(e))


class ExtractTool:
    """
    Tool: extract structured fields from retrieved context as JSON.
    """

    name = "extract"
    description = "Extract structured information fields from document context."

    def __init__(self, llm_client):
        self.llm = llm_client

    def run(self, context_docs: List[Document], fields: List[str]) -> ToolResult:
        try:
            extracted = self.llm.extract(context_docs, fields)
            sources = list({d.metadata.get("source", "") for d in context_docs})
            return ToolResult(
                tool_name=self.name,
                output=extracted,
                sources=sources,
                metadata={"requested_fields": fields},
            )
        except Exception as e:
            return ToolResult(tool_name=self.name, output={}, error=str(e))
