"""
LLM client: wraps Groq (via OpenAI-compatible API) for RAG generation.

Groq provides free access to open-source models (Llama 3.3, Mixtral) through
an OpenAI-compatible interface, so LangChain's ChatOpenAI works without changes —
we just point base_url to Groq's endpoint.

Temperature=0 ensures deterministic, grounded answers (less creative drift).
"""

import json
import logging
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import config
from .prompt_builder import (
    PromptBuilder,
    QA_TEMPLATE,
    SUMMARIZE_TEMPLATE,
    COMPARE_TEMPLATE,
    EXTRACT_TEMPLATE,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    High-level interface for LLM-based generation in RAG workflows.
    Uses Groq's free API with Llama 3.3 70B by default.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        self.prompt_builder = PromptBuilder()

    def answer(self, question: str, context_docs: List[Any]) -> str:
        """Generate an answer grounded in retrieved context."""
        inputs = self.prompt_builder.build_qa_prompt(question, context_docs)
        chain = QA_TEMPLATE | self.llm
        response = chain.invoke(inputs)
        return response.content

    def summarize(self, context_docs: List[Any], focus: str = "general") -> str:
        """Summarize retrieved document chunks."""
        inputs = self.prompt_builder.build_summary_prompt(context_docs, focus)
        chain = SUMMARIZE_TEMPLATE | self.llm
        response = chain.invoke(inputs)
        return response.content

    def compare(self, question: str, docs_a: List[Any], docs_b: List[Any]) -> str:
        """Compare two sets of document chunks on a given question."""
        inputs = self.prompt_builder.build_compare_prompt(question, docs_a, docs_b)
        chain = COMPARE_TEMPLATE | self.llm
        response = chain.invoke(inputs)
        return response.content

    def extract(self, context_docs: List[Any], fields: List[str]) -> Dict:
        """Extract structured fields from document context as a JSON object."""
        inputs = self.prompt_builder.build_extract_prompt(context_docs, fields)
        chain = EXTRACT_TEMPLATE | self.llm
        response = chain.invoke(inputs)

        try:
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON for extraction. Returning raw.")
            return {"raw": response.content}

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Low-level chat completion for custom prompts."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = self.llm.invoke(messages)
        return response.content
