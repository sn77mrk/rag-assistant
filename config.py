"""
Centralized configuration for the RAG assistant.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Groq LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # Embedding backend: "sentence-transformers" (free/local) or "openai"
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"

    # Vector store
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Retrieval
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # LLM generation
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))


config = Config()
