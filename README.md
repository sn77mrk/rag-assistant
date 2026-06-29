# RAG-Based AI Assistant with Agent-Like Workflows

A document question-answering system that grounds LLM answers in retrieved source context rather than relying on the model's general knowledge. Built with LangChain, ChromaDB, and OpenAI.

## What it does

- Ingests documents (PDF, TXT, DOCX) into a persistent vector database
- Answers questions using semantic retrieval + LLM generation
- Supports multi-step agent-like workflows: QA, research/summarization, cross-document comparison, and structured extraction
- Exposes everything through a FastAPI REST API
- Includes evaluation of retrieval quality, answer grounding, and hallucination risk

## Architecture

```
User Question
     │
     ▼
[FastAPI Layer]
     │
     ▼
[RAG Workflow]
  ├── SearchTool  →  ChromaDB (semantic similarity search)
  │                       │
  │                  [Embeddings]
  │             sentence-transformers / OpenAI
  │
  ├── AnswerTool  →  OpenAI GPT-4o-mini (grounded generation)
  ├── SummarizeTool
  ├── CompareTool
  └── ExtractTool
     │
     ▼
[Evaluation Layer]
  Retrieval relevance · Answer grounding · Hallucination risk
```

### Ingestion pipeline

```
File (PDF/TXT/DOCX)
    → DocumentLoader   (extract raw text per page)
    → TextCleaner      (remove noise, normalize whitespace)
    → Chunker          (split into 800-char overlapping chunks)
    → Embedder         (all-MiniLM-L6-v2 or OpenAI)
    → ChromaDB         (persist vectors to disk)
```

## Tech stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` (default, free) or OpenAI |
| Vector DB | ChromaDB (local persistence) |
| RAG framework | LangChain |
| API | FastAPI + Uvicorn |
| Document parsing | pdfplumber, python-docx |
| Testing | pytest |

## Setup

```bash
# Clone and enter directory
cd rag-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the API
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

## API endpoints

### Ingest a document
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "data/sample_docs/company_policy.txt"}'
```

### Upload a file directly
```bash
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "file=@document.pdf"
```

### Ask a question
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund window?", "workflow": "ask"}'
```

### Research / summarize
```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "shipping options", "workflow": "research"}'
```

### Compare two documents
```bash
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{
    "question": "refund policy",
    "source_a": "data/sample_docs/company_policy.txt",
    "source_b": "data/sample_docs/competitor_policy.txt"
  }'
```

### Extract structured data
```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"fields": ["refund_window", "shipping_cost", "support_hours"]}'
```

### Evaluate an answer
```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"query": "refund policy", "answer": "Returns accepted in 30 days."}'
```

## Workflow types

| Workflow | Steps | Use case |
|----------|-------|----------|
| `ask` | retrieve → answer | Standard question answering |
| `research` | retrieve (k=8) → summarize | Broad topic overview |
| `deep_answer` | retrieve with scores → filter → answer | High-precision QA with relevance check |
| `compare` | retrieve from A + B → compare | Cross-document analysis |
| `extract` | retrieve → structured JSON | Data extraction |

## Evaluation

The system evaluates every answer across four dimensions:

- **Retrieval relevance** — are retrieved chunks relevant to the query?
- **Answer grounding** — is the answer supported by context (not hallucinated)?
- **Hallucination risk** — classified as low / medium / high
- **Context coverage** — does the knowledge base have enough info?

Heuristic evaluation (no API cost) runs by default. Set `use_llm=true` for LLM-assisted scoring.

## Running tests

```bash
# Run all tests (no API key needed — LLM calls are mocked)
pytest tests/ -v

# Run only ingestion tests
pytest tests/test_ingestion.py -v
```

## Design decisions and experiments

**Chunk size (800 chars, 100 overlap)**: Experimentally chosen. Smaller chunks (400) improved retrieval precision but lost context. Larger chunks (1200) improved context but reduced search accuracy. 800 was the best tradeoff for the tested documents.

**sentence-transformers as default embedder**: `all-MiniLM-L6-v2` gives good semantic similarity without API cost. OpenAI `text-embedding-3-small` is available via config for higher quality.

**Temperature=0 for generation**: Factual grounding requires deterministic output. Creative temperature increases hallucination risk in RAG systems.

**MMR retrieval**: Maximum Marginal Relevance is available as an alternative to plain similarity search. It reduces chunk redundancy when multiple similar passages exist in the document.

**Explicit workflow steps**: Each workflow logs its steps (search → filter → answer) for traceability. This makes debugging retrieval failures easier than black-box chains.

## Limitations

- No streaming response support (single synchronous call per query)
- ChromaDB is not suited for large-scale production (use Pinecone or Weaviate for >1M chunks)
- Heuristic evaluation is approximate — LLM-assisted evaluation is more accurate but costs tokens
- Multi-turn conversation (chat history) not implemented
