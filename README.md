# DocuMind — Enterprise Document Q&A Assistant (RAG Chatbot)

A working prototype of a Retrieval-Augmented Generation chatbot that lets
employees ask natural-language questions against internal documents
(policies, manuals, reports) and get grounded, cited answers instead of
manually searching PDFs.

Built as a Low-Level Design (LLD) reference project: every layer is
behind an interface (Strategy / Factory / Repository patterns), so
swapping the embedding model, vector store, or LLM vendor never touches
calling code. See `docs/LLD_Documentation.docx` for the full design
write-up, diagrams, API contracts, and database schema.

## Why this project

- Demonstrates layered architecture, not just an ML script: ingestion,
  embedding, retrieval, generation, caching, persistence, and a REST API
  are cleanly separated.
- Runs **completely offline**, with zero API keys and zero external model
  downloads (uses TF-IDF for embeddings and an extractive fallback for
  generation) — but every interface has a documented swap-in point for
  production-grade components (neural embeddings, FAISS, Claude).
- If `ANTHROPIC_API_KEY` is set and `LLM_PROVIDER=anthropic`, the system
  automatically switches to real grounded generation via the Claude API.
- Small talk (greetings, thanks, "who are you") is detected and answered
  before retrieval ever runs — see `src/smalltalk.py`.

## Project structure

```
documind/
├── src/
│   ├── config.py            # Singleton app configuration
│   ├── ingestion.py         # DocumentLoaderFactory + Chunker (Strategy)
│   ├── embeddings.py        # Embedder interface + TF-IDF / Anthropic stub
│   ├── vector_store.py      # VectorStore interface + in-memory / FAISS stub
│   ├── retrieval.py         # Retriever (embed query -> search -> filter)
│   ├── llm_provider.py      # LLMProvider interface + extractive / Claude
│   ├── cache.py             # LRU + TTL response cache
│   ├── smalltalk.py         # Rule-based greeting/thanks/farewell detection
│   ├── rag_pipeline.py      # RAGPipeline orchestrator (Facade)
│   ├── db/
│   │   ├── models.py        # SQLAlchemy models (users, documents, chunks, conversations, messages)
│   │   ├── database.py      # Engine / session factory
│   │   └── repository.py    # Repository pattern (data access)
│   ├── api/
│   │   ├── main.py          # FastAPI app + REST endpoints
│   │   └── schemas.py       # Pydantic request/response contracts
│   └── utils/logger.py
├── static/                  # Reference web chat UI (vanilla HTML/CSS/JS, no build step)
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
├── data/sample_docs/        # Sample HR + IT security policy docs for the demo
├── docs/
│   ├── LLD_Documentation.docx
│   └── diagrams/            # architecture, class, ER, sequence diagrams
├── tests/test_pipeline.py
├── run_demo.py               # CLI demo, no server needed
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# Option A: CLI demo (no server, ingests sample docs, asks 4 questions)
python run_demo.py

# Option B: Run the REST API + web UI
uvicorn src.api.main:app --reload
# then visit http://127.0.0.1:8000        for the chat UI
#       or   http://127.0.0.1:8000/docs   for interactive Swagger UI
```

### Web UI

`uvicorn` also serves a small reference chat client at `/` (light/dark theme,
drag-and-drop document upload, citation chips with confidence scores under
every answer). It's a thin client over the same documented REST API — built
to make the API tangible for a demo, not part of the LLD's documented system
boundary (see `docs/LLD_Documentation.docx`, Section 2, for the scope this
design document covers).

### Try the API

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@data/sample_docs/hr_policy.txt"

curl -X POST http://127.0.0.1:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many remote work days are allowed per week?"}'
```

### Run tests

```bash
pytest -v
```

## Answer quality: two modes, by design

`ExtractiveLLMProvider` (the offline default) does **not** paste the whole
retrieved chunk back as the answer. It runs a second, sentence-level pass
(`select_best_sentences` in `src/llm_provider.py`) that re-ranks sentences
*within* the retrieved chunk against the query and returns only the one or
two that are actually relevant. This is a real precision improvement over
naive extraction — but it is still pattern matching, not understanding: it
can't paraphrase or combine facts across sentences the way a language
model can.

For genuinely natural, synthesized answers — "the way Claude gives" — set
`LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` (see Quick start above).
`AnthropicLLMProvider` is already fully implemented; switching to it is a
one-line config change, which is the entire point of the Strategy pattern
here (Section 5 of the LLD doc).

## Switching to production-grade components

All swaps are one-line config changes in `src/config.py` (env vars), because
every component is injected through an interface:

| Component | Demo default | Production swap |
|---|---|---|
| Embeddings | `TfidfEmbedder` (offline) | `AnthropicEmbedder` / Voyage / OpenAI / sentence-transformers |
| Vector store | `InMemoryVectorStore` | `FaissVectorStore`, or a managed service (Pinecone/Weaviate/pgvector) |
| Generation | `ExtractiveLLMProvider` (offline) | `AnthropicLLMProvider` (set `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`) |
| Database | SQLite | PostgreSQL (just change `DB_URL`) |
| Cache | In-process LRU | Redis (same `get/set` interface) |

## Design patterns used (see LLD doc for full rationale)

- **Strategy** — Chunker, Embedder, LLMProvider, VectorStore are all
  swappable implementations behind one interface.
- **Factory Method** — `DocumentLoaderFactory`, `ChunkerFactory`,
  `EmbedderFactory`, `VectorStoreFactory`, `LLMProviderFactory`.
- **Facade** — `RAGPipeline` is the single entry point the API layer
  talks to; it hides every sub-component.
- **Repository** — all DB access goes through repository classes, never
  raw queries in the API layer.
- **Singleton** — `Config` guarantees one consistent settings object
  across the process.
