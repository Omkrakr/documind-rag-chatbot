"""
api/main.py
------------
REST API Layer. Thin by design: every endpoint just validates the request
(via Pydantic schemas), calls into the RAGPipeline facade or a Repository,
and serializes the response. No business logic lives here.

Run it with:
    uvicorn src.api.main:app --reload

Auth note: a single demo user is used so the prototype is runnable without
a login flow. Swapping in real auth means adding a dependency that resolves
`current_user` from a JWT/OAuth token -- every endpoint below already takes
a `user_id`, so the change is localized to `get_current_user_id()`.
"""

from __future__ import annotations
import json
import tempfile
import os
import pathlib

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.config import get_config
from src.utils.logger import configure_logging
from src.db.database import init_db, get_session, SessionLocal
from src.db.repository import UserRepository, DocumentRepository, ChunkRepository, ConversationRepository
from src.db.models import DocumentStatus, MessageRole
from src.embeddings import EmbedderFactory
from src.vector_store import VectorStoreFactory
from src.llm_provider import LLMProviderFactory
from src.cache import LRUCache
from src.rag_pipeline import RAGPipeline
from src.api.schemas import (
    DocumentUploadResponse, DocumentStatusResponse, DocumentListResponse, ChatQueryRequest,
    ChatQueryResponse, SourceSnippet, ConversationHistoryResponse, MessageItem,
)

configure_logging()
config = get_config()

app = FastAPI(title="DocuMind - Enterprise Document Q&A Assistant", version="1.0.0")

# Reference web UI (static/index.html + css + js). This is a thin demo
# client for the API above -- it calls the same documented endpoints a
# Slack bot or internal portal would, nothing more. Resolved relative to
# this file (not the process cwd) so `uvicorn` works the same regardless
# of which directory it's launched from.
STATIC_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")

# ----------------------------------------------------------------------
# Composition root: build the RAGPipeline once at startup and inject it.
# This is where you'd point at FaissVectorStore / AnthropicLLMProvider
# instead, purely by changing config -- no code elsewhere changes.
# ----------------------------------------------------------------------
pipeline = RAGPipeline(
    embedder=EmbedderFactory.create(config.EMBEDDING_PROVIDER),
    vector_store=VectorStoreFactory.create("memory"),
    llm_provider=LLMProviderFactory.create(config.LLM_PROVIDER, model=config.LLM_MODEL)
        if config.LLM_PROVIDER == "anthropic" else LLMProviderFactory.create("extractive"),
    cache=LRUCache(max_size=config.CACHE_SIZE, ttl_seconds=config.CACHE_TTL_SECONDS),
    chunk_size=config.CHUNK_SIZE,
    top_k=config.TOP_K,
    score_threshold=config.SIMILARITY_THRESHOLD,
)

DEMO_USER_EMAIL = "demo.user@documind.local"


def get_db():
    with get_session() as session:
        yield session


def get_current_user_id(db: Session = Depends(get_db)) -> str:
    user = UserRepository(db).get_or_create(email=DEMO_USER_EMAIL, name="Demo User")
    return user.id


@app.on_event("startup")
def on_startup():
    init_db()


# ----------------------------------------------------------------------
# Document endpoints
# ----------------------------------------------------------------------
@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    doc = doc_repo.create(user_id=user_id, filename=file.filename)
    doc_repo.set_status(doc.id, DocumentStatus.PROCESSING)

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunk_count = pipeline.ingest_document(tmp_path, file.filename, document_id=doc.id)
        doc_repo.set_status(doc.id, DocumentStatus.READY, chunk_count=chunk_count)

        # Mirror chunk text into the relational DB for audit/search-by-text;
        # the embedding vectors themselves live only in the vector store.
        chunk_repo.bulk_create(doc.id, [
            {"chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "text": c.text}
            for c in pipeline._all_chunks if c.document_id == doc.id
        ])
    except Exception as exc:
        doc_repo.set_status(doc.id, DocumentStatus.FAILED)
        raise HTTPException(status_code=422, detail=f"Ingestion failed: {exc}")
    finally:
        os.unlink(tmp_path)

    return DocumentUploadResponse(
        document_id=doc.id, filename=doc.filename,
        status=doc.status.value, chunk_count=doc.chunk_count,
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db), user_id: str = Depends(get_current_user_id)):
    docs = DocumentRepository(db).list_for_user(user_id)
    return DocumentListResponse(documents=[
        DocumentStatusResponse(
            document_id=d.id, filename=d.filename, status=d.status.value,
            chunk_count=d.chunk_count, uploaded_at=d.uploaded_at,
        ) for d in docs
    ])


@app.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(document_id: str, db: Session = Depends(get_db)):
    doc = DocumentRepository(db).get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(
        document_id=doc.id, filename=doc.filename, status=doc.status.value,
        chunk_count=doc.chunk_count, uploaded_at=doc.uploaded_at,
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    doc = DocumentRepository(db).get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pipeline.delete_document(document_id)
    DocumentRepository(db).delete(document_id)
    return {"deleted": True, "document_id": document_id}


# ----------------------------------------------------------------------
# Chat endpoints
# ----------------------------------------------------------------------
@app.post("/chat/query", response_model=ChatQueryResponse)
def query_chat(
    request: ChatQueryRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    convo_repo = ConversationRepository(db)
    conversation = (
        convo_repo.get(request.conversation_id)
        if request.conversation_id else None
    ) or convo_repo.create(user_id=user_id, title=request.query[:60])

    result = pipeline.answer_query(request.query)

    sources_payload = [
        SourceSnippet(document_id=s.document_id, chunk_id=s.chunk_id, text=s.text, score=s.score)
        for s in result.sources
    ]

    convo_repo.add_message(conversation.id, MessageRole.USER.value, request.query)
    convo_repo.add_message(
        conversation.id, MessageRole.ASSISTANT.value, result.answer,
        sources=json.dumps([s.chunk_id for s in result.sources]),
    )

    return ChatQueryResponse(
        conversation_id=conversation.id, answer=result.answer,
        sources=sources_payload, cache_hit=result.cache_hit,
        is_smalltalk=result.is_smalltalk,
    )


@app.get("/chat/{conversation_id}/history", response_model=ConversationHistoryResponse)
def get_history(conversation_id: str, db: Session = Depends(get_db)):
    messages = ConversationRepository(db).history(conversation_id)
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=[
            MessageItem(role=m.role.value, content=m.content, created_at=m.created_at)
            for m in messages
        ],
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "llm_provider": config.LLM_PROVIDER,
    }
