"""
api/schemas.py
----------------
Pydantic models define the API contract. Keeping them separate from the
SQLAlchemy models in db/models.py is deliberate: the DB schema and the
wire format are allowed to evolve independently (e.g. hiding internal
fields from responses, renaming a column without breaking clients).
"""

from pydantic import BaseModel, Field
from typing import List, Optional
import datetime


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunk_count: int
    uploaded_at: datetime.datetime


class DocumentListResponse(BaseModel):
    documents: List[DocumentStatusResponse]


class ChatQueryRequest(BaseModel):
    conversation_id: Optional[str] = Field(None, description="Omit to start a new conversation")
    query: str = Field(..., min_length=1, max_length=2000)


class SourceSnippet(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    score: float


class ChatQueryResponse(BaseModel):
    conversation_id: str
    answer: str
    sources: List[SourceSnippet]
    cache_hit: bool
    is_smalltalk: bool = False


class MessageItem(BaseModel):
    role: str
    content: str
    created_at: datetime.datetime


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: List[MessageItem]
