"""
db/models.py
------------
Relational schema for everything that ISN'T a vector (the vector store
holds embeddings; this DB holds business/operational data: who uploaded
what, conversation history, audit trail).

Tables:
  users              -- account that owns documents & conversations
  documents          -- one row per uploaded file, tracks ingestion status
  document_chunks    -- metadata for each chunk (text lives here too, for
                        a quick reference; the embedding vector itself
                        lives only in the vector store, keyed by chunk_id)
  conversations       -- a chat session
  messages           -- one row per turn (user question or assistant answer)
"""

from sqlalchemy import (
    Column, String, Integer, Text, DateTime, ForeignKey, Enum
)
from sqlalchemy.orm import declarative_base, relationship
import enum
import datetime
import uuid

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="owner", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False)
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=_uuid)  # == vector_id in the vector store
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String(255), default="New conversation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation",
                             cascade="all, delete-orphan", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=_uuid)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON-encoded list of chunk_ids cited
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")
