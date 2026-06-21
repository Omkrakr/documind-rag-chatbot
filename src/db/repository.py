"""
db/repository.py
------------------
Design pattern: REPOSITORY. The API layer never imports SQLAlchemy or
writes a query directly -- it calls methods like
`document_repo.create(...)`. This keeps persistence logic in one place,
makes the API layer trivially testable with an in-memory fake repository,
and means switching ORMs later touches only this file.
"""

from __future__ import annotations
from typing import List, Optional
from sqlalchemy.orm import Session

from src.db.models import User, Document, DocumentChunk, Conversation, Message, DocumentStatus


class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_or_create(self, email: str, name: str) -> User:
        user = self.session.query(User).filter_by(email=email).first()
        if user:
            return user
        user = User(name=name, email=email)
        self.session.add(user)
        self.session.flush()
        return user


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, user_id: str, filename: str) -> Document:
        doc = Document(user_id=user_id, filename=filename, status=DocumentStatus.PENDING)
        self.session.add(doc)
        self.session.flush()
        return doc

    def set_status(self, document_id: str, status: DocumentStatus, chunk_count: int = None) -> None:
        doc = self.session.query(Document).get(document_id)
        if doc:
            doc.status = status
            if chunk_count is not None:
                doc.chunk_count = chunk_count

    def get(self, document_id: str) -> Optional[Document]:
        return self.session.query(Document).get(document_id)

    def list_for_user(self, user_id: str) -> List[Document]:
        return self.session.query(Document).filter_by(user_id=user_id).all()

    def delete(self, document_id: str) -> None:
        doc = self.session.query(Document).get(document_id)
        if doc:
            self.session.delete(doc)


class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def bulk_create(self, document_id: str, chunks: List[dict]) -> None:
        for c in chunks:
            self.session.add(DocumentChunk(
                id=c["chunk_id"], document_id=document_id,
                chunk_index=c["chunk_index"], text=c["text"],
            ))


class ConversationRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, user_id: str, title: str = "New conversation") -> Conversation:
        convo = Conversation(user_id=user_id, title=title)
        self.session.add(convo)
        self.session.flush()
        return convo

    def get(self, conversation_id: str) -> Optional[Conversation]:
        return self.session.query(Conversation).get(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str, sources: str = None) -> Message:
        msg = Message(conversation_id=conversation_id, role=role, content=content, sources=sources)
        self.session.add(msg)
        self.session.flush()
        return msg

    def history(self, conversation_id: str) -> List[Message]:
        convo = self.get(conversation_id)
        return convo.messages if convo else []
