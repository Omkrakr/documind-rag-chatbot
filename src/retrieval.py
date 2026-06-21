"""
retrieval.py
------------
Retrieval Layer. Wraps Embedder + VectorStore behind one call:
`retrieve(query) -> List[RetrievedChunk]`.

Kept as its own class (rather than folded into the pipeline) because in
production this is where you'd add re-ranking (e.g. a cross-encoder),
hybrid search (BM25 + vector), or metadata filtering -- all without
touching ingestion or generation code.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from src.embeddings import Embedder
from src.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore,
                 top_k: int = 4, score_threshold: float = 0.05):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        query_vector = self.embedder.embed([query])[0]
        results = self.vector_store.search(query_vector, top_k=self.top_k)

        retrieved = [
            RetrievedChunk(
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                text=record.text,
                score=score,
                metadata=record.metadata,
            )
            for record, score in results
            if score >= self.score_threshold
        ]
        return retrieved
