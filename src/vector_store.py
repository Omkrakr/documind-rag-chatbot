"""
vector_store.py
----------------
Vector Storage Layer.

Design pattern: STRATEGY / ADAPTER. VectorStore is the interface the rest
of the system codes against. InMemoryVectorStore is a pure NumPy cosine-
similarity index -- zero infra, good for the demo and for unit tests.
FaissVectorStore / PineconeVectorStore stubs show the swap points for
production scale (millions of vectors, persistence, approximate nearest
neighbor search).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class VectorRecord:
    vector_id: str
    chunk_id: str
    document_id: str
    vector: np.ndarray
    text: str
    metadata: dict


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, records: List[VectorRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[VectorRecord, float]]:
        """Return [(record, similarity_score), ...] sorted best-first."""
        raise NotImplementedError

    @abstractmethod
    def delete_by_document(self, document_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Used when an embedder must re-embed the whole corpus (e.g. TF-IDF
        after a vocabulary change) and the index needs a full rebuild."""
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self._records: List[VectorRecord] = []

    def upsert(self, records: List[VectorRecord]) -> None:
        self._records.extend(records)

    def clear(self) -> None:
        self._records = []

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[VectorRecord, float]]:
        if not self._records:
            return []

        matrix = np.stack([r.vector for r in self._records])  # (n, dim)
        scores = self._cosine_similarity(query_vector, matrix)

        ranked_idx = np.argsort(-scores)[:top_k]
        return [(self._records[i], float(scores[i])) for i in ranked_idx]

    def delete_by_document(self, document_id: str) -> int:
        before = len(self._records)
        self._records = [r for r in self._records if r.document_id != document_id]
        return before - len(self._records)

    @staticmethod
    def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query) + 1e-8
        matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-8
        return (matrix @ query) / (query_norm * matrix_norm)


class FaissVectorStore(VectorStore):
    """
    Production stub. Swap in for the in-memory store once the corpus grows
    past what fits comfortably in a single process -- FAISS (or a managed
    service like Pinecone/Weaviate/pgvector) gives approximate nearest
    neighbor search, persistence, and horizontal scaling. The interface
    above is identical, so the rest of the pipeline needs zero changes.
    """

    def __init__(self, dim: int):
        raise NotImplementedError("Wire up faiss-cpu / faiss-gpu here for production scale.")

    def upsert(self, records):
        raise NotImplementedError

    def search(self, query_vector, top_k):
        raise NotImplementedError

    def delete_by_document(self, document_id):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError


class VectorStoreFactory:
    @staticmethod
    def create(backend: str = "memory", **kwargs) -> VectorStore:
        if backend == "memory":
            return InMemoryVectorStore()
        if backend == "faiss":
            return FaissVectorStore(**kwargs)
        raise ValueError(f"Unknown vector store backend: {backend}")
