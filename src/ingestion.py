"""
ingestion.py
------------
Document Ingestion Layer.

Design patterns used:
- FACTORY METHOD (DocumentLoaderFactory): picks the right loader class based
  on file extension, so the rest of the system never has to know if a file
  is .txt, .pdf, or .docx.
- STRATEGY (Chunker / FixedSizeChunker / SentenceAwareChunker): chunking
  algorithm is interchangeable at runtime without touching calling code.

Both patterns exist for the same reason: new file types or new chunking
strategies should be addable by writing one new class, not by editing
existing logic (Open/Closed Principle).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
import re
import uuid


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Loaders (Factory Method)
# --------------------------------------------------------------------------
class DocumentLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> str:
        """Return raw text content of the document."""
        raise NotImplementedError


class TxtLoader(DocumentLoader):
    def load(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


class MarkdownLoader(DocumentLoader):
    def load(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        # strip the most common markdown noise so chunks read cleanly
        text = re.sub(r"[#*`>_-]{1,}", " ", text)
        return text


class PdfLoader(DocumentLoader):
    """
    Production note: implement with a library such as pypdf or
    pdfplumber. Kept as a stub here so the Factory's contract is visible
    without pulling a heavy PDF dependency into the demo.
    """

    def load(self, path: str) -> str:
        raise NotImplementedError(
            "Wire up pypdf/pdfplumber here for production PDF support."
        )


class DocumentLoaderFactory:
    _registry = {
        ".txt": TxtLoader,
        ".md": MarkdownLoader,
        ".pdf": PdfLoader,
    }

    @classmethod
    def get_loader(cls, filename: str) -> DocumentLoader:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        loader_cls = cls._registry.get(ext)
        if loader_cls is None:
            raise ValueError(f"No loader registered for extension '{ext}'")
        return loader_cls()

    @classmethod
    def register(cls, ext: str, loader_cls: type) -> None:
        """Lets new file types be plugged in without modifying this class."""
        cls._registry[ext] = loader_cls


# --------------------------------------------------------------------------
# Chunkers (Strategy)
# --------------------------------------------------------------------------
class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        raise NotImplementedError


class FixedSizeChunker(Chunker):
    """Simple sliding-window chunker. Fast, predictable, good baseline."""

    def __init__(self, chunk_size: int = 500, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        text = re.sub(r"\s+", " ", text).strip()
        chunks: List[Chunk] = []
        start, index = 0, 0
        step = max(self.chunk_size - self.overlap, 1)
        while start < len(text):
            piece = text[start:start + self.chunk_size].strip()
            if piece:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        text=piece,
                        chunk_index=index,
                    )
                )
                index += 1
            start += step
        return chunks


class SentenceAwareChunker(Chunker):
    """
    Groups whole sentences into chunks up to chunk_size, instead of cutting
    mid-sentence. Slightly slower, noticeably better retrieval quality
    because chunks stay semantically coherent.
    """

    _SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, chunk_size: int = 500, overlap_sentences: int = 1):
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        text = re.sub(r"\s+", " ", text).strip()
        sentences = [s for s in self._SENTENCE_SPLIT.split(text) if s]

        chunks: List[Chunk] = []
        current: List[str] = []
        current_len = 0
        index = 0

        def flush():
            nonlocal current, current_len, index
            if current:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        text=" ".join(current),
                        chunk_index=index,
                    )
                )
                index += 1

        for sentence in sentences:
            if current_len + len(sentence) > self.chunk_size and current:
                flush()
                # carry the last N sentences forward for context continuity
                current = current[-self.overlap_sentences:]
                current_len = sum(len(s) for s in current)
            current.append(sentence)
            current_len += len(sentence)

        flush()
        return chunks


class ChunkerFactory:
    """Selects a chunking Strategy by name (kept separate from loader factory
    on purpose: file type and chunking strategy vary independently)."""

    _registry = {
        "fixed": FixedSizeChunker,
        "sentence": SentenceAwareChunker,
    }

    @classmethod
    def get_chunker(cls, strategy: str, **kwargs) -> Chunker:
        chunker_cls = cls._registry.get(strategy, SentenceAwareChunker)
        return chunker_cls(**kwargs)
