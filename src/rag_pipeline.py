"""
rag_pipeline.py
----------------
Orchestration Layer.

Design pattern: FACADE. RAGPipeline is the single entry point the API
layer talks to. It coordinates ingestion -> embedding -> vector store ->
retrieval -> cache -> generation without exposing any of those
sub-components to the caller. This is the seam that makes the system
testable: every dependency is injected, so unit tests can swap in fakes
for the embedder, vector store, or LLM provider.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import logging

from src.ingestion import DocumentLoaderFactory, ChunkerFactory, Chunk
from src.embeddings import Embedder
from src.vector_store import VectorStore, VectorRecord
from src.retrieval import Retriever, RetrievedChunk
from src.llm_provider import LLMProvider
from src.cache import LRUCache
from src.smalltalk import SmallTalkDetector

logger = logging.getLogger("documind.pipeline")


@dataclass
class QueryResult:
    answer: str
    sources: List[RetrievedChunk]
    cache_hit: bool
    is_smalltalk: bool = False


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        cache: LRUCache,
        chunk_strategy: str = "sentence",
        chunk_size: int = 500,
        top_k: int = 4,
        score_threshold: float = 0.05,
        smalltalk_detector: SmallTalkDetector = None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.cache = cache
        self.chunker = ChunkerFactory.get_chunker(chunk_strategy, chunk_size=chunk_size)
        self.retriever = Retriever(embedder, vector_store, top_k=top_k,
                                    score_threshold=score_threshold)
        self.smalltalk_detector = smalltalk_detector or SmallTalkDetector()

        # Embedders like TF-IDF must be (re)fit on the whole corpus, so the
        # pipeline keeps the running corpus in memory to refit on each
        # ingest. A neural embedder (AnthropicEmbedder/Voyage/etc.) would
        # skip this entirely since it doesn't need fitting.
        self._corpus_texts: List[str] = []
        self._all_chunks: List[Chunk] = []

    # ----------------------------------------------------------------
    # Ingestion path
    # ----------------------------------------------------------------
    def ingest_document(self, file_path: str, filename: str, document_id: str) -> int:
        loader = DocumentLoaderFactory.get_loader(filename)
        raw_text = loader.load(file_path)

        new_chunks = self.chunker.chunk(raw_text, document_id=document_id)
        if not new_chunks:
            logger.warning("No chunks produced for document %s", document_id)
            return 0

        self._all_chunks.extend(new_chunks)

        # TF-IDF's vocabulary depends on the whole corpus, so every ingest
        # refits on all chunks seen so far and re-embeds everything, then
        # rebuilds the index. This is the one real cost of the offline
        # demo embedder: a neural/API embedder (Anthropic/Voyage/OpenAI)
        # embeds each chunk independently, so production ingestion only
        # embeds the NEW chunks and upserts incrementally -- no rebuild.
        all_texts = [c.text for c in self._all_chunks]
        self.embedder.fit(all_texts)
        all_vectors = self.embedder.embed(all_texts)

        records = [
            VectorRecord(
                vector_id=c.chunk_id,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                vector=all_vectors[i],
                text=c.text,
                metadata={"chunk_index": c.chunk_index, "filename": filename},
            )
            for i, c in enumerate(self._all_chunks)
        ]
        self.vector_store.clear()
        self.vector_store.upsert(records)
        self.cache.invalidate_all()  # corpus changed -> stale answers possible
        logger.info("Ingested %s (%d new chunks, %d total)", filename, len(new_chunks), len(self._all_chunks))
        return len(new_chunks)

    def delete_document(self, document_id: str) -> int:
        removed = self.vector_store.delete_by_document(document_id)
        self._all_chunks = [c for c in self._all_chunks if c.document_id != document_id]
        self.cache.invalidate_all()
        return removed

    # ----------------------------------------------------------------
    # Query path
    # ----------------------------------------------------------------
    def answer_query(self, query: str) -> QueryResult:
        # Checked first and unconditionally: small talk should never hit the
        # cache, the vector store, or the LLM -- it's not a document
        # question, so none of the RAG machinery is relevant to it.
        smalltalk_reply = self.smalltalk_detector.detect(query)
        if smalltalk_reply is not None:
            return QueryResult(answer=smalltalk_reply, sources=[], cache_hit=False, is_smalltalk=True)

        cache_key = self.cache.make_key(query)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return QueryResult(answer=cached, sources=[], cache_hit=True)

        if not self.embedder.is_fitted:
            return QueryResult(
                answer="No documents have been indexed yet. Upload a document first.",
                sources=[],
                cache_hit=False,
            )

        retrieved_chunks = self.retriever.retrieve(query)
        answer = self.llm_provider.generate(query, retrieved_chunks)

        self.cache.set(cache_key, answer)
        return QueryResult(answer=answer, sources=retrieved_chunks, cache_hit=False)
