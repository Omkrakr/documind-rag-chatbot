"""
run_demo.py
------------
End-to-end smoke test / live demo of the RAG pipeline, bypassing the HTTP
layer entirely (useful for quickly proving the core logic works, and for
showing an interviewer the pipeline in isolation).

Usage:
    python run_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.config import get_config
from src.embeddings import EmbedderFactory
from src.vector_store import VectorStoreFactory
from src.llm_provider import LLMProviderFactory
from src.cache import LRUCache
from src.rag_pipeline import RAGPipeline


def main():
    config = get_config()
    print("=" * 70)
    print("DocuMind RAG Pipeline -- live demo")
    print(f"Embedding provider : {config.EMBEDDING_PROVIDER}")
    print(f"LLM provider        : {config.LLM_PROVIDER}")
    print("=" * 70)

    pipeline = RAGPipeline(
        embedder=EmbedderFactory.create(config.EMBEDDING_PROVIDER),
        vector_store=VectorStoreFactory.create("memory"),
        llm_provider=LLMProviderFactory.create(config.LLM_PROVIDER, model=config.LLM_MODEL)
            if config.LLM_PROVIDER == "anthropic" else LLMProviderFactory.create("extractive"),
        cache=LRUCache(),
        chunk_size=config.CHUNK_SIZE,
        top_k=config.TOP_K,
        score_threshold=config.SIMILARITY_THRESHOLD,
    )

    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_docs")
    for fname in sorted(os.listdir(sample_dir)):
        path = os.path.join(sample_dir, fname)
        count = pipeline.ingest_document(path, fname, document_id=fname)
        print(f"\n[ingest] {fname}: {count} chunks indexed")

    # Small talk is short-circuited before retrieval -- no chunks, no LLM call
    greeting = pipeline.answer_query("hello!")
    print("\n" + "-" * 70)
    print("Q: hello!")
    print(f"A: {greeting.answer}")
    print(f"   (is_smalltalk={greeting.is_smalltalk}, sources={len(greeting.sources)})")

    questions = [
        "How many days can I work from home per week?",
        "What is the maternity leave policy?",
        "How quickly do I need to report a phishing email?",
        "What is the minimum password length?",
    ]

    for q in questions:
        result = pipeline.answer_query(q)
        print("\n" + "-" * 70)
        print(f"Q: {q}")
        print(f"A: {result.answer}")
        if result.sources:
            print(f"   (top source score: {result.sources[0].score:.3f}, "
                  f"from document '{result.sources[0].document_id}')")

    # Demonstrate cache hit on a repeated question
    repeat = pipeline.answer_query(questions[0])
    print("\n" + "-" * 70)
    print(f"Repeated query cache_hit = {repeat.cache_hit}")


if __name__ == "__main__":
    main()
