"""
tests/test_pipeline.py
-----------------------
Unit tests targeting the seams created by the Strategy/Factory/Repository
patterns -- each layer can be tested in isolation because every dependency
is injected rather than hard-coded.

Run with:  pytest -v
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ingestion import FixedSizeChunker, SentenceAwareChunker, DocumentLoaderFactory
from src.embeddings import TfidfEmbedder
from src.vector_store import InMemoryVectorStore, VectorRecord
from src.retrieval import Retriever
from src.llm_provider import ExtractiveLLMProvider, select_best_sentences
from src.smalltalk import SmallTalkDetector
from src.cache import LRUCache
from src.rag_pipeline import RAGPipeline


# --------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------
def test_fixed_size_chunker_respects_overlap():
    text = "A" * 1000
    chunker = FixedSizeChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk(text, document_id="doc1")
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_sentence_aware_chunker_keeps_sentences_whole():
    text = "First sentence here. Second sentence here. Third one too."
    chunker = SentenceAwareChunker(chunk_size=1000)
    chunks = chunker.chunk(text, document_id="doc1")
    assert len(chunks) == 1
    assert chunks[0].text.count(".") == 3


def test_loader_factory_raises_on_unknown_extension():
    try:
        DocumentLoaderFactory.get_loader("file.unknownext")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_txt_loader_reads_file_content():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        path = f.name
    loader = DocumentLoaderFactory.get_loader("sample.txt")
    assert loader.load(path) == "hello world"
    os.unlink(path)


# --------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------
def test_retriever_returns_best_match_first():
    embedder = TfidfEmbedder()
    corpus = ["cats are great pets", "dogs are loyal animals", "the stock market fell today"]
    embedder.fit(corpus)
    vectors = embedder.embed(corpus)

    store = InMemoryVectorStore()
    store.upsert([
        VectorRecord(vector_id=str(i), chunk_id=str(i), document_id="d1",
                     vector=vectors[i], text=corpus[i], metadata={})
        for i in range(len(corpus))
    ])

    retriever = Retriever(embedder, store, top_k=2, score_threshold=0.0)
    results = retriever.retrieve("tell me about pet cats")
    assert results[0].text == "cats are great pets"


def test_retriever_filters_below_threshold():
    embedder = TfidfEmbedder()
    corpus = ["completely unrelated text about gardening"]
    embedder.fit(corpus)
    vectors = embedder.embed(corpus)

    store = InMemoryVectorStore()
    store.upsert([VectorRecord(vector_id="1", chunk_id="1", document_id="d1",
                                vector=vectors[0], text=corpus[0], metadata={})])

    retriever = Retriever(embedder, store, top_k=5, score_threshold=0.99)
    results = retriever.retrieve("quantum physics")
    assert results == []


# --------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------
def test_cache_hit_and_eviction():
    cache = LRUCache(max_size=2, ttl_seconds=600)
    k1, k2, k3 = cache.make_key("q1"), cache.make_key("q2"), cache.make_key("q3")

    cache.set(k1, "answer1")
    cache.set(k2, "answer2")
    assert cache.get(k1) == "answer1"

    cache.set(k3, "answer3")  # should evict k2 (least recently used after k1 was touched)
    assert cache.get(k2) is None
    assert cache.get(k1) == "answer1"
    assert cache.get(k3) == "answer3"


# --------------------------------------------------------------------
# Full pipeline (extractive provider, no network/API key needed)
# --------------------------------------------------------------------
def _build_pipeline():
    return RAGPipeline(
        embedder=TfidfEmbedder(),
        vector_store=InMemoryVectorStore(),
        llm_provider=ExtractiveLLMProvider(),
        cache=LRUCache(),
        chunk_size=200,
        top_k=3,
        score_threshold=0.0,
    )


def test_pipeline_end_to_end_answers_from_ingested_doc():
    pipeline = _build_pipeline()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("The company offers 18 days of paid leave per year.")
        path = f.name

    chunk_count = pipeline.ingest_document(path, "leave.txt", document_id="d1")
    assert chunk_count >= 1

    result = pipeline.answer_query("how many paid leave days are offered?")
    assert "18 days" in result.answer
    assert result.cache_hit is False

    repeat = pipeline.answer_query("how many paid leave days are offered?")
    assert repeat.cache_hit is True

    os.unlink(path)


def test_pipeline_handles_query_before_any_ingestion():
    pipeline = _build_pipeline()
    result = pipeline.answer_query("anything?")
    assert "No documents" in result.answer


# --------------------------------------------------------------------
# Small talk
# --------------------------------------------------------------------
def test_smalltalk_detector_catches_common_greetings():
    detector = SmallTalkDetector()
    for greeting in ["hi", "Hello!", "hey there", "good morning"]:
        assert detector.detect(greeting) is not None


def test_smalltalk_detector_catches_thanks_variants():
    detector = SmallTalkDetector()
    for phrase in ["thanks", "thanks a lot", "thank you so much", "thx", "cheers"]:
        assert detector.detect(phrase) is not None, f"expected '{phrase}' to be detected as smalltalk"


def test_smalltalk_detector_ignores_real_questions():
    detector = SmallTalkDetector()
    # A real question must never be swallowed by a small-talk match, even
    # if it happens to start with a greeting-like word.
    assert detector.detect("hi, what is the leave policy?") is None
    assert detector.detect("how many remote days are allowed?") is None


def test_pipeline_short_circuits_smalltalk_without_touching_retrieval():
    pipeline = _build_pipeline()
    result = pipeline.answer_query("hello!")
    assert result.is_smalltalk is True
    assert result.sources == []
    assert result.cache_hit is False


def test_pipeline_does_not_cache_smalltalk():
    pipeline = _build_pipeline()
    pipeline.answer_query("hi")
    # A real document question with the same cache key space must still
    # be evaluated fresh -- smalltalk must never pollute the answer cache.
    assert pipeline.cache.get(pipeline.cache.make_key("hi")) is None


# --------------------------------------------------------------------
# Sentence-level extraction (offline answer quality)
# --------------------------------------------------------------------
def test_select_best_sentences_picks_relevant_sentence_not_whole_chunk():
    chunk = (
        "The office has a rooftop garden. Employees get 18 days of paid leave "
        "per year. The cafeteria serves lunch until 3 PM."
    )
    focused = select_best_sentences("how many paid leave days", chunk, max_sentences=1)
    assert "18 days" in focused
    assert "rooftop garden" not in focused


def test_select_best_sentences_falls_back_when_no_overlap():
    chunk = "First sentence. Second sentence. Third sentence."
    focused = select_best_sentences("completely unrelated query xyz", chunk, max_sentences=1)
    assert focused  # falls back to the start of the chunk rather than empty


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
