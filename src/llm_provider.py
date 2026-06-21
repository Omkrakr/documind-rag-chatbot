"""
llm_provider.py
----------------
Generation Layer.

Design pattern: STRATEGY. LLMProvider is the interface; the pipeline
depends only on `generate(query, context_chunks) -> str`. This means the
generation backend can change (different vendor, different model, fully
offline fallback) with a one-line config change and no edits anywhere
else -- the single biggest reason production RAG systems use this pattern,
since LLM vendors and pricing shift constantly.

Two concrete implementations are provided:
- ExtractiveLLMProvider: no network calls, no API key. Returns the most
  relevant retrieved sentences directly. Used as the default so the demo
  always runs end-to-end out of the box.
- AnthropicLLMProvider: real call to the Claude API. Activates automatically
  if ANTHROPIC_API_KEY is set in the environment.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import os
import re

from src.retrieval import RetrievedChunk

_STOPWORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "of", "to", "in",
    "for", "with", "that", "this", "it", "are", "was", "were", "be", "by", "as",
    "from", "how", "what", "when", "where", "who", "does", "do", "did", "can",
    "i", "my", "me", "you", "your",
}


def _tokenize(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOPWORDS}


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def select_best_sentences(query: str, text: str, max_sentences: int = 2) -> str:
    """Re-rank sentences *within* an already-retrieved chunk by word overlap
    with the query, and return only the top few.

    Why this exists: retrieval finds the right ~500-character chunk, but a
    chunk is still a paragraph -- dumping the whole thing back as "the
    answer" reads like a copy-paste, not a response. This second, sentence-
    level pass is what turns "here's a relevant paragraph" into "here's the
    specific sentence that answers your question," without needing an LLM.
    It's a real precision improvement, not just cosmetic truncation -- but
    it is still pattern matching, not understanding: it can't paraphrase,
    combine facts across sentences, or handle a question whose answer
    isn't stated in any single sentence. That ceiling is exactly why
    AnthropicLLMProvider exists as a swap-in (see LLMProviderFactory).
    """
    query_tokens = _tokenize(query)
    sentences = _split_sentences(text)
    if not sentences:
        return text
    if not query_tokens:
        return " ".join(sentences[:max_sentences])

    scored = [(len(query_tokens & _tokenize(s)), s) for s in sentences]
    best_score = max(score for score, _ in scored)
    if best_score == 0:
        # nothing overlapped meaningfully -- fall back to the chunk's start
        # rather than returning nothing
        return " ".join(sentences[:max_sentences])

    top = {s for score, s in scored if score > 0}
    # keep original sentence order for readability, cap at max_sentences
    ordered = [s for s in sentences if s in top][:max_sentences]
    return " ".join(ordered)


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, query: str, context_chunks: List[RetrievedChunk]) -> str:
        raise NotImplementedError


class ExtractiveLLMProvider(LLMProvider):
    """Zero-dependency fallback: no network call, no API key. Two-stage
    extraction -- retrieval already found the right chunk; this picks the
    specific sentence(s) inside it that actually answer the query (see
    select_best_sentences) -- so the reply reads like a focused answer
    instead of a pasted paragraph. It still can't paraphrase or combine
    facts the way a real LLM can; that tradeoff is the whole point of
    keeping this provider swappable (see AnthropicLLMProvider below)."""

    def generate(self, query: str, context_chunks: List[RetrievedChunk]) -> str:
        if not context_chunks:
            return ("I couldn't find anything relevant in the indexed documents "
                    "to answer that.")

        best = context_chunks[0]
        focused = select_best_sentences(query, best.text, max_sentences=2)
        filename = (best.metadata or {}).get("filename", best.document_id)

        answer = f"{focused}\n\n(Source: {filename})"
        if len(context_chunks) > 1:
            answer += (f"\n\nI found {len(context_chunks) - 1} more related passage(s) "
                       f"— see the sources below.")
        return answer


class AnthropicLLMProvider(LLMProvider):
    """Real generation backend using the Claude API. Builds a grounded
    prompt from retrieved chunks so the model answers strictly from the
    provided context and can decline when context is insufficient."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")

        import anthropic  # local import: keeps this an optional dependency
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, query: str, context_chunks: List[RetrievedChunk]) -> str:
        context_text = "\n\n".join(
            f"[Source {i+1} | doc:{c.document_id}]\n{c.text}"
            for i, c in enumerate(context_chunks)
        ) or "No relevant context was retrieved."

        system_prompt = (
            "You are DocuMind, an internal knowledge assistant. Answer the user's "
            "question in your own words, the way a knowledgeable colleague would "
            "explain it, using ONLY the provided context. Synthesize across "
            "passages naturally -- do not quote large blocks verbatim. If the "
            "context does not contain the answer, say so plainly instead of "
            "guessing. Cite sources inline like [Source 1] so the user knows "
            "where each claim comes from."
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

        response = self._client.messages.create(
            model=self.model,
            max_tokens=600,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class LLMProviderFactory:
    @staticmethod
    def create(provider: str, **kwargs) -> LLMProvider:
        if provider == "anthropic":
            return AnthropicLLMProvider(**kwargs)
        return ExtractiveLLMProvider()
