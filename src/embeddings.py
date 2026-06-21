"""
embeddings.py
-------------
Embedding Layer.

Design pattern: STRATEGY. Embedder is an interface; TfidfEmbedder is the
offline, dependency-free default used by this demo. AnthropicEmbedder and
SentenceTransformerEmbedder are shown as drop-in replacements for
production -- swapping providers never requires touching the retriever,
vector store, or pipeline code, only the Strategy passed in at startup.

Why TF-IDF for the demo instead of a neural embedder?
This prototype is meant to run anywhere with zero external downloads and
zero API keys, so an interview reviewer can clone it and run it in seconds.
TF-IDF is a legitimate lightweight baseline; the interface is what matters
for the architecture, not the specific vectors.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """Some embedders (like TF-IDF) need to see the corpus first."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) float32 matrix."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    def __init__(self, max_features: int = 4096):
        self._vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        self._vectorizer.fit(corpus)
        self._fitted = True

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed().")
        matrix = self._vectorizer.transform(texts).toarray().astype("float32")
        return matrix

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class AnthropicEmbedder(Embedder):
    """
    Production stub. Anthropic's API is generation-focused, so in practice
    you'd pair Claude for generation with a dedicated embedding model
    (Voyage AI embeddings are Anthropic's recommended partner, or
    OpenAI / Cohere / a local sentence-transformers model). This class
    documents the swap point -- implement `embed()` to call that provider's
    embeddings endpoint and the rest of the pipeline is unaffected.
    """

    def __init__(self, model: str = "voyage-3"):
        self.model = model
        self._fitted = True  # neural embedders don't need corpus fitting

    def fit(self, corpus: List[str]) -> None:
        pass  # no-op: stateless API-based embedder

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError(
            "Call your embeddings provider's API here and return a float32 matrix."
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class EmbedderFactory:
    @staticmethod
    def create(provider: str) -> Embedder:
        if provider == "tfidf":
            return TfidfEmbedder()
        if provider == "anthropic":
            return AnthropicEmbedder()
        raise ValueError(f"Unknown embedding provider: {provider}")
