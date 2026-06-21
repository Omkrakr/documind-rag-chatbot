"""
config.py
---------
Centralized configuration using the Singleton pattern.

Why a Singleton here?
A RAG pipeline has many components (chunker, embedder, vector store, LLM
provider, cache) that all need the same settings (chunk size, top-k, model
names). A Singleton guarantees every component reads the exact same config
object for the lifetime of the process, and avoids re-parsing environment
variables in five different places.
"""

import os
import threading


class Config:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        # Double-checked locking so the Singleton is thread-safe even if
        # multiple FastAPI worker threads ask for it at once.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_defaults()
        return cls._instance

    def _init_defaults(self):
        # --- Chunking ---
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))        # characters
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))   # characters

        # --- Retrieval ---
        self.TOP_K = int(os.getenv("TOP_K", 4))
        self.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.05))

        # --- Embeddings ---
        self.EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "tfidf")

        # --- Generation ---
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "extractive")  # extractive | anthropic
        self.LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

        # --- Cache ---
        self.CACHE_SIZE = int(os.getenv("CACHE_SIZE", 256))
        self.CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 600))

        # --- Storage ---
        self.DB_URL = os.getenv("DB_URL", "sqlite:///./documind.db")


# Module-level accessor so callers just do `from src.config import get_config`
def get_config() -> Config:
    return Config()
