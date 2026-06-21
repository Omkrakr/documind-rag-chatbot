"""
cache.py
--------
Response cache for the query path.

Why cache here and not at the HTTP layer?
Caching inside the pipeline (keyed on normalized query + doc-set version)
survives even if the API layer changes (e.g. REST today, gRPC tomorrow),
and lets the cache be invalidated precisely when documents change, not on
a blanket TTL alone.

Implementation: a simple OrderedDict-based LRU with TTL. For multi-instance
deployments swap this for Redis -- same interface, no caller changes.
"""

from __future__ import annotations
from collections import OrderedDict
import time
import hashlib
from typing import Optional


class LRUCache:
    def __init__(self, max_size: int = 256, ttl_seconds: int = 600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[str, tuple[float, str]]" = OrderedDict()

    @staticmethod
    def make_key(*parts: str) -> str:
        raw = "||".join(parts).lower().strip()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[str]:
        entry = self._store.get(key)
        if entry is None:
            return None
        timestamp, value = entry
        if time.time() - timestamp > self.ttl_seconds:
            del self._store[key]
            return None
        self._store.move_to_end(key)  # mark as recently used
        return value

    def set(self, key: str, value: str) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (time.time(), value)
        if len(self._store) > self.max_size:
            self._store.popitem(last=False)  # evict least recently used

    def invalidate_all(self) -> None:
        self._store.clear()
