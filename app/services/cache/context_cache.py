import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.core.errors import CacheContextMissingError


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class InMemoryContextCache:
    """Small Redis-shaped cache used by the minimal runnable implementation."""

    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, CacheEntry] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = CacheEntry(value=value, expires_at=time.time() + self.ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return entry.value

    def require(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise CacheContextMissingError(f"Cache context not found: {key}")
        return value

