"""A process-wide daily LLM budget that survives restarts.

Per-query limits cannot protect a free tier: one client making allowed requests
can still spend a whole day's provider quota in minutes. Once the budget is spent
the LLM layer stops calling providers, so the service answers extractively
instead of collecting 429s.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from diskcache import Cache

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger("rag.quota")

_EXPIRE_SECONDS = 3 * 24 * 3600
_STORE_ERRORS = (sqlite3.Error, OSError)

_cache: Cache | None = None
_cache_path: Path | None = None
_lock = threading.Lock()


def configure(path: Path | None) -> None:
    """Point the quota at another store (``None`` restores the configured one)."""
    global _cache, _cache_path
    with _lock:
        if _cache is not None:
            _cache.close()
        _cache = None
        _cache_path = path


def _store() -> Cache:
    global _cache
    with _lock:
        if _cache is None:
            path = _cache_path or (PROJECT_ROOT / settings.llm_quota_path)
            path.mkdir(parents=True, exist_ok=True)
            _cache = Cache(str(path))
        return _cache


def _key(name: str) -> str:
    return f"{datetime.now(UTC).date().isoformat()}:{name}"


def usage() -> dict[str, int]:
    """Return today's spend, in UTC days. An unreadable store reads as no spend."""
    try:
        store = _store()
        return {
            "tokens": int(store.get(_key("tokens"), 0)),
            "requests": int(store.get(_key("requests"), 0)),
        }
    except _STORE_ERRORS:
        logger.exception("Could not read the daily LLM quota.")
        return {"tokens": 0, "requests": 0}


def exhausted() -> bool:
    """Return whether today's token or request budget is spent (0 = unlimited)."""
    tokens, requests = settings.llm_daily_token_budget, settings.llm_daily_request_budget
    if not tokens and not requests:
        return False
    spent = usage()
    return bool(
        (tokens and spent["tokens"] >= tokens)
        or (requests and spent["requests"] >= requests)
    )


def record(tokens: int) -> None:
    """Add one provider request and its token count to today's spend.

    A failure here must not lose an answer the provider has already returned.
    """
    try:
        store = _store()
        for name, amount in (("requests", 1), ("tokens", tokens)):
            key = _key(name)
            store.incr(key, amount, default=0)
            store.touch(key, expire=_EXPIRE_SECONDS)
    except _STORE_ERRORS:
        logger.exception("Could not record LLM usage in the daily quota.")
