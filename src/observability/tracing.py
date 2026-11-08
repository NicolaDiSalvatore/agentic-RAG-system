"""Optional Langfuse tracing. Any failure here must never affect a request."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from src.config import settings

logger = logging.getLogger("rag.tracing")

_client: Any = None
_client_failed = False


# Fields this service sets itself and the mask leaves alone. Only full_text can
# hold content, and it is only set on the query span when the setting allows it.
_SAFE_METADATA = frozenset(
    {
        "request_id",
        "tenant",
        "route",
        "answer_status",
        "fallback",
        "llm_calls",
        "index_version",
        "redacted_text",
        "full_text",
    }
)


def _redact(*, data: Any, **_: Any) -> Any:
    """Replace text with its length so traces keep their shape but not content.

    Langfuse applies one mask to input, output and metadata alike, so without the
    allowlist a trace could not show even which route a query took.
    """
    if isinstance(data, str):
        return f"[redacted {len(data)} chars]"
    if isinstance(data, dict):
        return {
            key: value if key in _SAFE_METADATA else _redact(data=value)
            for key, value in data.items()
        }
    if isinstance(data, (list, tuple)):
        return [_redact(data=value) for value in data]
    return data


def _query_text(text: str | None) -> Any:
    """Return what the query span records: the text and its length, or just the text to mask."""
    if text is None or not settings.langfuse_capture_query_text:
        return text
    return {"redacted_text": _redact(data=text), "full_text": text}


def get_client() -> Any:
    """Return the shared Langfuse client, or ``None`` when tracing is off."""
    global _client, _client_failed
    if _client is not None or _client_failed or not settings.langfuse_enabled:
        return _client
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_host,
            sample_rate=settings.langfuse_sample_rate,
            mask=None if settings.langfuse_capture_content else _redact,
        )
    except Exception:
        _client_failed = True
        logger.exception("Langfuse could not start; continuing without tracing.")
    return _client


def reset_client(client: Any = None) -> None:
    """Replace the shared client. Used by tests."""
    global _client, _client_failed
    _client = client
    _client_failed = False


def callbacks() -> list[Any]:
    """Return LangChain callbacks that attach LLM calls to the current trace."""
    if get_client() is None:
        return []
    try:
        from langfuse.langchain import CallbackHandler

        return [CallbackHandler()]
    except Exception:
        logger.exception("Langfuse callback could not start.")
        return []


def tenant_hash(tenant_id: str) -> str:
    """Return a stable, non-reversible label for a tenant."""
    return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()[:16]


class QueryTrace:
    """Handle for recording a query's outcome on its trace."""

    def __init__(self, client: Any = None) -> None:
        self._client = client

    def finish(
        self, metadata: dict, scores: dict[str, float], output: str | None = None
    ) -> None:
        if self._client is None:
            return
        try:
            self._client.update_current_span(metadata=metadata, output=_query_text(output))
            for name, value in scores.items():
                self._client.score_current_trace(name=name, value=value)
        except Exception:
            logger.exception("Could not record the Langfuse trace outcome.")


@contextmanager
def query_trace(
    request_id: str, tenant_id: str, question: str | None = None
) -> Iterator[QueryTrace]:
    """Open one trace around a query, or do nothing when tracing is off."""
    client = get_client()
    if client is None:
        yield QueryTrace()
        return
    try:
        span = client.start_as_current_observation(
            name="query",
            as_type="span",
            input=_query_text(question),
            metadata={"request_id": request_id, "tenant": tenant_hash(tenant_id)},
        )
        entered = span.__enter__()
    except Exception:
        logger.exception("Could not open a Langfuse trace.")
        yield QueryTrace()
        return
    try:
        yield QueryTrace(client)
    except BaseException as exc:
        try:
            entered.update(level="ERROR", status_message=type(exc).__name__)
        except Exception:
            logger.exception("Could not mark the Langfuse trace as failed.")
        raise
    finally:
        try:
            span.__exit__(None, None, None)
        except Exception:
            logger.exception("Could not close the Langfuse trace.")


def shutdown() -> None:
    """Flush buffered traces on process exit."""
    if _client is None:
        return
    try:
        _client.shutdown()
    except Exception:
        logger.exception("Could not flush Langfuse.")
