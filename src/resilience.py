"""Request deadlines, LLM-call budgets, error classification and circuit breaking."""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import httpx


class DeadlineExceeded(Exception):
    """The request ran out of time before it could produce any answer."""


class LLMUnavailable(RuntimeError):
    """No provider could serve the call; ``reason`` says why, for metrics."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or "The language model is temporarily unavailable.")
        self.reason = reason


@dataclass
class RequestContext:
    """Per-request limits and accounting, shared by the nodes and the LLM layer.

    Graph state cannot carry this: LangGraph only applies a node's return value,
    so a counter updated inside the LLM layer would never reach it. ``deadline``
    is a ``time.monotonic()`` timestamp; ``None`` means no limit, which is how
    offline tools such as the evaluation run behave.
    """

    deadline: float | None = None
    max_llm_calls: int | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    models_used: list[str] = field(default_factory=list)

    def remaining(self) -> float | None:
        """Return seconds left before the deadline, or ``None`` without one."""
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def expired(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0.0

    def budget_exhausted(self) -> bool:
        return self.max_llm_calls is not None and self.llm_calls >= self.max_llm_calls


_current: contextvars.ContextVar[RequestContext | None] = contextvars.ContextVar(
    "rag_request_context", default=None
)


def current_context() -> RequestContext | None:
    """Return the active request's context, or ``None`` outside a request."""
    return _current.get()


@contextmanager
def request_scope(context: RequestContext) -> Iterator[RequestContext]:
    """Make ``context`` the active request context for the enclosed block."""
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


def ensure_deadline() -> None:
    """Raise ``DeadlineExceeded`` when the active request is out of time."""
    context = current_context()
    if context is not None and context.expired():
        raise DeadlineExceeded("The request deadline passed.")


RATE_LIMITED = "rate_limited"
TIMEOUT = "timeout"
TRANSIENT = "transient"
PERMANENT = "permanent"
UNKNOWN = "unknown"

_TRANSIENT_STATUSES = {408, 409, 500, 502, 503, 504}
_PROVIDER_PACKAGES = {"groq", "openai", "httpx", "httpcore"}


def classify_error(exc: BaseException) -> str:
    """Classify a provider failure so callers know whether a retry can help.

    Args:
        exc: The exception raised by a provider call.

    Returns:
        One of the classification constants. ``UNKNOWN`` means the exception did
        not come from a provider, so it is a bug that must not be swallowed.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) or getattr(exc, "status_code", None)
    if status == 429:
        return RATE_LIMITED
    if status in _TRANSIENT_STATUSES:
        return TRANSIENT
    if isinstance(status, int):
        return PERMANENT
    if isinstance(exc, httpx.TimeoutException) or "Timeout" in type(exc).__name__:
        return TIMEOUT
    if isinstance(exc, httpx.TransportError) or "Connection" in type(exc).__name__:
        return TRANSIENT
    # A provider SDK error with no status, such as a missing API key.
    if type(exc).__module__.split(".")[0] in _PROVIDER_PACKAGES:
        return PERMANENT
    return UNKNOWN


class CircuitBreaker:
    """Stop calling a failing target, then let one probe test its recovery."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int, reset_seconds: float) -> None:
        self._failure_threshold = failure_threshold
        self._reset_seconds = reset_seconds
        self._failures = 0
        self._open_until: float | None = None
        self._probing = False
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._open_until is None:
                return self.CLOSED
            return self.OPEN if time.monotonic() < self._open_until else self.HALF_OPEN

    def allow(self) -> bool:
        """Return whether a call may proceed, claiming the probe when half-open."""
        with self._lock:
            if self._open_until is None:
                return True
            if time.monotonic() < self._open_until or self._probing:
                return False
            self._probing = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = None
            self._probing = False

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._probing or self._failures >= self._failure_threshold:
                self._open_until = time.monotonic() + self._reset_seconds
            self._probing = False

    def open_for(self, seconds: float) -> None:
        """Open immediately for a provider-stated wait, such as ``Retry-After``."""
        with self._lock:
            self._open_until = time.monotonic() + seconds
            self._probing = False


_breakers: dict[tuple[str, str], CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def breaker_for(provider: str, model: str, threshold: int, reset_seconds: float) -> CircuitBreaker:
    """Return the shared breaker for one provider and model.

    Keyed by model as well as provider because providers rate-limit per model:
    an exhausted primary model must not block a fallback on the same provider.
    """
    key = (provider, model)
    with _breakers_lock:
        if key not in _breakers:
            _breakers[key] = CircuitBreaker(threshold, reset_seconds)
        return _breakers[key]


def reset_breakers() -> None:
    """Forget all breaker state. Used by tests."""
    with _breakers_lock:
        _breakers.clear()
