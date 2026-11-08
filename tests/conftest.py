"""Shared fixtures: keep process-wide LLM state from leaking between tests."""

import pytest

from src import quota
from src.config import settings
from src.observability import tracing
from src.resilience import reset_breakers


@pytest.fixture(autouse=True)
def _isolated_llm_state(tmp_path, monkeypatch):
    """Give each test closed breakers, an empty quota and no real Langfuse client."""
    reset_breakers()
    quota.configure(tmp_path / "quota")
    # A developer's .env may enable Langfuse; tests must never send traces to it.
    monkeypatch.setattr(settings, "langfuse_enabled", False)
    tracing.reset_client(None)
    yield
    quota.configure(None)
    reset_breakers()
