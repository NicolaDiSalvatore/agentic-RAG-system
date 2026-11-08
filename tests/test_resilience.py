"""Failure injection for the LLM layer: no test here calls a real provider."""

import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage

from src import quota
from src.config import settings
from src.llm import langchain_llm as llm
from src.llm.langchain_llm import LLMTarget, generate_answer
from src.resilience import (
    PERMANENT,
    RATE_LIMITED,
    TIMEOUT,
    TRANSIENT,
    UNKNOWN,
    CircuitBreaker,
    DeadlineExceeded,
    LLMUnavailable,
    RequestContext,
    classify_error,
    ensure_deadline,
    request_scope,
)

PRIMARY = LLMTarget("groq", "primary")
BACKUP = LLMTarget("groq", "backup")


class StatusError(Exception):
    def __init__(self, status, headers=None):
        super().__init__(f"status {status}")
        self.status_code = status
        self.response = MagicMock(status_code=status, headers=headers or {})


class FakeProviders:
    """Stands in for chain building; each target fails or answers as scripted."""

    def __init__(self, **behaviors):
        self.behaviors = behaviors
        self.calls = []
        self.timeouts = []

    def build(self, target, timeout):
        def invoke(inputs, config=None):
            self.calls.append(target.model)
            self.timeouts.append(timeout)
            behavior = self.behaviors[target.model]
            if isinstance(behavior, Exception):
                raise behavior
            if callable(behavior):
                return behavior(timeout)
            return behavior

        return MagicMock(invoke=invoke)


@pytest.fixture
def two_targets(monkeypatch):
    monkeypatch.setattr(llm, "llm_targets", lambda: [PRIMARY, BACKUP])
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)


def _generate(providers):
    with patch("src.llm.langchain_llm.build_rag_chain", side_effect=providers.build):
        return generate_answer("q", None)


def _context(seconds=30.0, max_calls=6):
    return RequestContext(deadline=time.monotonic() + seconds, max_llm_calls=max_calls)


@pytest.mark.parametrize(
    "exc,expected",
    [
        (StatusError(429), RATE_LIMITED),
        (StatusError(500), TRANSIENT),
        (StatusError(503), TRANSIENT),
        (StatusError(401), PERMANENT),
        (StatusError(404), PERMANENT),
        (httpx.ReadTimeout("slow"), TIMEOUT),
        (httpx.ConnectError("down"), TRANSIENT),
        (ValueError("a bug"), UNKNOWN),
    ],
)
def test_classify_error(exc, expected):
    assert classify_error(exc) == expected


def test_primary_answers_without_touching_the_fallback(two_targets):
    providers = FakeProviders(primary="from primary", backup="from backup")
    assert _generate(providers) == "from primary"
    assert providers.calls == ["primary"]


def test_server_error_retries_then_falls_back(two_targets):
    providers = FakeProviders(primary=StatusError(500), backup="from backup")
    with request_scope(_context()):
        assert _generate(providers) == "from backup"
    assert providers.calls == [
        *["primary"] * settings.llm_max_attempts_per_provider,
        "backup",
    ]


def test_timeout_goes_straight_to_the_fallback(two_targets):
    providers = FakeProviders(primary=httpx.ReadTimeout("hung"), backup="from backup")
    with request_scope(_context()):
        assert _generate(providers) == "from backup"
    # A retry here would spend the time the fallback needs.
    assert providers.calls == ["primary", "backup"]


def test_hung_primary_leaves_time_for_the_fallback():
    # Configured values, because the budget only works if they fit together.
    assert settings.llm_attempt_timeout_seconds + settings.llm_min_call_seconds < (
        settings.request_timeout_seconds
    )


def test_attempt_timeout_is_capped_by_the_remaining_deadline(two_targets):
    providers = FakeProviders(primary="ok", backup="unused")
    with request_scope(_context(seconds=5.0)):
        _generate(providers)
    assert providers.timeouts[0] <= 5.0


def test_permanent_error_does_not_retry(two_targets):
    providers = FakeProviders(primary=StatusError(401), backup="from backup")
    with request_scope(_context()):
        assert _generate(providers) == "from backup"
    assert providers.calls == ["primary", "backup"]


def test_long_retry_after_moves_on_and_opens_the_breaker(two_targets):
    limited = StatusError(429, headers={"retry-after": "120"})
    providers = FakeProviders(primary=limited, backup="from backup")
    with request_scope(_context()):
        assert _generate(providers) == "from backup"
        assert _generate(providers) == "from backup"
    # The second query must not spend a call on a model known to be limited.
    assert providers.calls == ["primary", "backup", "backup"]


def test_offline_tools_wait_out_rate_limits_instead_of_falling_back(two_targets):
    attempts = iter([StatusError(429, headers={"retry-after": "30"}), "from primary"])

    def primary(timeout):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    providers = FakeProviders(primary=primary, backup="from backup")
    # No request scope: an evaluation run should not silently change models.
    assert _generate(providers) == "from primary"
    assert providers.calls == ["primary", "primary"]


def test_bugs_are_not_swallowed_as_provider_failures(two_targets):
    providers = FakeProviders(primary=ValueError("a bug"), backup="from backup")
    with pytest.raises(ValueError):
        _generate(providers)


def test_all_providers_failing_raises_unavailable(two_targets):
    providers = FakeProviders(primary=StatusError(500), backup=StatusError(503))
    with request_scope(_context()), pytest.raises(LLMUnavailable) as caught:
        _generate(providers)
    assert caught.value.reason == "all_providers_failed"


def test_call_budget_counts_every_attempt(two_targets):
    providers = FakeProviders(primary=StatusError(500), backup=StatusError(500))
    context = _context(max_calls=3)
    with request_scope(context), pytest.raises(LLMUnavailable) as caught:
        _generate(providers)
    assert caught.value.reason == "call_budget"
    assert len(providers.calls) == 3
    assert context.llm_calls == 3


def test_no_call_starts_without_enough_time_left(two_targets):
    providers = FakeProviders(primary="unused", backup="unused")
    with request_scope(_context(seconds=0.5)), pytest.raises(LLMUnavailable) as caught:
        _generate(providers)
    assert caught.value.reason == "deadline"
    assert providers.calls == []


def test_ensure_deadline_raises_only_after_expiry():
    with request_scope(_context(seconds=30)):
        ensure_deadline()
    with request_scope(_context(seconds=-1)), pytest.raises(DeadlineExceeded):
        ensure_deadline()
    ensure_deadline()  # Outside a request there is no deadline.


def test_token_usage_reaches_the_context_and_the_quota(two_targets):
    message = AIMessage(
        content="answer",
        usage_metadata={"input_tokens": 90, "output_tokens": 10, "total_tokens": 100},
    )
    providers = FakeProviders(primary=message, backup="unused")
    context = _context()
    with request_scope(context):
        assert _generate(providers) == "answer"
    assert (context.input_tokens, context.output_tokens) == (90, 10)
    assert quota.usage() == {"tokens": 100, "requests": 1}


def test_spent_daily_quota_stops_all_provider_calls(two_targets, monkeypatch):
    monkeypatch.setattr(settings, "llm_daily_token_budget", 100)
    quota.record(100)
    providers = FakeProviders(primary="unused", backup="unused")
    with request_scope(_context()), pytest.raises(LLMUnavailable) as caught:
        _generate(providers)
    assert caught.value.reason == "daily_quota"
    assert providers.calls == []


def test_request_budget_is_enforced_separately(monkeypatch):
    monkeypatch.setattr(settings, "llm_daily_token_budget", 0)
    monkeypatch.setattr(settings, "llm_daily_request_budget", 2)
    quota.record(0)
    assert not quota.exhausted()
    quota.record(0)
    assert quota.exhausted()


def test_zero_budgets_mean_unlimited(monkeypatch):
    monkeypatch.setattr(settings, "llm_daily_token_budget", 0)
    monkeypatch.setattr(settings, "llm_daily_request_budget", 0)
    quota.record(10**9)
    assert not quota.exhausted()


def test_quota_survives_a_restart(tmp_path):
    quota.configure(tmp_path / "restart")
    quota.record(250)
    quota.configure(tmp_path / "restart")
    assert quota.usage()["tokens"] == 250


def test_breaker_opens_fails_fast_and_probes_once(monkeypatch):
    # A controlled clock: short real sleeps are unreliable on Windows' coarse timer.
    now = [100.0]
    monkeypatch.setattr("src.resilience.time.monotonic", lambda: now[0])
    breaker = CircuitBreaker(failure_threshold=2, reset_seconds=30)

    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.CLOSED
    breaker.record_failure()
    assert breaker.state == CircuitBreaker.OPEN
    assert not breaker.allow()

    now[0] += 31
    assert breaker.state == CircuitBreaker.HALF_OPEN
    assert breaker.allow()
    assert not breaker.allow()  # Only one probe at a time.

    breaker.record_failure()
    assert breaker.state == CircuitBreaker.OPEN

    now[0] += 31
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == CircuitBreaker.CLOSED
    assert breaker.allow()


def test_open_breaker_is_skipped_without_a_call(two_targets, monkeypatch):
    monkeypatch.setattr(settings, "breaker_failure_threshold", 1)
    providers = FakeProviders(primary=httpx.ReadTimeout("hung"), backup="from backup")
    with request_scope(_context()):
        _generate(providers)
        _generate(providers)
    assert providers.calls == ["primary", "backup", "backup"]


def test_fallback_chain_parsing(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_model", "main")
    monkeypatch.setattr(settings, "openai_api_key", "")
    monkeypatch.setattr(
        settings,
        "llm_fallback_chain",
        "groq:main, groq:openai/gpt-oss-20b, openai:gpt-4o-mini, ollama:qwen2.5:1.5b, nope:x, junk",
    )
    assert llm.llm_targets() == [
        LLMTarget("groq", "main"),
        LLMTarget("groq", "openai/gpt-oss-20b"),
        # openai is skipped because it has no key; the colon in the tag survives.
        LLMTarget("ollama", "qwen2.5:1.5b"),
    ]
