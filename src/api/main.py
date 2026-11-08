import asyncio
import hmac
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from src.graph.build_graph import build_graph
from src.config import settings
from src.indexing.build_index import active_index_version, index_is_ready
from src.observability import metrics, tracing
from src.resilience import DeadlineExceeded, LLMUnavailable, RequestContext, request_scope
from src.schemas import Citation

logger = logging.getLogger("rag.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# One worker per allowed in-flight query. A plain threading semaphore is used
# because the slot must be released by the graph thread when it actually
# finishes, not by an event-loop callback that may no longer be running.
MAX_CONCURRENT_QUERIES = max(1, settings.max_concurrent_queries)
executor = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_QUERIES, thread_name_prefix="rag-query"
)
query_slots = threading.BoundedSemaphore(MAX_CONCURRENT_QUERIES)
graph = build_graph()

# The graph gets the request timeout as its deadline. The HTTP wait runs slightly
# longer, so an answer finished right at the deadline can still be returned.
DEADLINE_GRACE_SECONDS = 1.0

# "idle" means warm-up never ran, so requests build what they need on demand.
_warmup_state = "idle"


def _warm_up() -> None:
    """Load the index and build the retriever before the first request needs them.

    Opening the index and building BM25 over the corpus can take longer than the
    request deadline, so leaving this to the first request would time it out.
    """
    global _warmup_state
    from src.graph.nodes import _get_retriever

    try:
        retriever = _get_retriever(settings.auth_tenant_id, _roles())
        retriever.retrieve("warm up")
        _warmup_state = "ready"
        logger.info(json.dumps({"event": "warmup", "status": "ready"}))
    except Exception as exc:
        _warmup_state = "failed"
        logger.error(
            json.dumps({"event": "warmup", "status": "failed", "error": type(exc).__name__})
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail fast on unsafe auth configuration and drain the query executor."""
    global _warmup_state
    if settings.require_auth and not settings.api_key:
        raise RuntimeError("REQUIRE_AUTH=true requires API_KEY to be set.")
    if settings.warmup_on_startup and index_is_ready():
        # In the background, so the process answers health checks meanwhile.
        _warmup_state = "running"
        threading.Thread(target=_warm_up, name="rag-warmup", daemon=True).start()
    yield
    executor.shutdown(wait=False, cancel_futures=True)
    tracing.shutdown()


app = FastAPI(title="rag-system-frameworks", lifespan=lifespan)


@app.middleware("http")
async def add_charset(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _observe_request(request, 500, time.perf_counter() - started)
        raise
    _observe_request(request, response.status_code, time.perf_counter() - started)
    if response.headers.get("content-type") == "application/json":
        response.headers["content-type"] = "application/json; charset=utf-8"
    response.headers["x-request-id"] = request_id
    logger.info(
        json.dumps(
            {
                "event": "request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
    )
    return response


def _observe_request(request: Request, status_code: int, seconds: float) -> None:
    """Record HTTP metrics, labeled by route template to keep labels bounded."""
    route = getattr(request.scope.get("route"), "path", "unmatched")
    if route == "/metrics" or route.startswith("/health"):
        return
    metrics.REQUESTS.labels(method=request.method, route=route, status=str(status_code)).inc()
    metrics.REQUEST_SECONDS.labels(route=route).observe(seconds)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=settings.max_question_chars)


class QueryResponse(BaseModel):
    answer: str
    route: str
    grounded: bool
    citations: list[Citation]
    index_version: str
    # ok | partial (evidence looked weak) | extractive (no LLM) | refused (no evidence)
    answer_status: str
    # The degraded path that produced the answer, or null.
    fallback: str | None = None


def _roles() -> list[str]:
    """Return the fixed roles configured for the local API key."""
    return [role.strip() for role in settings.auth_roles.split(",") if role.strip()]


def _identity(authorization: str | None) -> tuple[str, list[str]]:
    """Authenticate the configured local API key and return fixed claims.

    This is a deliberately simple single-tenant mechanism, not an identity
    provider: every valid key maps to the same tenant and roles.
    """
    if not settings.require_auth:
        return settings.auth_tenant_id, _roles()
    expected = f"Bearer {settings.api_key}"
    if (
        not settings.api_key
        or not authorization
        or not hmac.compare_digest(authorization, expected)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid bearer credentials are required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return settings.auth_tenant_id, _roles()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/health/ready")
def readiness() -> dict:
    if not index_is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The persisted search index is not ready. Run ingestion first.",
        )
    if _warmup_state in {"running", "failed"}:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"The retriever warm-up is {_warmup_state}.",
        )
    return {"status": "ready", "index_version": active_index_version()}


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics(authorization: str | None = Header(default=None)) -> Response:
    """Expose metrics, behind the API key whenever authentication is required."""
    if not settings.metrics_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    _identity(authorization)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _log_query(request_id: str, result: dict, context: RequestContext) -> None:
    """Log one query's outcome. Never includes the question or any document text."""
    logger.info(
        json.dumps(
            {
                "event": "query",
                "request_id": request_id,
                "route": result.get("route"),
                "escalated": bool(result.get("escalated")),
                "answer_status": result.get("answer_status"),
                "fallback": result.get("fallback"),
                "grounded": result.get("grounded"),
                "citation_count": len(result.get("citations") or []),
                "retrieval_count": len(result.get("retrieved_chunks") or []),
                "llm_calls": context.llm_calls,
                "input_tokens": context.input_tokens,
                "output_tokens": context.output_tokens,
                "models": context.models_used,
            }
        )
    )


@app.post("/query", response_model=QueryResponse)
async def query(
    request: Request,
    req: QueryRequest,
    authorization: str | None = Header(default=None),
) -> QueryResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question is required.",
        )
    # Authenticate before taking the single query slot so unauthenticated
    # traffic cannot make the service look busy.
    tenant_id, roles = _identity(authorization)

    if _warmup_state == "running":
        metrics.REJECTED.labels(reason="warming_up").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The service is warming up; retry shortly.",
            headers={"Retry-After": "10"},
        )

    if not query_slots.acquire(blocking=False):
        metrics.REJECTED.labels(reason="busy").inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="The local RAG service is busy; retry shortly.",
            headers={"Retry-After": "5"},
        )

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    timeout = settings.request_timeout_seconds
    context = RequestContext(
        deadline=time.monotonic() + timeout,
        max_llm_calls=settings.llm_max_calls_per_query,
    )

    def _invoke(state: dict) -> dict:
        try:
            with request_scope(context), tracing.query_trace(
                request_id, tenant_id, state["question"]
            ) as trace:
                result = graph.invoke(state)
                trace.finish(
                    {
                        "route": result.get("route"),
                        "answer_status": result.get("answer_status"),
                        # Langfuse drops None values, so a normal answer needs a word.
                        "fallback": result.get("fallback") or "none",
                        "llm_calls": context.llm_calls,
                        "index_version": active_index_version(),
                    },
                    {"grounded": float(bool(result.get("grounded")))},
                    output=result.get("answer"),
                )
                return result
        finally:
            # Released by the worker, so a timed-out request keeps its slot
            # until the graph really stops using the CPU and the provider.
            query_slots.release()

    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(
            executor,
            _invoke,
            {
                "question": question,
                "retry_count": 0,
                "tenant_id": tenant_id,
                "roles": roles,
            },
        )
    except Exception:
        # The worker never started, so release the slot here.
        query_slots.release()
        raise
    # Retrieve the exception of a future abandoned after a client timeout so
    # asyncio does not log "exception was never retrieved".
    future.add_done_callback(
        lambda done: None if done.cancelled() else done.exception()
    )

    try:
        result = await asyncio.wait_for(
            asyncio.shield(future), timeout=timeout + DEADLINE_GRACE_SECONDS
        )
    except (TimeoutError, DeadlineExceeded) as exc:
        metrics.REJECTED.labels(reason="deadline").inc()
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The RAG request timed out.",
        ) from exc
    except LLMUnavailable as exc:
        metrics.REJECTED.labels(reason="llm_unavailable").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
            headers={"Retry-After": "30"},
        ) from exc
    except RuntimeError as exc:
        metrics.REJECTED.labels(reason="index_unavailable").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    _log_query(request_id, result, context)
    return QueryResponse(
        answer=result.get("answer", ""),
        route=result.get("route", ""),
        grounded=result.get("grounded", False),
        citations=result.get("citations", []),
        index_version=active_index_version(),
        answer_status=result.get("answer_status", "ok"),
        fallback=result.get("fallback"),
    )
