"""Prometheus metrics. Labels stay bounded: never a question, user or request ID."""

from prometheus_client import Counter, Gauge, Histogram

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60)

REQUESTS = Counter(
    "rag_requests_total", "HTTP requests.", ["method", "route", "status"]
)
REQUEST_SECONDS = Histogram(
    "rag_request_duration_seconds", "HTTP latency.", ["route"], buckets=_LATENCY_BUCKETS
)
REJECTED = Counter("rag_rejected_total", "Queries rejected or abandoned.", ["reason"])
NODE_SECONDS = Histogram(
    "rag_node_duration_seconds", "Graph node latency.", ["node"], buckets=_LATENCY_BUCKETS
)
ROUTES = Counter("rag_routes_total", "Initial route decisions.", ["route"])
ESCALATIONS = Counter(
    "rag_escalations_total", "Simple queries escalated after weak evidence."
)
TOP_DENSE_SCORE = Histogram(
    "rag_retrieval_top_dense_score",
    "Best cosine similarity per retrieved query.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
LLM_CALLS = Counter(
    "rag_llm_calls_total", "Provider attempts.", ["provider", "model", "role", "status"]
)
LLM_SECONDS = Histogram(
    "rag_llm_call_duration_seconds", "Provider latency.", ["provider", "model"],
    buckets=_LATENCY_BUCKETS,
)
LLM_TOKENS = Counter(
    "rag_llm_tokens_total", "Provider tokens.", ["provider", "model", "direction"]
)
FALLBACKS = Counter(
    "rag_fallbacks_total", "Queries served by a degraded path.", ["kind"]
)
BREAKER_STATE = Gauge(
    "rag_circuit_breaker_open", "1 when a target's breaker is not closed.", ["provider", "model"]
)
ANSWER_STATUS = Counter("rag_answer_status_total", "Answer outcomes.", ["status"])
GROUNDED = Counter("rag_grounded_total", "Grounded answer outcomes.", ["grounded"])
CITATION_MARKERS = Counter(
    "rag_citation_markers_total",
    "Citation markers emitted by the model.",
    ["result"],
)
UNSUPPORTED_CLAIMS = Counter(
    "rag_citation_unsupported_sentences_total",
    "Cited sentences sharing too little vocabulary with their cited chunks.",
)
QUOTA_USED = Gauge("rag_daily_quota_used", "Today's LLM spend.", ["unit"])
QUOTA_EXHAUSTED = Counter(
    "rag_daily_quota_exhausted_total", "LLM calls refused by the daily budget."
)
