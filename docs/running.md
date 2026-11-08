# Running on a laptop

Everything here runs on an 8 GB laptop at no cost, with no server and no Docker.
Groq's free tier generates the answers; retrieval stays on your machine.

Only one process can open the local index at a time, so stop the API before running
`scripts.ingest`, `scripts.ask` or `scripts.calibrate_evidence`.

## Start the API

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Wait until `/health/ready` returns 200, then ask a question in a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/ready
$body = @{ question = "Where did Phileas Fogg live?" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/query -Method Post -ContentType "application/json" -Body $body
```

## Monitoring

There are four levels. Start with the first two, which need nothing installed.

### 1. Logs

The API terminal prints one JSON line per query, with its route, status, LLM calls,
tokens and models. Questions and passages are never logged.

### 2. Metrics page

Open <http://127.0.0.1:8000/metrics>, or filter it in PowerShell:

```powershell
(Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/metrics).Content -split "`n" | Select-String "^rag_(answer_status|grounded|llm_calls|daily_quota|fallbacks)"
```

| Metric | What it tells you |
|---|---|
| `rag_answer_status_total` | Answers that were ok, partial, extractive or refused |
| `rag_grounded_total` | Answers whose citations passed every check |
| `rag_llm_calls_total` | Provider attempts, by model and result |
| `rag_llm_tokens_total` | Tokens used |
| `rag_daily_quota_used` | Today's spend against the daily budget |
| `rag_escalations_total` | Simple questions that needed decomposing |
| `rag_circuit_breaker_open` | 1 when a model is being skipped |
| `rag_request_duration_seconds` | Latency |

Counters reset when the API restarts, except the daily quota.

### 3. Langfuse: see each query (free, cloud)

This shows each query as a trace, with its LLM calls, tokens and a `grounded` score.

1. Create a free project at <https://cloud.langfuse.com> and copy its API keys.
2. Add them to `.env`, then restart the API:

   ```dotenv
   LANGFUSE_ENABLED=true
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   ```

3. Ask a question, then look under Tracing in Langfuse.

The `query` entry shows the question and final answer next to their lengths, so they
are sent to Langfuse's servers. Set `LANGFUSE_CAPTURE_QUERY_TEXT=false` before sharing
a demo, so visitors' questions stay on your machine.

The LLM entries show text lengths rather than prompts and passages. To see that text
when debugging, set `LANGFUSE_CAPTURE_CONTENT=true`.

### 4. Prometheus: charts over time (optional)

This is only worth running if you want history, and it uses extra memory. Download
the Windows `.zip` from <https://prometheus.io/download/>, then run this from the
repository root:

```powershell
C:\path\to\prometheus.exe --config.file=deploy\prometheus\prometheus.yml --storage.tsdb.path=.cache\prometheus --storage.tsdb.retention.time=7d
```

Open <http://localhost:9090>. Try `rag_answer_status_total` on the Graph page, and see
rules on the Alerts page. The alert rules only display there; they don't notify you.

## Sharing a demo (optional)

The API only listens on your laptop. To let someone else try it for free, use a
Cloudflare Quick Tunnel while your laptop is on:

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

Before doing that, set `REQUIRE_AUTH=true` and a strong `API_KEY` in `.env`, or anyone
with the link can spend your Groq quota. Stop the tunnel when you finish.
