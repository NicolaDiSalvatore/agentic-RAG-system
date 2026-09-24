from fastapi import FastAPI
from pydantic import BaseModel
from src.graph.build_graph import build_graph

app = FastAPI(title="rag-system-frameworks")
graph = build_graph()


@app.middleware("http")
async def add_charset(request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type") == "application/json":
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    route: str
    sources: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = graph.invoke({"question": req.question, "retry_count": 0})
    return QueryResponse(
        answer=result.get("answer", ""),
        route=result.get("route", ""),
        sources=result.get("retrieved_chunks", []),
    )
