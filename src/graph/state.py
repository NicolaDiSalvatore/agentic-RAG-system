"""
The shared state that flows through every node in the LangGraph graph.
"""

from typing import TypedDict


class GraphState(TypedDict):
    question: str
    route: str
    sub_questions: list[str]
    retrieved_chunks: list[dict]
    context_sufficient: bool
    retry_count: int
    answer: str
    grade_details: list[dict]
    tenant_id: str
    roles: list[str]
    citations: list[dict]
    grounded: bool
    evidence: list[dict]
    escalated: bool
    answer_status: str
    fallback: str | None
