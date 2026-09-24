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
