"""
Wire the node functions into a compiled LangGraph graph.
"""

import time
from collections.abc import Callable

from langgraph.graph import END, StateGraph

from src.observability import metrics
from src.resilience import ensure_deadline

from .state import GraphState
from .nodes import (
    MAX_RETRIES,
    decompose_node,
    escalate_node,
    generate_node,
    grade_node,
    retrieve_node,
    route_node,
)


def _node(name: str, function: Callable, *, enforce_deadline: bool = True) -> Callable:
    """Wrap a node so it stops at the deadline and reports its latency."""

    def wrapped(state: GraphState) -> dict:
        if enforce_deadline:
            ensure_deadline()
        started = time.monotonic()
        try:
            return function(state)
        finally:
            metrics.NODE_SECONDS.labels(node=name).observe(time.monotonic() - started)

    return wrapped


def _after_grade(state: GraphState) -> str:
    """Choose the next step from how well retrieval covered the question.

    Weak evidence on a ``simple`` question most often means it had several parts,
    so it is decomposed rather than repeated. After that, the budget widens
    retrieval until ``MAX_RETRIES`` and the answer states any remaining gaps.
    """
    if state.get("context_sufficient"):
        return "generate"
    if state.get("route") == "simple" and not state.get("escalated"):
        return "escalate"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "retrieve"
    return "generate"


def build_graph():

    graph = StateGraph(GraphState)

    graph.add_node("route", _node("route", route_node))
    graph.add_node("decompose", _node("decompose", decompose_node))
    graph.add_node("escalate", _node("escalate", escalate_node))
    graph.add_node("retrieve", _node("retrieve", retrieve_node))
    graph.add_node("grade", _node("grade", grade_node))
    # Generation handles a spent deadline itself, because it can still answer
    # from the chunks already retrieved.
    graph.add_node("generate", _node("generate", generate_node, enforce_deadline=False))

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"simple": "retrieve", "complex": "decompose"},
    )

    graph.add_edge("decompose", "retrieve")
    graph.add_edge("escalate", "retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        _after_grade,
        {"generate": "generate", "escalate": "escalate", "retrieve": "retrieve"},
    )

    graph.add_edge("generate", END)

    return graph.compile()
