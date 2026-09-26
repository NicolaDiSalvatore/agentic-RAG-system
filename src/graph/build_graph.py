"""
Wire the node functions into a compiled LangGraph graph.
"""

from langgraph.graph import END, StateGraph

from .state import GraphState
from .nodes import (
    MAX_RETRIES,
    decompose_node,
    generate_node,
    grade_node,
    no_retrieval_node,
    retrieve_node,
    route_node,
)


def build_graph():

    graph = StateGraph(GraphState)

    graph.add_node("route", route_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_retrieval", no_retrieval_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {
            "no_retrieval": "no_retrieval",
            "simple": "retrieve",
            "complex": "decompose"
        },
    )

    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        lambda state: (
            "generate"
            if state.get("context_sufficient")
            or state.get("retry_count", 0) >= MAX_RETRIES
            else "retrieve"
        ),
        {"generate": "generate", "retrieve": "retrieve"},
    )

    graph.add_edge("generate", END)
    graph.add_edge("no_retrieval", END)

    return graph.compile()
