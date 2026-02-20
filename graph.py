"""
LangGraph workflow definition for the MDI ticket triage agent.

Graph topology:
    [Classifier] → [Entity Extractor] → [Completeness Check]
                                                ↓
                                       ┌───────┴────────┐
                                       ↓                ↓
                                  [Complete]       [Incomplete]
                                       ↓                ↓
                              [Response Drafter] [Clarification Drafter]
                                       ↓                ↓
                                  [Output Assembler] ←───┘
"""

from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import (
    classify_ticket,
    extract_entities,
    check_completeness,
    draft_response,
    draft_clarification,
    assemble_output,
)


def route_on_completeness(state: dict) -> str:
    """Conditional edge: route to response drafter or clarification drafter."""
    if state.get("is_complete"):
        return "draft_response"
    return "draft_clarification"


def build_graph() -> StateGraph:
    """Construct and compile the triage agent graph."""

    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("classify", classify_ticket)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("check_completeness", check_completeness)
    graph.add_node("draft_response", draft_response)
    graph.add_node("draft_clarification", draft_clarification)
    graph.add_node("assemble_output", assemble_output)

    # Linear edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "extract_entities")
    graph.add_edge("extract_entities", "check_completeness")

    # Conditional branch
    graph.add_conditional_edges(
        "check_completeness",
        route_on_completeness,
        {
            "draft_response": "draft_response",
            "draft_clarification": "draft_clarification",
        },
    )

    # Both branches converge to output assembly
    graph.add_edge("draft_response", "assemble_output")
    graph.add_edge("draft_clarification", "assemble_output")
    graph.add_edge("assemble_output", END)

    return graph.compile()


# Singleton compiled graph
triage_agent = build_graph()
