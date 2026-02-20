"""
Graph state definition using TypedDict for LangGraph compatibility.
Pydantic models from models.py are used for structured data within the state.
"""

from typing import Optional, TypedDict
from models import TicketCategory, ExtractedEntities, TicketOutput


class GraphState(TypedDict):
    raw_ticket: str
    category: Optional[TicketCategory]
    entities: Optional[ExtractedEntities]
    is_complete: Optional[bool]
    suggested_response: Optional[str]
    clarification_request: Optional[str]
    needs_human_review: bool
    processing_errors: list[str]
    output: Optional[TicketOutput]
