"""
Pydantic models defining the state schema, extracted entities,
and final structured output for the MDI ticket triage agent.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TicketCategory(str, Enum):
    BILLING = "Billing"
    CLAIMS_WORKFLOW = "Claims Workflow"
    USER_ERROR = "User Error"
    BUG = "Bug"
    ENHANCEMENT_REQUEST = "Enhancement Request"
    SYSTEM_INCOMPATIBILITY = "System Incompatibility"
    UNKNOWN = "Unknown"


class UrgencyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ExtractedEntities(BaseModel):
    client_name: Optional[str] = Field(None, description="Name of the client or organization")
    module_affected: Optional[str] = Field(None, description="Software module or feature affected")
    urgency: UrgencyLevel = Field(UrgencyLevel.MEDIUM, description="Assessed urgency level")
    additional_context: Optional[str] = Field(None, description="Any other relevant details extracted")


class TicketOutput(BaseModel):
    """Final structured output returned by the agent."""
    category: TicketCategory
    entities: ExtractedEntities
    is_complete: bool = Field(description="Whether the ticket had sufficient information to respond")
    suggested_response: Optional[str] = Field(None, description="Drafted response if ticket was complete")
    clarification_request: Optional[str] = Field(None, description="Follow-up questions if ticket was incomplete")
    needs_human_review: bool = Field(False, description="Whether a processing failure requires manual review")
    processing_errors: list[str] = Field(default_factory=list, description="Log of any issues encountered during processing")


class AgentState(BaseModel):
    """
    Shared state passed between LangGraph nodes.
    Each node reads what it needs and writes its outputs.
    """
    raw_ticket: str
    category: Optional[TicketCategory] = None
    entities: Optional[ExtractedEntities] = None
    is_complete: Optional[bool] = None
    suggested_response: Optional[str] = None
    clarification_request: Optional[str] = None
    needs_human_review: bool = False
    processing_errors: list[str] = Field(default_factory=list)
    output: Optional[TicketOutput] = None

    class Config:
        arbitrary_types_allowed = True
