
"""
State Definitions and Pydantic Schemas

This module defines the state objects and structured schemas used throughout
the research workflow for maintaining conversation context and structured output.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

# ===== STATE DEFINITIONS =====

class AgentState(MessagesState):
    """
    Main state object for research workflow.

    Extends MessagesState to include conversation history plus
    the generated research brief for guiding research activities.
    """
    research_brief: Optional[str]

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decisions during scoping phase."""
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Schema for research brief generation."""
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class Summary(BaseModel):
    """Schema for webpage content summarization."""
    summary: str
    key_excerpts: str
