
import operator
from typing import Optional, Annotated, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langchain_core.messages import MessageLikeRepresentation

def override_reducer(current_value, new_value):
    """
    Custom state reducer for flexible state updates.

    Allows either additive updates (default) or complete override of state values.
    When new_value is a dict with {"type": "override", "value": data}, 
    the existing state is replaced with the new value.
    Otherwise, new values are added to existing state using operator.add.
    """
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)

class AgentInputState(MessagesState):
    """Input state for the full agent - only contains messages from user input."""
    pass

class AgentState(MessagesState):
    """
    Main state for the full multi-agent research system.

    Extends MessagesState with additional fields for research coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and the main workflow.
    """
    supervisor_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    research_brief: Optional[str]

    # Research output fields - these are duplicated in AgentState to capture
    # the final outputs from the supervisor subgraph for report generation
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

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

class SupervisorState(TypedDict):
    """State for supervisor subgraph that coordinates research agents."""
    supervisor_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherState(TypedDict):
    """State for individual researcher agents."""
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ResearcherOutputState(BaseModel):
    """Output schema for researcher subgraph."""
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []

class ConductResearch(BaseModel):
    """Tool schema for supervisor to delegate research tasks."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Tool schema for supervisor to indicate research completion."""
    pass
