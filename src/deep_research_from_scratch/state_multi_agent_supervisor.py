
"""
State Definitions for Multi-Agent Research Supervisor

This module defines the state objects and tools used for the multi-agent
research supervisor workflow, including coordination state and research tools.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import MessageLikeRepresentation
from langchain_core.tools import tool
from pydantic import BaseModel, Field


def override_reducer(current_value, new_value):
    """
    Reducer function that allows complete override of list values.

    If new_value is a dict with type "override", replaces the current value entirely.
    Otherwise, uses the default operator.add behavior to append values.

    Args:
        current_value: Existing list value
        new_value: New value to add or override with

    Returns:
        Updated list value
    """
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)


class SupervisorState(TypedDict):
    """
    State for the multi-agent research supervisor.

    Manages coordination between supervisor and research agents, tracking
    research progress and accumulating findings from multiple sub-agents.
    """
    # Messages exchanged with supervisor for coordination and decision-making
    supervisor_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    # Detailed research brief that guides the overall research direction
    research_brief: str
    # Processed and structured notes ready for final report generation
    notes: Annotated[list[str], override_reducer] = []
    # Counter tracking the number of research iterations performed
    research_iterations: int = 0
    # Raw unprocessed research notes collected from sub-agent research
    raw_notes: Annotated[list[str], override_reducer] = []


@tool
class ConductResearch(BaseModel):
    """Tool for delegating research tasks to specialized sub-agents."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


@tool
class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""
    pass
