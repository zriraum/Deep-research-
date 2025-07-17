
"""
Multi-Agent Research Supervisor System

This module implements a multi-agent research system where a supervisor coordinates
multiple research agents to conduct parallel research on different topics.

Architecture:
- Supervisor: Decides what research to conduct and when to complete
- Researcher Subgraph: Individual research agents that perform tool calls
- State Management: Tracks conversation history and research progress
"""

import asyncio
from datetime import datetime
from typing_extensions import Literal

import nest_asyncio
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, MessageLikeRepresentation, filter_messages, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research_from_scratch.utils import tavily_search
from deep_research_from_scratch.prompts import research_agent_prompt, compress_research_system_prompt
from deep_research_from_scratch.multi_agent_supervisor_state import SupervisorState, ResearcherState, ResearcherOutputState, ConductResearch, ResearchComplete

# ===== UTILITY FUNCTIONS =====

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]) -> list[str]:
    """Extract content from tool messages in the conversation history."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

# ===== CONFIGURATION =====

# Enable nested event loops for Jupyter
nest_asyncio.apply()

# Initialize researcher model and tools
researcher_tools = [tavily_search]
researcher_tools_by_name = {tool.name: tool for tool in researcher_tools}
researcher_model = init_chat_model(model="openai:gpt-4.1")
researcher_model_with_tools = researcher_model.bind_tools(researcher_tools)

# Initialize supervisor model and tools
supervisor_tools = [ConductResearch, ResearchComplete]
supervisor_model = init_chat_model(model="openai:gpt-4.1")
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

# System constants
max_researcher_iterations = 3

# ===== RESEARCHER NODES =====

def researcher_llm_call(state: ResearcherState) -> Command[Literal["researcher_tool_node"]]:
    """
    Researcher LLM decision node.

    The researcher decides whether to call tools based on the current conversation state.
    Always proceeds to tool execution to maintain the research flow.
    """
    result = researcher_model_with_tools.invoke(
        [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
    )

    return Command(
        goto="researcher_tool_node",
        update={
            "researcher_messages": [result],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

def researcher_tool_node(state: ResearcherState) -> Command[Literal["compress_research", "researcher_llm_call"]]:
    """
    Researcher tool execution node.

    Executes tool calls concurrently and decides whether to continue research
    or proceed to compression based on iteration limits.
    """
    tool_calls = state["researcher_messages"][-1].tool_calls

    async def execute_tools():
        """Execute all tool calls concurrently for better performance."""
        coros = []
        for tool_call in tool_calls:
            tool = researcher_tools_by_name[tool_call["name"]]
            coros.append(tool.ainvoke(tool_call["args"]))

        # Execute all tool calls concurrently
        observations = await asyncio.gather(*coros)

        # Create tool message outputs
        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ) for observation, tool_call in zip(observations, tool_calls)
        ]

        return tool_outputs

    # Run async function in sync context with nested event loop support
    tool_outputs = asyncio.run(execute_tools())

    # Determine next step: compress if max iterations reached or continue research
    should_compress = (
        state.get("tool_call_iterations", 0) >= max_researcher_iterations or 
        any(tool_call["name"] == "ResearchComplete" for tool_call in tool_calls)
    )

    if should_compress:
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    return Command(
        goto="researcher_llm_call",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState):
    """
    Research compression node.

    Summarizes and compresses the research findings into a concise report
    while preserving all raw notes for reference.
    """
    response = await researcher_model.ainvoke([
        SystemMessage(content=compress_research_system_prompt.format(date=get_today_str())),
        *state.get("researcher_messages", [])
    ])

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join([
            str(m.content) for m in filter_messages(
                state["researcher_messages"], 
                include_types=["tool", "ai"]
            )
        ])]
    }

# ===== SUPERVISOR NODES =====

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """
    Supervisor decision node.

    The supervisor coordinates research activities by deciding what research
    to conduct and when the research process is complete.
    """
    supervisor_messages = state.get("supervisor_messages", [])

    # Make decision about next research steps
    response = await supervisor_model_with_tools.ainvoke(supervisor_messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """
    Supervisor tool execution node.

    Executes supervisor decisions by either launching research sub-agents
    or terminating the research process based on completion criteria.
    """
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # Check exit criteria
    exceeded_allowed_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

    # Launch parallel research sub-agents
    try:
        conduct_research_calls = [
            tool_call for tool_call in most_recent_message.tool_calls 
            if tool_call["name"] == "ConductResearch"
        ]

        # Create concurrent research tasks
        coros = [
            researcher_subgraph.ainvoke({
                "researcher_messages": [
                    HumanMessage(content=tool_call["args"]["research_topic"])
                ],
                "research_topic": tool_call["args"]["research_topic"]
            }) 
            for tool_call in conduct_research_calls
        ]

        # Wait for all research to complete
        tool_results = await asyncio.gather(*coros)

        # Format results as tool messages for supervisor
        tool_messages = [
            ToolMessage(
                content=observation.get("compressed_research", "Error synthesizing research report"),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ) for observation, tool_call in zip(tool_results, conduct_research_calls)
        ]

        return Command(
            goto="supervisor",
            update={
                "supervisor_messages": tool_messages,
                "raw_notes": ["\n".join([
                    "\n".join(observation.get("raw_notes", [])) 
                    for observation in tool_results
                ])]
            }
        )

    except Exception as e:
        print(f"Error in supervisor tools: {e}")
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

# ===== GRAPH CONSTRUCTION =====

# Build researcher subgraph
researcher_subgraph = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add researcher nodes
researcher_subgraph.add_node("researcher_llm_call", researcher_llm_call)
researcher_subgraph.add_node("researcher_tool_node", researcher_tool_node)
researcher_subgraph.add_node("compress_research", compress_research)

# Add researcher edges
researcher_subgraph.add_edge(START, "researcher_llm_call")
researcher_subgraph.add_edge("compress_research", END)

# Compile researcher subgraph
researcher_subgraph = researcher_subgraph.compile()

# Build supervisor subgraph
supervisor_builder = StateGraph(SupervisorState)

# Add supervisor nodes
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)

# Add supervisor edges
supervisor_builder.add_edge(START, "supervisor")

# Compile main agent
agent = supervisor_builder.compile()
