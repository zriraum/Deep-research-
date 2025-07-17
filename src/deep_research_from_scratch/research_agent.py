
"""
Research Agent Implementation

This module implements a research agent that can perform iterative web searches
and synthesis to answer complex research questions. The agent uses async operations
for concurrent search execution and includes tools for web search and content processing.
"""

import asyncio
import nest_asyncio
from typing_extensions import Literal

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain.chat_models import init_chat_model
from IPython.display import Image, display

from deep_research_from_scratch.utils import tavily_search
from deep_research_from_scratch.prompts import research_agent_prompt

# ===== CONFIGURATION =====

# Enable nested event loops for Jupyter notebook compatibility
nest_asyncio.apply()

# Set up tools and model binding
tools = [tavily_search]
tools_by_name = {tool.name: tool for tool in tools}

# Initialize model with tool binding
model = init_chat_model(model="openai:gpt-4.1")
model_with_tools = model.bind_tools(tools)

# ===== AGENT NODES =====

def llm_call(state: MessagesState):
    """
    LLM decision node.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt)] + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    """
    Tool execution node.

    Executes all tool calls from the previous LLM response concurrently.
    This is where the async benefits become apparent - multiple search queries
    can be executed simultaneously rather than sequentially.

    Returns updated state with tool execution results.
    """
    tool_calls = state["messages"][-1].tool_calls

    async def execute_tools():
        """Execute all tool calls concurrently for better performance."""
        # Create coroutines for all tool calls
        coros = []
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
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
    messages = asyncio.run(execute_tools())
    return {"messages": messages}

# ===== ROUTING LOGIC =====

def should_continue(state: MessagesState) -> Literal["environment", "__end__"]:
    """
    Conditional routing function.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "environment": Continue to tool execution
        "__end__": Stop and return final answer
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        return "environment"
    # Otherwise, we have a final answer
    return "__end__"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder = StateGraph(MessagesState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "environment": "environment",  # Continue research loop
        "__end__": END,               # Provide final answer
    },
)
agent_builder.add_edge("environment", "llm_call")  # Loop back for more research

# Compile the agent
agent = agent_builder.compile()
