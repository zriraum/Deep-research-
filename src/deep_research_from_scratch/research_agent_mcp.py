
"""
Research Agent with MCP Integration

This module implements a research agent that integrates with Model Context Protocol (MCP)
servers to access tools and resources. The agent demonstrates how to use MCP filesystem
server for local document research and analysis.

Key features:
- MCP server integration for tool access
- Async operations for concurrent tool execution (required by MCP protocol)
- Filesystem operations for local document research
- Secure directory access with permission checking
"""

import os
import asyncio
from typing_extensions import Literal

# Import nest_asyncio only when needed (in Jupyter environments)
try:
    import nest_asyncio
    # Only apply if running in Jupyter/IPython environment
    try:
        # TODO: We may need to import this?
        get_ipython()
        nest_asyncio.apply()
    except NameError:
        pass  # Not in Jupyter, no need for nest_asyncio
except ImportError:
    pass  # nest_asyncio not available, proceed without it

from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from deep_research_from_scratch.prompts import research_agent_prompt_with_mcp

# ===== CONFIGURATION =====

# Nested event loops are automatically handled above if in Jupyter environment

# MCP server configuration for filesystem access
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",  # Auto-install if needed
            "@modelcontextprotocol/server-filesystem",
            os.path.abspath("./files/")  # Path to research documents
        ],
        "transport": "stdio"  # Communication via stdin/stdout
    }
}

# Initialize MCP client
client = MultiServerMCPClient(mcp_config)

# ===== AGENT NODES =====

async def llm_call(state: MessagesState):
    """
    LLM decision node with MCP tool integration.

    This node:
    1. Retrieves available tools from MCP server
    2. Binds tools to the language model
    3. Processes user input and decides on tool usage

    Returns updated state with model response.
    """
    # Get available tools from MCP server
    mcp_tools = await client.get_tools()

    # Use MCP tools for local document access
    tools = mcp_tools

    # Initialize model with tool binding
    model = init_chat_model(model="anthropic:claude-sonnet-4-20250514")
    model_with_tools = model.bind_tools(tools)

    # Process user input with system prompt
    return {
        "messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt_with_mcp)] + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    """
    Tool execution node for MCP tools.

    This node:
    1. Retrieves current tool calls from the last message
    2. Executes all tool calls using async operations (required for MCP)
    3. Returns formatted tool results

    Note: MCP requires async operations due to inter-process communication
    with the MCP server subprocess. This is unavoidable.
    """
    tool_calls = state["messages"][-1].tool_calls

    async def execute_tools():
        """Execute all tool calls. MCP tools require async execution."""
        # Get fresh tool references from MCP server
        mcp_tools = await client.get_tools()
        tools_by_name = {tool.name: tool for tool in mcp_tools}

        # Execute tool calls (sequentially for reliability)
        observations = []
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            observation = await tool.ainvoke(tool_call["args"])
            observations.append(observation)

        # Format results as tool messages
        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ) for observation, tool_call in zip(observations, tool_calls)
        ]

        return tool_outputs

    # Handle async execution in different contexts
    try:
        # Try to get current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In Jupyter or other async context, create task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, execute_tools())
                messages = future.result()
        else:
            # Standard event loop, can use asyncio.run
            messages = asyncio.run(execute_tools())
    except RuntimeError:
        # No event loop, create new one
        messages = asyncio.run(execute_tools())

    return {"messages": messages}

# ===== ROUTING LOGIC =====

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """
    Conditional routing function.

    Determines whether to continue with tool execution or provide final answer
    based on whether the LLM made tool calls.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Continue to tool execution if tools were called
    if last_message.tool_calls:
        return "tool_node"
    # Otherwise, provide final answer
    return "__end__"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder_mcp = StateGraph(MessagesState)

# Add nodes to the graph
agent_builder_mcp.add_node("llm_call", llm_call)
agent_builder_mcp.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder_mcp.add_edge(START, "llm_call")
agent_builder_mcp.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",  # Continue to tool execution
        "__end__": END,            # Provide final answer
    },
)
agent_builder_mcp.add_edge("tool_node", "llm_call")  # Loop back for more processing

# Compile the agent
agent_mcp = agent_builder_mcp.compile()
