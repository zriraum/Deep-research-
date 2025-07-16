
import os
import asyncio
import nest_asyncio
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from typing_extensions import Literal
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.chat_models import init_chat_model
from IPython.display import Image, display

# Enable nested event loops for Jupyter
nest_asyncio.apply()

# Import our custom tool and utilities
from deep_research_from_scratch.utils import tavily_search
from deep_research_from_scratch.prompts import research_agent_prompt_with_mcp

# Using our custom @tool decorator with raw Tavily API and MCP server
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",  # Auto-install if needed
            "@modelcontextprotocol/server-filesystem",
             os.path.abspath("./files/")
        ],
        "transport": "stdio"
    }
}

# Create the MCP client
client = MultiServerMCPClient(mcp_config)

# Nodes
async def llm_call(state: MessagesState):

    # Combine the MCP tools with our custom tool
    mcp_tools = await client.get_tools()
    # tools = [tavily_search] + mcp_tools
    tools = mcp_tools
    tools_by_name = {tool.name: tool for tool in tools}

    # Initialize model
    model = init_chat_model(model="anthropic:claude-sonnet-4-20250514")
    model_with_tools = model.bind_tools(tools)

    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(content=research_agent_prompt_with_mcp)
                ]
                + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    """Performs the tool call - handles async tools concurrently in Jupyter"""
    tool_calls = state["messages"][-1].tool_calls

    async def execute_tools():

        # Combine the MCP tools with our custom tool
        mcp_tools = await client.get_tools()
        # tools = [tavily_search] + mcp_tools
        tools = mcp_tools
        tools_by_name = {tool.name: tool for tool in tools}

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

# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["environment", "__end__"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "environment"
    # Otherwise, we stop (reply to the user)
    return "__end__"

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        # Name returned by should_continue : Name of next node to visit
        "environment": "environment",
        "__end__": END,
    },
)
agent_builder.add_edge("environment", "llm_call")

# Compile the agent
agent = agent_builder.compile()
