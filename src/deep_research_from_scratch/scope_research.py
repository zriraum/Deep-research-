
from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,  AIMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.graph import MessagesState

from deep_research_from_scratch.prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from deep_research_from_scratch.state import AgentState, ClarifyWithUser, ResearchQuestion

from typing import Literal

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

# Initialize model
model = init_chat_model(model="openai:gpt-4.1")

# TODO: Resolve "Task clarify_with_user with path ('__pregel_pull', 'clarify_with_user')" wrote to unknown channel branch:to:__end__, ignoring it
def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", "__end__"]]:

    # Structured output 
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages = state["messages"]), 
            date=get_today_str()
        ))
    ])

    # Route based on clarification need
    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
    else:
        return Command(goto="write_research_brief", update={"messages": [AIMessage(content=response.verification)]})

def write_research_brief(state: AgentState):

    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # Generate research brief
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # Update state and route to end
    return {
            "research_brief": response.research_brief,
        }

# Build the workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=MessagesState)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("write_research_brief", END)

scope_research = deep_researcher_builder.compile()
