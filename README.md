# 🧱 Deep Research From Scratch 

Deep research has broken out as one the most popular agent applications. [OpenAI](https://openai.com/index/introducing-deep-research/), [Anthropic](https://www.anthropic.com/engineering/built-multi-agent-research-system), [Perplexity](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research), and [Google](https://gemini.google/overview/deep-research/?hl=en) all have deep research products that can search the web or [work context](https://www.anthropic.com/news/research) to perform research on user-defined topics. There are also many [open](https://huggingface.co/blog/open-deep-research) [source](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) implementations, including [ours](https://github.com/langchain-ai/open_deep_research). We built an open deep researcher that is simple and configurable, working across many model providers, search tools, and MCP servers. In this repo, we'll build a deep researcher from scratch! Here is a map of the major pieces that we will build:

![overview](https://github.com/user-attachments/assets/b71727bd-0094-40c4-af5e-87cdb02123b4)

## 🚀 Quickstart 

### Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository and activate a virtual environment:
```bash
git clone https://github.com/langchain-ai/deep_research_from_scratch
cd deep_research_from_scratch
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install the package and dependencies:
```bash
uv pip install -e .
```

## Background 

Research is an open‑ended task; the best strategy for a user request can’t be easily known in advance. Requests can require a wide breadth of research strategies, from A / B comparisons (e.g., compare these two products) to open-ended search followed by filtering (e.g., find me the top 20 candidates for this role). Requests also require variable depth of research. Some require one-shot search and summarization whereas others require many turns of search and reasoning. With the above challenges in mind, our deep researcher design uses two guiding principles:

1. **Flexibility to handle a breadth of research strategies** 
2. **Flexibility to handle variable research depth**

## Architecture 

The agent architecture is designed with our principles in mind. It has three stages:

- **Scope** – clarify research scope
- **Research** – perform research
- **Write** – produce the final report

## 📝 Organization 

This repo contains 5 tutorial notebooks that build a deep research system from scratch:

### 📚 Tutorial Notebooks

#### 1. User Clarification and Brief Generation (`notebooks/1_scoping.ipynb`)
**Purpose**: Clarify research scope and transform user input into structured research briefs

**Key Concepts**:
- **User Clarification**: Determines if additional context is needed from the user using structured output
- **Brief Generation**: Transforms conversations into detailed research questions
- **Context Engineering**: Managing state effectively through structured schemas
- **LangGraph Commands**: Using Command system for flow control and state updates

**Implementation Highlights**:
- Pydantic schemas for structured decision making (`ClarifyWithUser`, `ResearchQuestion`)
- Binary decision logic to minimize hallucination and improve reliability
- Conditional routing based on clarification needs
- Prompt engineering for context gathering vs. efficiency balance

**Learning Exercises**: Modify prompts to adjust clarification behavior, extend schemas for richer context capture, test boundary cases

---

#### 2. Research Agent with Search Tools (`notebooks/2_research_agent.ipynb`)
**Purpose**: Build research agents using both native and external search tools

**Key Concepts**:
- **Native Search Tools**: Using provider-integrated search (Anthropic Web Search API, OpenAI Web Search)
- **External Tool Integration**: ReAct pattern with tools like Tavily Search
- **Runtime Configuration**: LangGraph runtime config for dynamic search provider selection
- **Agent Loop Patterns**: Iterative tool calling and reasoning

**Implementation Highlights**:
- Native search simplifies architecture and improves performance
- Traditional ReAct loops for external tools with explicit tool execution
- Configurable search providers through runtime configuration
- System prompts optimized for thorough research with source citations

**Benefits Comparison**: Native search offers lower latency and cost vs. external tools providing transparency and control

---

#### 3. Research Agent with MCP (`notebooks/3_research_agent_mcp.ipynb`)
**Purpose**: Integrate Model Context Protocol (MCP) servers as research tools

**Key Concepts**:
- **Model Context Protocol**: Standardized protocol for AI model-external resource interaction
- **MCP Architecture**: Hosts, clients, servers, resources, tools, and prompts
- **LangChain MCP Adapters**: Seamless integration of MCP servers as LangChain tools
- **Local vs Remote MCP**: Deployment pattern considerations

**Implementation Highlights**:
- `MultiServerMCPClient` for managing multiple MCP servers
- Configuration-driven MCP server setup (stdio and HTTP transports)
- Tool filtering for research-relevant capabilities
- Hybrid deployments combining local and remote MCP servers

**Use Cases**: File system access, database operations, enterprise knowledge bases, specialized APIs

---

#### 4. Research Supervisor (`notebooks/4_research_supervisor.ipynb`)
**Purpose**: Multi-agent coordination for complex research tasks

**Key Concepts**:
- **Multi-Agent Architecture**: Specialization, parallel processing, quality control, scalability
- **Supervisor Pattern**: Decision-making node + execution node architecture
- **Parallel Research**: Concurrent research agents for independent topics
- **Research Coordination**: Iterative research with exit conditions

**Implementation Highlights**:
- Two-node supervisor pattern (`supervisor` + `supervisor_tools`)
- Structured tools for research delegation (`ConductResearch`, `ResearchComplete`)
- Configurable concurrency limits and iteration controls
- Native search integration for efficient research agents

**Design Principles**: Clear role definition, structured communication, hierarchical coordination, fault tolerance

---

#### 5. Full Multi-Agent Research System (`notebooks/5_full_agent.ipynb`)
**Purpose**: Complete integration of scoping, research supervision, and agent coordination

**Key Concepts**:
- **Three-Phase Architecture**: Scope → Research → Write
- **End-to-End Workflow**: From user input to comprehensive research reports
- **System Integration**: Combining all previous components into unified system
- **Configuration Management**: Centralized configuration for all system components

**Implementation Highlights**:
- Complete state flow from `AgentInputState` through `AgentState`
- Integrated scoping and research phases with proper state transitions
- Comprehensive configuration schema supporting all system parameters
- Interactive clarification handling and research execution

**System Flow**: Clarification → Brief Generation → Multi-Agent Research with supervisor coordination

---

#### 6. Evals

---

#### 7. Deployment

---

### 🎯 Key Learning Outcomes

- **Structured Output**: Using Pydantic schemas for reliable AI decision making
- **Workflow Design**: LangGraph patterns for complex multi-step processes
- **Search Integration**: Native vs. external search tools trade-offs
- **Multi-Agent Patterns**: Coordination, specialization, and parallel processing
- **Configuration Management**: Runtime configuration for flexible system behavior
- **Error Handling**: Graceful failure handling and system resilience

Each notebook builds on the previous concepts, culminating in a production-ready deep research system that can handle complex, multi-faceted research queries with intelligent scoping and coordinated execution. 
