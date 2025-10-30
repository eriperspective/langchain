# Multi-Agent Systems

**LangChain Version:** v1.0+  
**Documentation:** https://docs.langchain.com/oss/python/langchain/multi-agent  
**Last Updated:** October 23, 2025

Break complex applications into specialized agents that work together.

---

## What is a Multi-Agent System?

**Multi-agent systems** use multiple specialized agents instead of one agent handling everything.

**When to use:**
- Single agent has too many tools and makes poor decisions
- Context or memory grows too large for one agent
- Tasks require specialization (planner, researcher, analyst)

**Key benefit:** Focused agents with clear responsibilities perform better than one generalist.

---

## Two Core Patterns

LangChain supports two multi-agent patterns:

| Pattern          | How it works                                  | Control flow  | Use case                    |
|------------------|-----------------------------------------------|---------------|-----------------------------|
| **Tool Calling** | Supervisor calls agents as tools              | Centralized   | Task orchestration          |
| **Handoffs**     | Agents transfer control to each other         | Decentralized | Multi-domain conversations  |

### Tool Calling Pattern

A **supervisor agent** calls other agents as tools. Sub-agents don't talk to users directly.

```
User → Supervisor → Sub-Agent (as tool) → Supervisor → Answer
```

**Flow:**
1. Supervisor receives input and decides which sub-agent to call
2. Sub-agent runs its task
3. Sub-agent returns results to supervisor
4. Supervisor decides next step or finishes

**Use when:**
- Need centralized workflow control
- Sub-agents perform specific tasks and return results
- Building structured workflows

### Handoffs Pattern

Agents directly transfer control to each other. The active agent changes.

```
User → Agent A → (handoff) → Agent B → User
```

**Flow:**
1. Current agent decides it needs help
2. Passes control to next agent
3. New agent interacts directly with user

**Use when:**
- Agents need to converse directly with users
- Complex, human-like conversations between specialists
- Agent switching based on conversation flow

---

## Tool Calling Pattern: Supervisor Agent

The supervisor calls sub-agents as tools. Sub-agents perform tasks and return results.

### Basic Implementation

```python
from langchain.tools import tool
from langchain.agents import create_agent

# Create a sub-agent
calendar_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],  # Give it relevant tools
    system_prompt="You are a calendar specialist. Handle scheduling tasks.",
    name="calendar_agent"
)

# Wrap sub-agent as a tool
@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language."""
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content

# Create supervisor with access to sub-agent tools
supervisor = create_agent(
    model="openai:gpt-4o-mini",
    tools=[schedule_event],
    system_prompt="You coordinate calendar and scheduling tasks.",
    name="supervisor"
)

# Use the supervisor
response = supervisor.invoke({
    "messages": [{"role": "user", "content": "Schedule a meeting tomorrow at 2pm"}]
})
print(response["messages"][-1].content)
```

### Key Points

1. **Sub-agent name and description matter** - The supervisor uses these to decide when to call sub-agents
2. **Sub-agents don't talk to users** - They return results to the supervisor
3. **Centralized control** - All routing goes through the supervisor

---

## Customizing Context Flow

The quality of multi-agent systems **depends heavily on context engineering** - controlling what information each agent sees.

### Control Input to Sub-Agents

**1. Via tool description:**

```python
@tool
def calendar_tool(request: str) -> str:
    """
    Handle calendar tasks including:
    - Scheduling meetings
    - Checking availability  
    - Managing events
    
    Be specific about time, date, and participants.
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content
```

**2. Via context injection:**

```python
from langchain.tools import ToolRuntime
from langchain.agents import AgentState

@tool
def calendar_tool(request: str, runtime: ToolRuntime[None, AgentState]) -> str:
    """Handle calendar tasks."""
    # Pass conversation history to sub-agent
    full_context = f"User request: {request}\n\nHistory: {runtime.state['messages']}"
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": full_context}]
    })
    return result["messages"][-1].content
```

### Control Output from Sub-Agents

**1. Via sub-agent prompt:**

```python
sub_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    system_prompt=(
        "You are a calendar specialist. "
        "IMPORTANT: Include all relevant details in your final response. "
        "The supervisor only sees your final message, not intermediate steps."
    ),
    name="calendar_agent"
)
```

**2. Via custom formatting:**

```python
from typing import Annotated
from langchain.tools import InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage

@tool
def calendar_tool(
    request: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Handle calendar tasks."""
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    # Return custom state with the result
    return Command(update={
        "messages": [
            ToolMessage(
                content=result["messages"][-1].content,
                tool_call_id=tool_call_id
            )
        ]
    })
```

---

## Best Practices

**1. Clear domain boundaries**
- Each sub-agent handles one specific domain
- Don't overlap responsibilities

**2. Focused tools and prompts**
- Give sub-agents only tools they need
- Write clear, specific system prompts

**3. Clear tool descriptions**
- Supervisor uses these to decide when to call sub-agents
- Be specific about what each tool handles

**4. Test independently**
- Test each sub-agent separately before integration
- Verify supervisor routing logic

**5. Emphasize final response in sub-agent prompts**
- Common failure: sub-agents perform tool calls but don't include results in final message
- Remind sub-agents that supervisor only sees their final output

---

## When to Use Multi-Agent

**Use multi-agent when:**
- Multiple distinct domains (calendar, email, CRM, database)
- Each domain has multiple tools or complex logic
- Want centralized workflow control
- Sub-agents don't need to converse with users

**Use single agent when:**
- Simple cases with few tools
- Tasks within one domain

**Use handoffs when:**
- Agents need to converse directly with users
- Agent-to-agent conversations required

---

## Quick Reference

### Supervisor Pattern (Tool Calling)

```python
from langchain.tools import tool
from langchain.agents import create_agent

# Step 1: Create sub-agent
sub_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],  # Give it relevant tools
    system_prompt="You are a specialist. Include all details in final response.",
    name="sub_agent"
)

# Step 2: Wrap as tool
@tool
def call_sub_agent(request: str) -> str:
    """Clear description of what this sub-agent handles."""
    result = sub_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content

# Step 3: Create supervisor
supervisor = create_agent(
    model="openai:gpt-4o-mini",
    tools=[call_sub_agent],
    system_prompt="You coordinate tasks using available sub-agents.",
    name="supervisor"
)

# Step 4: Use
response = supervisor.invoke({
    "messages": [{"role": "user", "content": "your query"}]
})
print(response["messages"][-1].content)
```

---

## Key Takeaways

- **Two patterns**: Tool calling (centralized) and handoffs (decentralized)
- **Tool calling = supervisor pattern**: Sub-agents are called as tools
- **Context engineering is critical**: Control what each agent sees
- **Sub-agent prompts must emphasize final output**: Include all results in final message
- **Test independently**: Verify each sub-agent before integration

---

## Resources

- [Multi-Agent Documentation](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [Supervisor Tutorial](https://docs.langchain.com/oss/python/langchain/supervisor)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph)
- [Previous: Agentic RAG](../rag/10-agentic-rag.md)
