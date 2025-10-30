# Life-cycle Context

**Life-cycle Context** controls what happens **between** the core agent steps. Unlike model context (transient) and tool context (writes only), life-cycle hooks let you intercept the agent loop to implement cross-cutting concerns like summarization, guardrails, and logging.

## What You Can Control

1. **Between Steps** - Intercept data flow between model and tool calls
2. **State Updates** - Persistently modify conversation history
3. **Flow Control** - Skip steps or repeat with modified context

## The Agent Loop

```
┌─────────────────────────────────────────┐
│         before_model (hook)             │
│  ↓                                      │
│         MODEL CALL                      │
│  ↓                                      │
│         after_model (hook)              │
│  ↓                                      │
│         before_tools (hook)             │
│  ↓                                      │
│         TOOL EXECUTION                  │
│  ↓                                      │
│         after_tools (hook)              │
└─────────────────────────────────────────┘
```

---

## 1. Automatic Message Summarization

The most common life-cycle pattern: condense long conversations.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="openai:gpt-4o-mini",
            max_tokens_before_summary=4000,  # Trigger at 4000 tokens
            messages_to_keep=20,             # Keep last 20 messages
        ),
    ],
    name="summarizing_agent"
)

# As the conversation grows, old messages are automatically
# replaced with a summary in State (permanently)
```

### How It Works:
1. Counts tokens in conversation history
2. When limit exceeded, summarizes older messages
3. Replaces old messages with summary in State
4. Keeps recent messages for context

### When to Use:
- Long conversations that exceed context limits
- Cost reduction (fewer tokens)
- Maintain essential context while pruning details

---

## 2. Content Moderation (Before Model)

Filter inappropriate content before it reaches the model.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langgraph.errors import GraphInterrupt

@before_model
def moderate_content(state: dict) -> dict:
    """Check for inappropriate content before model call."""
    
    # Get the latest user message
    messages = state.get("messages", [])
    if not messages:
        return state
    
    latest_message = messages[-1]
    
    # Simple keyword filter (in production, use a real moderation API)
    banned_words = ["spam", "scam", "hack"]
    content = latest_message.get("content", "").lower()
    
    if any(word in content for word in banned_words):
        # Add a response and stop processing
        state["messages"].append({
            "role": "assistant",
            "content": "I cannot respond to that request as it violates our content policy."
        })
        # Interrupt the graph to stop further processing
        raise GraphInterrupt(
            "Content moderation triggered",
            state=state
        )
    
    return state

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[moderate_content],
    name="moderated_agent"
)

# Inappropriate messages are blocked
result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I spam people?"}]
})
```

### When to Use:
- Content filtering
- Input validation
- Safety checks before expensive model calls

---

## 3. Logging and Monitoring (After Model)

Track model responses for analytics or debugging.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import after_model
import json
from datetime import datetime

@after_model
def log_model_calls(state: dict) -> dict:
    """Log all model responses for monitoring."""
    
    messages = state.get("messages", [])
    if not messages:
        return state
    
    # Get the latest assistant message
    latest_message = messages[-1]
    
    if latest_message.get("role") == "assistant":
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "content": latest_message.get("content", ""),
            "tool_calls": latest_message.get("tool_calls", []),
            "message_count": len(messages)
        }
        
        # In production, send to logging service
        print(f"[MODEL LOG] {json.dumps(log_entry, indent=2)}")
    
    return state

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[log_model_calls],
    name="logged_agent"
)

# All model responses are logged
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello!"}]
})
```

### When to Use:
- Analytics and monitoring
- Debugging agent behavior
- Audit trails
- Cost tracking

---

## 4. Token Budget Enforcement (Before Model)

Prevent expensive model calls from exceeding budget.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langgraph.errors import GraphInterrupt

@before_model
def enforce_token_budget(state: dict) -> dict:
    """Stop processing if token budget exceeded."""
    
    # Track tokens used (simplified - in production, use actual token counting)
    tokens_used = state.get("tokens_used", 0)
    max_tokens = state.get("token_budget", 10000)
    
    if tokens_used >= max_tokens:
        state["messages"].append({
            "role": "assistant",
            "content": "Token budget exceeded. Please start a new conversation."
        })
        raise GraphInterrupt(
            "Token budget exceeded",
            state=state
        )
    
    return state

@after_model
def track_token_usage(state: dict) -> dict:
    """Update token count after each model call."""
    
    # Estimate tokens (in production, get actual usage from model response)
    messages = state.get("messages", [])
    estimated_tokens = sum(len(msg.get("content", "")) // 4 for msg in messages)
    
    state["tokens_used"] = estimated_tokens
    
    return state

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[enforce_token_budget, track_token_usage],
    name="budgeted_agent"
)

# Usage with budget
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello!"}],
    "token_budget": 10000,
    "tokens_used": 0
})
```

### When to Use:
- Cost control
- Per-user token limits
- Prevent runaway conversations

---

## 5. Response Formatting (After Model)

Transform model output before final return.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import after_model

@after_model
def add_citations(state: dict) -> dict:
    """Add source citations to responses."""
    
    messages = state.get("messages", [])
    if not messages:
        return state
    
    latest_message = messages[-1]
    
    if latest_message.get("role") == "assistant":
        content = latest_message.get("content", "")
        
        # Add citation footer
        citation = "\n\n---\n*Response generated by AI assistant. Verify important information.*"
        
        latest_message["content"] = content + citation
        messages[-1] = latest_message
        state["messages"] = messages
    
    return state

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[add_citations],
    name="citing_agent"
)

# All responses include citation footer
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is Python?"}]
})
```

### When to Use:
- Add disclaimers
- Format responses for UI
- Add metadata or attribution

---

## 6. Tool Result Validation (After Tools)

Verify tool outputs before continuing.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import after_tools
from langchain.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the database for information."""
    # Simulate database search
    if "error" in query.lower():
        return "ERROR: Database connection failed"
    return f"Results for: {query}"

@after_tools
def validate_tool_results(state: dict) -> dict:
    """Check tool results for errors and handle them."""
    
    messages = state.get("messages", [])
    if not messages:
        return state
    
    latest_message = messages[-1]
    
    # Check if this is a tool result
    if latest_message.get("type") == "tool":
        content = latest_message.get("content", "")
        
        # Check for error in tool result
        if "ERROR:" in content:
            # Inject a helpful message for the model
            messages.append({
                "role": "system",
                "content": "The tool encountered an error. Please inform the user and suggest alternatives."
            })
            state["messages"] = messages
    
    return state

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_database],
    middleware=[validate_tool_results],
    name="validated_agent"
)

# Tool errors are caught and handled gracefully
result = agent.invoke({
    "messages": [{"role": "user", "content": "Search for error data"}]
})
```

### When to Use:
- Error handling
- Result validation
- Retry logic
- Fallback strategies

---

## 7. Multi-hook Example: Complete Monitoring

Combining multiple hooks for comprehensive monitoring.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import before_model, after_model, after_tools
from datetime import datetime
import json

# State to track metrics
def init_metrics(state: dict) -> dict:
    """Initialize metrics tracking in state."""
    if "metrics" not in state:
        state["metrics"] = {
            "model_calls": 0,
            "tool_calls": 0,
            "start_time": datetime.now().isoformat()
        }
    return state

@before_model
def track_model_call_start(state: dict) -> dict:
    """Track when model call starts."""
    state = init_metrics(state)
    state["metrics"]["model_calls"] += 1
    state["current_model_start"] = datetime.now().isoformat()
    return state

@after_model
def track_model_call_end(state: dict) -> dict:
    """Track model call completion."""
    if "current_model_start" in state:
        start = datetime.fromisoformat(state["current_model_start"])
        duration = (datetime.now() - start).total_seconds()
        print(f"[METRICS] Model call took {duration:.2f}s")
        del state["current_model_start"]
    return state

@after_tools
def track_tool_execution(state: dict) -> dict:
    """Track tool usage."""
    state = init_metrics(state)
    
    messages = state.get("messages", [])
    if messages and messages[-1].get("type") == "tool":
        state["metrics"]["tool_calls"] += 1
    
    return state

@after_model
def log_final_metrics(state: dict) -> dict:
    """Log metrics at end of conversation."""
    messages = state.get("messages", [])
    if not messages:
        return state
    
    latest = messages[-1]
    
    # Check if conversation is ending (no tool calls requested)
    if latest.get("role") == "assistant" and not latest.get("tool_calls"):
        metrics = state.get("metrics", {})
        print(f"[METRICS] Session complete: {json.dumps(metrics, indent=2)}")
    
    return state

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[
        track_model_call_start,
        track_model_call_end,
        track_tool_execution,
        log_final_metrics
    ],
    name="monitored_agent"
)

# Complete metrics tracked throughout conversation
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello!"}]
})
```

### When to Use:
- Production monitoring
- Performance analytics
- Usage tracking
- Billing metrics

---

## 8. Custom Message Trimming (Before Model)

Alternative to summarization - simply drop old messages.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import before_model

@before_model
def trim_old_messages(state: dict) -> dict:
    """Keep only the last N messages to stay within context limits."""
    
    messages = state.get("messages", [])
    max_messages = 20  # Keep last 20 messages
    
    if len(messages) > max_messages:
        # Keep system message (if present) plus recent messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        recent_msgs = messages[-max_messages:]
        
        state["messages"] = system_msgs + recent_msgs
        
        # Optionally log that we trimmed
        print(f"[TRIM] Removed {len(messages) - len(state['messages'])} old messages")
    
    return state

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[trim_old_messages],
    name="trimming_agent"
)

# Conversation stays within limits automatically
```

### When to Use:
- Simpler than summarization
- When old context isn't critical
- Cost optimization
- Fixed context window management

---

## Key Takeaways

1. **Life-cycle Hooks**: `before_model`, `after_model`, `before_tools`, `after_tools`
2. **Persistent Changes**: Life-cycle hooks modify state permanently
3. **Cross-cutting Concerns**: Logging, monitoring, validation, formatting
4. **Flow Control**: Use `GraphInterrupt` to stop processing when needed
5. **Multiple Hooks**: Combine hooks for comprehensive control

## Hook Types

| Hook | When It Runs | Common Uses |
|------|-------------|-------------|
| `before_model` | Before LLM call | Validation, moderation, trimming |
| `after_model` | After LLM responds | Logging, formatting, metrics |
| `before_tools` | Before tool execution | Authorization, validation |
| `after_tools` | After tools complete | Error handling, result validation |

## Common Patterns

1. **Read-Modify-Write**: Read state → Make changes → Update state
2. **Guard Pattern**: Check condition → Interrupt if needed
3. **Wrap Pattern**: Before hook sets up → After hook tears down
4. **Metrics Pattern**: Track metrics across multiple hooks

## Next Steps

- Review the complete [Middleware documentation](https://docs.langchain.com/oss/python/langchain/middleware)
- Learn about [Guardrails](https://docs.langchain.com/oss/python/langchain/guardrails)
- Explore [Runtime patterns](https://docs.langchain.com/oss/python/langchain/runtime)
- Combine with **Model Context** and **Tool Context** for full control

