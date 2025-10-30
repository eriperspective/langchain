# Model Context

**Model Context** controls what goes into each LLM call - instructions, available tools, which model to use, and output format. These are **transient** changes that only affect a single model call without permanently changing state.

## What You Can Control

1. **System Prompt** - Base instructions for the LLM
2. **Messages** - Conversation history sent to the LLM
3. **Tools** - Which tools are available for this call
4. **Model** - Which model to use
5. **Response Format** - Schema for structured output

## Key Concept: Transient vs Persistent

**Transient** (Model Context): Changes only affect the current model call. The original state is unchanged.

**Persistent** (Life-cycle Context): Changes are saved to state permanently.

---

## 1. Dynamic System Prompt

Change instructions based on conversation state.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def conversation_aware_prompt(request: ModelRequest) -> str:
    """Adjust instructions based on conversation length."""
    # request.messages gives you the conversation history
    message_count = len(request.messages)
    
    base = "You are a helpful assistant."
    
    if message_count > 10:
        base += "\nThis is a long conversation - be extra concise."
    
    return base

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[conversation_aware_prompt],
    name="adaptive_assistant"
)
```

### When to Use:
- Adjust tone based on conversation stage
- Add context-specific instructions
- Change behavior based on user preferences

---

## 2. Injecting Additional Messages

Add context from uploaded files or session data without permanently modifying state.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def inject_file_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Add information about uploaded files to the conversation."""
    
    # Read from state: get uploaded files
    uploaded_files = request.state.get("uploaded_files", [])
    
    if uploaded_files:
        # Build context message
        file_list = []
        for file in uploaded_files:
            file_list.append(f"- {file['name']} ({file['type']})")
        
        file_context = f"""Files available in this conversation:
{chr(10).join(file_list)}

You can reference these files when answering."""
        
        # Inject context before processing
        messages = [
            *request.messages,
            {"role": "system", "content": file_context}
        ]
        request = request.override(messages=messages)
    
    # Continue with modified request
    return handler(request)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[inject_file_context],
    name="file_aware_assistant"
)

# Example usage with state
result = agent.invoke({
    "messages": [{"role": "user", "content": "What files do I have?"}],
    "uploaded_files": [
        {"name": "report.pdf", "type": "PDF"},
        {"name": "data.csv", "type": "CSV"}
    ]
})
```

### When to Use:
- Add session-specific context
- Inject user preferences
- Include relevant background information

---

## 3. Dynamic Tool Selection

Show different tools based on conversation state or user permissions.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool
from typing import Callable

# Define tools
@tool
def basic_search(query: str) -> str:
    """Search for basic information."""
    return f"Basic search results for: {query}"

@tool
def advanced_search(query: str) -> str:
    """Perform advanced search with filters."""
    return f"Advanced search results for: {query}"

@wrap_model_call
def permission_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Show tools based on user permissions."""
    
    # Check user permission from state
    is_premium = request.state.get("is_premium_user", False)
    
    if is_premium:
        # Premium users get both tools
        tools = [basic_search, advanced_search]
    else:
        # Free users get basic tool only
        tools = [basic_search]
    
    request = request.override(tools=tools)
    return handler(request)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[basic_search, advanced_search],
    middleware=[permission_based_tools],
    name="permission_aware_agent"
)

# Example: Free user
result = agent.invoke({
    "messages": [{"role": "user", "content": "Search for Python tutorials"}],
    "is_premium_user": False
})

# Example: Premium user
result = agent.invoke({
    "messages": [{"role": "user", "content": "Search for Python tutorials"}],
    "is_premium_user": True
})
```

### When to Use:
- Control tool access based on permissions
- Show different capabilities per user tier
- Enable/disable tools based on context

---

## 4. Dynamic Model Selection

Switch between models based on conversation complexity or cost constraints.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def smart_model_selection(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Use faster model for simple queries, powerful model for complex ones."""
    
    # Get the latest user message
    latest_message = request.messages[-1]["content"] if request.messages else ""
    
    # Simple heuristic: long messages or certain keywords = complex
    is_complex = (
        len(latest_message) > 200 or
        any(word in latest_message.lower() for word in ["analyze", "compare", "explain in detail"])
    )
    
    if is_complex:
        model = "openai:gpt-4o"  # More capable model
    else:
        model = "openai:gpt-4o-mini"  # Faster, cheaper model
    
    request = request.override(model=model)
    return handler(request)

agent = create_agent(
    model="openai:gpt-4o-mini",  # Default model
    tools=[],
    middleware=[smart_model_selection],
    name="smart_routing_agent"
)

# Simple query - uses gpt-4o-mini
result = agent.invoke({
    "messages": [{"role": "user", "content": "What is Python?"}]
})

# Complex query - switches to gpt-4o
result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze and compare the performance characteristics of Python's list vs deque for various operations, explaining in detail when to use each."}]
})
```

### When to Use:
- Balance cost and performance
- Route simple queries to faster models
- Use powerful models for complex reasoning

---

## 5. Structured Output (Response Format)

Force the model to return data in a specific schema.

```python
from langchain.agents import create_agent
from pydantic import BaseModel, Field

# Define output schema
class CustomerTicket(BaseModel):
    """Structured information extracted from customer message."""
    
    category: str = Field(
        description="Issue category: 'billing', 'technical', 'account', or 'product'"
    )
    priority: str = Field(
        description="Urgency: 'low', 'medium', 'high', or 'critical'"
    )
    summary: str = Field(
        description="One-sentence summary of the issue"
    )

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    response_format=CustomerTicket,  # Forces structured output
    name="ticket_classifier"
)

# The response will be a CustomerTicket object
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "I've been charged twice for my subscription this month! I need this fixed ASAP!"
    }]
})

# Access structured fields
print(result["response"].category)  # "billing"
print(result["response"].priority)  # "high"
print(result["response"].summary)   # "Customer was double-charged for subscription"
```

### When to Use:
- Extract specific fields from text
- Ensure consistent output format
- Feed data to downstream systems

---

## 6. Dynamic Response Format

Change output schema based on conversation state.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from pydantic import BaseModel, Field
from typing import Callable

class SimpleResponse(BaseModel):
    """Quick response for early conversation."""
    answer: str = Field(description="A brief answer")

class DetailedResponse(BaseModel):
    """Comprehensive response for established conversation."""
    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Step-by-step reasoning")
    confidence: float = Field(description="Confidence score 0-1")
    sources: list[str] = Field(description="Information sources used")

@wrap_model_call
def adaptive_output_format(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Use simple format early, detailed format later."""
    
    message_count = len(request.messages)
    
    if message_count < 3:
        # Early conversation - keep it simple
        request = request.override(response_format=SimpleResponse)
    else:
        # Established conversation - provide details
        request = request.override(response_format=DetailedResponse)
    
    return handler(request)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[adaptive_output_format],
    name="adaptive_response_agent"
)
```

### When to Use:
- Start simple, add detail as needed
- Different formats for different user tiers
- Adapt output based on conversation stage

---

## Key Takeaways

1. **Transient Changes**: Model context modifications don't persist to state
2. **Use `wrap_model_call`**: For intercepting and modifying model requests
3. **Use `dynamic_prompt`**: Simple way to change system instructions
4. **Access State**: Read from `request.state` to make context-aware decisions
5. **Override Request**: Use `request.override()` to change model call parameters

## Common Patterns

**Read from state** → **Make decision** → **Override request** → **Continue processing**

This is the core pattern for all model context engineering.

## Next Steps

- Learn about **Tool Context** to control what tools read/write
- Learn about **Life-cycle Context** for persistent changes
- Explore the [Middleware documentation](https://docs.langchain.com/oss/python/langchain/middleware) for advanced patterns

