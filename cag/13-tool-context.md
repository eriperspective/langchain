# Tool Context

**Tool Context** controls what tools can access (reads) and produce (writes). Unlike model context which is transient, tool writes are **persistent** - they modify state and store permanently.

## What You Can Control

1. **Reads** - What information tools access (state, store, runtime context)
2. **Writes** - What tools save or update (state, store)

## Data Sources Tools Can Access

| Data Source | Scope | Examples |
|------------|-------|----------|
| **State** | Current conversation | Messages, session data, temp flags |
| **Store** | Cross-conversation | User preferences, memories, history |
| **Runtime Context** | Static config | API keys, user ID, permissions |

---

## 1. Reading from State

Access current conversation data within a tool.

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

@tool
def check_authentication(
    runtime: ToolRuntime
) -> str:
    """Check if the user is currently authenticated."""
    
    # Read from State: access current session data
    current_state = runtime.state
    is_authenticated = current_state.get("authenticated", False)
    username = current_state.get("username", "Guest")
    
    if is_authenticated:
        return f"User '{username}' is authenticated"
    else:
        return "User is not authenticated. Please log in."

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[check_authentication],
    name="auth_checker"
)

# Example: Not authenticated
result = agent.invoke({
    "messages": [{"role": "user", "content": "Am I logged in?"}],
    "authenticated": False
})

# Example: Authenticated
result = agent.invoke({
    "messages": [{"role": "user", "content": "Am I logged in?"}],
    "authenticated": True,
    "username": "alice"
})
```

### When to Use:
- Check session flags (authenticated, premium_user, etc.)
- Read temporary data from current conversation
- Access user input or file uploads from this session

---

## 2. Reading from Store (Long-term Memory)

Access user data across conversations.

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

@tool
def get_user_preferences(
    runtime: ToolRuntime
) -> str:
    """Get the user's saved preferences."""
    
    # Read from Store: cross-conversation data
    # Store is keyed by user_id from runtime context
    store = runtime.store
    user_id = runtime.config.get("user_id", "default")
    
    # Get preferences from long-term memory
    prefs = store.get(
        namespace=["users", user_id],
        key="preferences"
    )
    
    if prefs:
        return f"User preferences: {prefs.value}"
    else:
        return "No preferences saved yet."

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_user_preferences],
    name="preference_reader"
)

# Usage with store
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
# Pre-populate some preferences
store.put(
    namespace=["users", "alice"],
    key="preferences",
    value={"theme": "dark", "language": "en"}
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What are my preferences?"}]},
    config={"configurable": {"user_id": "alice"}},
    store=store
)
```

### When to Use:
- Load user preferences
- Access historical data
- Read saved insights or memories

---

## 3. Reading from Runtime Context

Access static configuration like API keys and permissions.

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

@tool
def make_api_request(
    endpoint: str,
    runtime: ToolRuntime
) -> str:
    """Make a request to an external API."""
    
    # Read from Runtime Context: static configuration
    api_key = runtime.config.get("api_key")
    user_tier = runtime.config.get("user_tier", "free")
    
    if not api_key:
        return "Error: No API key configured"
    
    # Check permissions based on user tier
    if endpoint == "/premium" and user_tier != "premium":
        return "Error: Premium endpoint requires premium subscription"
    
    # Simulate API call
    return f"Success: Called {endpoint} with tier={user_tier}"

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[make_api_request],
    name="api_caller"
)

# Usage with runtime context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Call the /premium endpoint"}]},
    config={
        "configurable": {
            "api_key": "sk-12345",
            "user_tier": "premium"
        }
    }
)
```

### When to Use:
- Access API keys and credentials
- Check user permissions or roles
- Read environment-specific configuration

---

## 4. Writing to State

Update current session data using `Command`.

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.types import Command

@tool
def authenticate_user(
    username: str,
    password: str,
    runtime: ToolRuntime
) -> Command:
    """Authenticate a user and update session state."""
    
    # Simulate authentication check
    if password == "correct":
        # Write to State: mark as authenticated
        return Command(
            update={
                "authenticated": True,
                "username": username,
                "login_time": "2025-10-23T10:30:00"
            },
            # Still return a message to the model
            result="Authentication successful! Welcome back."
        )
    else:
        return Command(
            update={"authenticated": False},
            result="Authentication failed. Invalid password."
        )

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[authenticate_user],
    name="auth_agent"
)

# The state will be updated after tool execution
result = agent.invoke({
    "messages": [{"role": "user", "content": "Log me in as alice, password: correct"}],
    "authenticated": False
})

# Check the updated state
print(result["authenticated"])  # True
print(result["username"])       # "alice"
```

### When to Use:
- Update session flags
- Save temporary data for current conversation
- Track conversation state changes

---

## 5. Writing to Store (Long-term Memory)

Save data that persists across conversations.

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.types import Command

@tool
def save_user_preference(
    preference_name: str,
    preference_value: str,
    runtime: ToolRuntime
) -> Command:
    """Save a user preference for future conversations."""
    
    # Get user ID from runtime context
    user_id = runtime.config.get("user_id", "default")
    
    # Read existing preferences from Store
    store = runtime.store
    existing_prefs = store.get(
        namespace=["users", user_id],
        key="preferences"
    )
    
    # Update preferences
    if existing_prefs:
        prefs = existing_prefs.value
    else:
        prefs = {}
    
    prefs[preference_name] = preference_value
    
    # Write to Store: save for future conversations
    store.put(
        namespace=["users", user_id],
        key="preferences",
        value=prefs
    )
    
    return Command(
        result=f"Saved preference: {preference_name}={preference_value}"
    )

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[save_user_preference],
    name="preference_saver"
)

# Usage with store
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Set my theme to dark mode"}]},
    config={"configurable": {"user_id": "alice"}},
    store=store
)

# Preference is now saved in store for future conversations
```

### When to Use:
- Save user preferences
- Store learned insights
- Build long-term memory

---

## 6. Complete Example: File Upload Tool

Combining reads and writes for a realistic tool.

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.types import Command
from typing import Annotated
from langchain.tools import InjectedToolCallId

@tool
def upload_file(
    filename: str,
    file_type: str,
    content_summary: str,
    runtime: ToolRuntime,
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Upload a file and track it in the current session."""
    
    # Read from State: get existing files
    current_state = runtime.state
    uploaded_files = current_state.get("uploaded_files", [])
    
    # Read from Runtime Context: check storage limit
    max_files = runtime.config.get("max_file_uploads", 5)
    
    if len(uploaded_files) >= max_files:
        return Command(
            result=f"Error: Maximum {max_files} files allowed"
        )
    
    # Create file record
    file_record = {
        "name": filename,
        "type": file_type,
        "summary": content_summary,
        "uploaded_at": "2025-10-23T10:30:00"
    }
    
    # Add to files list
    uploaded_files.append(file_record)
    
    # Write to State: save updated file list
    return Command(
        update={"uploaded_files": uploaded_files},
        result=f"Successfully uploaded {filename}. You now have {len(uploaded_files)} file(s)."
    )

@tool
def list_files(
    runtime: ToolRuntime
) -> str:
    """List all uploaded files in the current session."""
    
    # Read from State
    uploaded_files = runtime.state.get("uploaded_files", [])
    
    if not uploaded_files:
        return "No files uploaded yet."
    
    file_list = []
    for f in uploaded_files:
        file_list.append(f"- {f['name']} ({f['type']}): {f['summary']}")
    
    return "Uploaded files:\n" + "\n".join(file_list)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[upload_file, list_files],
    name="file_manager"
)

# Upload a file
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "Upload report.pdf, it's a PDF with Q4 sales data"}],
    "uploaded_files": []
}, config={"configurable": {"max_file_uploads": 5}})

# List files (state persists)
result2 = agent.invoke({
    "messages": result1["messages"] + [{"role": "user", "content": "What files do I have?"}],
    "uploaded_files": result1["uploaded_files"]
})
```

### Key Patterns:
1. **Read** existing state
2. **Validate** against runtime context limits
3. **Update** state with new data
4. **Return** result to model using Command

---

## 7. Tool Context Best Practices

### Reading Best Practices

```python
# GOOD: Provide defaults
authenticated = runtime.state.get("authenticated", False)

# GOOD: Check before using
if "user_id" in runtime.config:
    user_id = runtime.config["user_id"]
else:
    return "Error: No user ID configured"

# BAD: Assume data exists
username = runtime.state["username"]  # May crash!
```

### Writing Best Practices

```python
# GOOD: Use Command for state updates
return Command(
    update={"key": "value"},
    result="Success message"
)

# GOOD: Read-modify-write for lists/dicts
items = runtime.state.get("items", [])
items.append(new_item)
return Command(update={"items": items})

# BAD: Forget to return a result
return Command(update={"key": "value"})  # Model gets no feedback!
```

---

## Key Takeaways

1. **Three Data Sources**: State (session), Store (persistent), Runtime Context (config)
2. **Reads**: Access via `runtime.state`, `runtime.store`, `runtime.config`
3. **Writes**: Use `Command` to update state/store
4. **Persistent Changes**: Tool writes permanently modify data
5. **Always Provide Feedback**: Return results so the model knows what happened

## Common Patterns

**Read current data** → **Validate/Transform** → **Write updates** → **Return result**

This is the core pattern for tool context engineering.

## Next Steps

- Learn about **Life-cycle Context** for cross-cutting concerns
- Explore the [Tools documentation](https://docs.langchain.com/oss/python/langchain/tools)
- Read about [Memory patterns](https://docs.langchain.com/oss/python/langchain/long-term-memory)

