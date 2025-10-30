# Managing Short-Term Memory in LangChain Agents

[LangChain Docs - Short-term memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)

### Overview

Memory is what allows an AI agent to recall information from previous interactions within a single conversation. For an application like an AI tutor, this is essential. It enables the agent to remember the student's name, track their progress through a lesson, and refer back to earlier questions, creating a coherent and personalized learning experience.

This is known as **short-term memory**, where the context is maintained within a specific conversation **thread**. Without it, each interaction would be isolated and stateless, making meaningful conversation impossible.

The primary challenge with memory is the limited **context window** of most LLMs. As a conversation grows, the full history of messages may no longer fit, leading to errors or a decline in performance as the model gets bogged down by old information. Therefore, managing this memory effectively is a key part of building robust agents.

### Basic Usage

To enable short-term memory, you must provide a **checkpointer** when creating an agent. The checkpointer is responsible for saving and loading the agent's state for each conversation thread.

For development and testing, you can use the simple `InMemorySaver`.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# The checkpointer enables memory for the agent.
agent = create_agent(
    "openai:gpt-4o",
    tools=[...],
    checkpointer=InMemorySaver(),
)

# A unique thread_id ensures conversations are kept separate.
config = {"configurable": {"thread_id": "math_lesson_1"}}

# First interaction
agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Alex."}]},
    config,
)

# Second interaction in the same thread
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What was my name again?"}]},
    config,
)

# The agent remembers the name from the first interaction.
print(response["messages"][-1].content)
# Output: "Your name is Alex."
```

For production applications, you should use a persistent checkpointer backed by a database like Postgres or Redis to ensure memory is not lost.

### Customizing Agent Memory

By default, an agent's memory (`AgentState`) only stores the conversation history in a `messages` field. You can extend this to track additional information relevant to your application.

For our AI Tutor, we might want to store the student's name and the current topic of the lesson.

```python
from langchain.agents import AgentState

# Inherit from AgentState to add custom fields
class TutorAgentState(AgentState):
    student_name: str
    current_topic: str

# Pass the custom schema when creating the agent
agent = create_agent(
    "openai:gpt-4o",
    tools=[...],
    state_schema=TutorAgentState,
    checkpointer=InMemorySaver(),
)

# Now you can pass these custom state values during invocation
result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Let's start with algebra."}],
        "student_name": "Alex",
        "current_topic": "Algebra"
    },
    {"configurable": {"thread_id": "math_lesson_1"}}
)
```

### Common Patterns for Managing Long Conversations

As a tutoring session progresses, the conversation history can exceed the model's context window. Here are common strategies to manage this.

#### Trim Messages

This strategy involves removing older messages from the history to make space for new ones. You can implement this using `@before_model` middleware, which modifies the agent's state just before calling the LLM. This example keeps the first system message and the most recent four messages.

```python
from langchain.agents.middleware import before_model
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from typing import Any

@before_model
def trim_conversation_history(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep the system prompt and the last 4 messages."""
    messages = state["messages"]
    
    # Only trim if the conversation is long enough
    if len(messages) <= 5:
        return None

    # Keep the first message (system prompt) and the last four
    new_history = [messages[0]] + messages[-4:]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_history
        ]
    }

# Add the middleware when creating the agent
agent = create_agent(
    ...,
    middleware=[trim_conversation_history],
    ...
)
```

#### Summarize Messages

Instead of simply deleting old messages and losing that context, a more sophisticated approach is to summarize them. This condenses the information, retaining key details while using far fewer tokens. LangChain provides a built-in `SummarizationMiddleware` for this.

```python
from langchain.agents.middleware import SummarizationMiddleware

# This middleware will automatically summarize the conversation
# when it exceeds a certain token count.
summarizer = SummarizationMiddleware(
    model="openai:gpt-4o-mini", # Use a fast model for summarizing
    max_tokens_before_summary=4000, # Summarize after 4000 tokens
    messages_to_keep=10, # Keep the last 10 messages untouched after summarizing
)

agent = create_agent(
    ...,
    middleware=[summarizer],
    ...
)
```

### Accessing Memory in Your Application

You can read from and write to the agent's memory from various parts of your application.

#### In Tools

Tools can access the agent's state via the `ToolRuntime` parameter. This allows them to perform context-aware actions.

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

# Reading from custom state
@tool
def get_current_topic(runtime: ToolRuntime[..., TutorAgentState]) -> str:
    """Gets the current topic of the lesson from memory."""
    return f"We are currently studying: {runtime.state['current_topic']}"

# Writing to custom state
@tool
def change_topic(new_topic: str) -> Command:
    """Updates the lesson topic in the agent's memory."""
    return Command(update={"current_topic": new_topic})
```

#### In Prompts and Middleware

You can also access memory to create dynamic prompts or to process messages before or after a model call.

  * **Dynamic Prompts**: Use middleware to create a system prompt that is personalized with information from the state.

    ```python
    from langchain.agents.middleware import dynamic_prompt, ModelRequest

    @dynamic_prompt
    def personalized_tutor_prompt(request: ModelRequest) -> str:
        student_name = request.state.get("student_name", "Student")
        return f"You are a friendly and encouraging tutor. Always address the user as {student_name}."
    ```

  * **Middleware**: Process the state before (`@before_model`) or after (`@after_model`) the LLM call. The message trimming example above uses `@before_model`. An `@after_model` example could be used to check the AI's response for complexity and add a follow-up if needed.