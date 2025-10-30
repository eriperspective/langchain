"""
LangChain Short-Term Memory Demo
=================================
Learn how to give agents memory so they can remember conversations.

What we'll cover:
1. Basic memory with InMemorySaver
2. Custom state schemas
3. Message trimming for long conversations
4. Message summarization
5. Accessing memory in tools and middleware

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/oss/python/langchain/short-term-memory
Last Updated: October 30, 2025
"""

import os
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import (
    before_model,
    dynamic_prompt,
    SummarizationMiddleware,
    ModelRequest
)
from langgraph.types import Command

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set your OPENAI_API_KEY environment variable")
    exit(1)


print("="*70)
print("PART 1: BASIC MEMORY WITH CHECKPOINTER")
print("="*70)

# ============================================================================
# ENABLING MEMORY
# ============================================================================
# Memory allows agents to remember previous interactions within a conversation thread
# The checkpointer saves and loads the agent's state

# Create a checkpointer (InMemorySaver for development, use Postgres/Redis for production)
checkpointer = InMemorySaver()

# Create agent with memory
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    checkpointer=checkpointer,  # This enables memory
    system_prompt="You are a friendly AI tutor. Remember the student's name and what you've discussed.",
    name="tutor_agent"
)

# Each conversation needs a unique thread_id
config = {"configurable": {"thread_id": "math_lesson_1"}}

print("\n--- Interaction 1: Introduce ---")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Alex."}]},
    config
)
print(f"Agent: {response['messages'][-1].content}\n")

print("--- Interaction 2: Test memory ---")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What was my name again?"}]},
    config  # Same thread_id, so agent remembers
)
print(f"Agent: {response['messages'][-1].content}\n")

print("--- Interaction 3: New conversation (different thread) ---")
new_config = {"configurable": {"thread_id": "math_lesson_2"}}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    new_config  # Different thread_id, so agent doesn't remember Alex
)
print(f"Agent: {response['messages'][-1].content}\n")


print("="*70)
print("PART 2: CUSTOM STATE SCHEMA")
print("="*70)

# ============================================================================
# EXTENDING AGENT STATE
# ============================================================================
# By default, agents only store messages. You can add custom fields.

# Create custom state schema
class TutorAgentState(AgentState):
    """Extended state for the AI tutor."""
    student_name: str = ""
    current_topic: str = ""


# Create agent with custom state
custom_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    state_schema=TutorAgentState,
    checkpointer=InMemorySaver(),
    system_prompt="You are an AI tutor. Track the student's name and current topic.",
    name="custom_tutor_agent"
)

print("\n--- Test: Using custom state ---")
response = custom_agent.invoke(
    {
        "messages": [{"role": "user", "content": "Let's start with algebra."}],
        "student_name": "Alex",
        "current_topic": "Algebra"
    },
    {"configurable": {"thread_id": "lesson_3"}}
)
print(f"Agent: {response['messages'][-1].content}")
print(f"State - Student: {response['student_name']}, Topic: {response['current_topic']}\n")


print("="*70)
print("PART 3: MESSAGE TRIMMING")
print("="*70)

# ============================================================================
# TRIMMING MESSAGES FOR LONG CONVERSATIONS
# ============================================================================
# As conversations grow, they exceed the model's context window
# Trimming removes old messages to make space for new ones

from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from typing import Any

@before_model
def trim_conversation_history(state: AgentState, runtime) -> dict[str, Any] | None:
    """Keep the system prompt and the last 4 messages."""
    messages = state["messages"]
    
    # Only trim if the conversation is long enough
    if len(messages) <= 5:
        return None  # No changes needed

    # Keep the first message (system prompt) and the last four
    new_history = [messages[0]] + messages[-4:]

    print(f"  [Trimming] Reduced {len(messages)} messages to {len(new_history)} messages")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_history
        ]
    }


# Create agent with message trimming
trimming_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[trim_conversation_history],
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful assistant.",
    name="trimming_agent"
)

print("\n--- Test: Message trimming (simulating long conversation) ---")
config = {"configurable": {"thread_id": "long_conversation"}}

# Simulate multiple interactions
for i in range(7):
    response = trimming_agent.invoke(
        {"messages": [{"role": "user", "content": f"Message number {i+1}"}]},
        config
    )

print(f"Final response: {response['messages'][-1].content}\n")


print("="*70)
print("PART 4: MESSAGE SUMMARIZATION")
print("="*70)

# ============================================================================
# SUMMARIZING MESSAGES
# ============================================================================
# Instead of deleting old messages, summarize them to retain context

# Built-in summarization middleware
summarizer = SummarizationMiddleware(
    model="openai:gpt-4o-mini",         # Model to use for summarizing
    max_tokens_before_summary=4000,     # Summarize after 4000 tokens
    messages_to_keep=10,                # Keep last 10 messages untouched
)

summarizing_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    middleware=[summarizer],
    checkpointer=InMemorySaver(),
    system_prompt="You are a helpful assistant.",
    name="summarizing_agent"
)

print("\n--- Test: Message summarization ---")
print("Note: Summarization triggers when conversation exceeds 4000 tokens")
print("(This demo would need a longer conversation to show actual summarization)\n")


print("="*70)
print("PART 5: ACCESSING MEMORY IN TOOLS")
print("="*70)

# ============================================================================
# TOOLS CAN READ AND WRITE MEMORY
# ============================================================================

class TutorState(AgentState):
    """State with custom topic tracking."""
    current_topic: str = ""


# Tool that reads from state
@tool
def get_current_topic(runtime: ToolRuntime[..., TutorState]) -> str:
    """Gets the current topic of the lesson from memory."""
    topic = runtime.state.get('current_topic', 'No topic set')
    return f"We are currently studying: {topic}"


# Tool that writes to state
@tool
def change_topic(new_topic: str) -> Command:
    """Updates the lesson topic in the agent's memory."""
    return Command(
        update={"current_topic": new_topic},
        result=f"Changed topic to: {new_topic}"
    )


# Create agent with memory-aware tools
memory_tools_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_current_topic, change_topic],
    state_schema=TutorState,
    checkpointer=InMemorySaver(),
    system_prompt="You are a tutor. Use tools to track and change topics.",
    name="memory_tools_agent"
)

config = {"configurable": {"thread_id": "lesson_4"}}

print("\n--- Test: Tool reading memory ---")
response = memory_tools_agent.invoke(
    {
        "messages": [{"role": "user", "content": "What are we studying?"}],
        "current_topic": "Calculus"
    },
    config
)
print(f"Agent: {response['messages'][-1].content}\n")

print("--- Test: Tool writing to memory ---")
response = memory_tools_agent.invoke(
    {"messages": [{"role": "user", "content": "Let's switch to geometry."}]},
    config
)
print(f"Agent: {response['messages'][-1].content}")
print(f"Updated topic in state: {response['current_topic']}\n")


print("="*70)
print("PART 6: DYNAMIC PROMPTS WITH MEMORY")
print("="*70)

# ============================================================================
# USING MEMORY IN DYNAMIC PROMPTS
# ============================================================================

@dynamic_prompt
def personalized_tutor_prompt(request: ModelRequest) -> str:
    """Create a personalized prompt based on state."""
    student_name = request.state.get("student_name", "Student")
    return f"You are a friendly and encouraging tutor. Always address the user as {student_name}."


personalized_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],
    state_schema=TutorAgentState,
    middleware=[personalized_tutor_prompt],
    checkpointer=InMemorySaver(),
    name="personalized_tutor"
)

print("\n--- Test: Dynamic prompt with memory ---")
response = personalized_agent.invoke(
    {
        "messages": [{"role": "user", "content": "Can you help me with math?"}],
        "student_name": "Jordan"
    },
    {"configurable": {"thread_id": "lesson_5"}}
)
print(f"Agent: {response['messages'][-1].content}\n")


print("="*70)
print("DEMO COMPLETE!")
print("="*70)
print("\nKey takeaways:")
print("1. Memory is enabled by adding a checkpointer to create_agent()")
print("2. InMemorySaver for development, Postgres/Redis for production")
print("3. Each conversation needs a unique thread_id")
print("4. Same thread_id = agent remembers, different thread_id = fresh start")
print("5. Custom state schemas add fields beyond messages")
print("6. Message trimming removes old messages to manage context length")
print("7. Summarization condenses old messages instead of deleting them")
print("8. Tools can read state with ToolRuntime and write with Command")
print("9. Dynamic prompts can use state for personalization")
print("10. @before_model middleware processes state before LLM calls")
print("="*70 + "\n")

