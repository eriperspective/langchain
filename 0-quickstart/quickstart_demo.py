"""
LangChain Quickstart Demo
==========================
This demonstrates the basics of building agents with LangChain v1.0.

What we'll build:
1. Part 1: Simple flight status agent with one tool
2. Part 2: Advanced agent with memory, multiple tools, and context awareness

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/oss/python/langchain
Last Updated: October 30, 2025
"""

import os
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set your OPENAI_API_KEY environment variable")
    print("Example: export OPENAI_API_KEY='your-key-here'")
    exit(1)


print("="*70)
print("PART 1: YOUR FIRST SIMPLE AGENT (FLIGHTS EDITION)")
print("="*70)

# ============================================================================
# STEP 1: Define a simple tool
# ============================================================================
# Tools are just Python functions that the agent can call.
# The @tool decorator makes them available to the agent.
# The docstring is critical - it tells the agent what the tool does.

@tool
def get_flight_status(flight_number: str) -> str:
    """Gets the status for a given flight number."""
    # This is mock data for demonstration purposes
    # In a real application, you'd call an actual API here
    return f"Flight {flight_number} is on time."


# ============================================================================
# STEP 2: Create the agent
# ============================================================================
# create_agent() is the modern v1.0 way to build agents.
# It takes:
#   - model: The LLM to use (e.g., "openai:gpt-4o")
#   - tools: A list of tools the agent can call
#   - system_prompt: Instructions for the agent's behavior
#   - name: A descriptive name for the agent

agent = create_agent(
    model="openai:gpt-4o-mini",  # Using the mini model to save costs
    tools=[get_flight_status],
    system_prompt="You are a helpful flight assistant.",
    name="simple_flight_agent"
)

# ============================================================================
# STEP 3: Use the agent
# ============================================================================
# Agents take messages as input and return messages as output
# The format is: {"messages": [{"role": "user", "content": "..."}]}

print("\nTest Query: What is the status of flight AA123?\n")

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the status of flight AA123?"}]}
)

# The response contains all messages, including the agent's final answer
print(f"Agent Response: {response['messages'][-1].content}\n")


print("\n" + "="*70)
print("PART 2: ADVANCED FLIGHT AGENT WITH MEMORY AND CONTEXT")
print("="*70)

# ============================================================================
# STEP 1: Write a detailed system prompt
# ============================================================================
# A good system prompt sets personality, rules, and instructions for tool use

SYSTEM_PROMPT = """Your goal is to provide passengers with accurate and helpful flight information. You should be professional, but with a friendly and slightly witty tone.

You have access to two tools:
- get_flight_details: Use this to get the status, gate, and departure time for a specific flight number.
- get_user_home_airport: If a user asks about flights from 'my airport' or 'home', use this to find their registered home airport.

Always confirm the flight number before providing details. If a user asks a general question, use their home airport to provide relevant examples."""


# ============================================================================
# STEP 2: Create smart tools with runtime context
# ============================================================================
# These tools can access user-specific information like user_id

@tool
def get_flight_details(flight_number: str) -> str:
    """Gets flight details like status, gate, and time for a given flight number."""
    # Mock data - in a real app, this would query a flight API
    flight_data = {
        "UA456": "Flight UA456 is on time, departing from Gate B12 at 8:45 PM.",
        "AA123": "Flight AA123 is delayed by 30 minutes, departing from Gate C5 at 9:15 PM.",
        "DL789": "Flight DL789 has been cancelled. Please contact airline for rebooking.",
    }
    
    return flight_data.get(
        flight_number.upper(),
        f"Flight {flight_number} information is not available at this time."
    )


# Define the context structure
# This tells the agent what data will be available at runtime
@dataclass
class Context:
    """Custom runtime context schema for user info."""
    user_id: str


# This tool uses runtime context to personalize responses
from langchain.tools import ToolRuntime

@tool
def get_user_home_airport(runtime: ToolRuntime[Context]) -> str:
    """Retrieves the user's home airport based on their user ID."""
    user_id = runtime.context.user_id
    
    # Mock database lookup - in a real app, query an actual database
    user_airports = {
        "user_abc": "JFK",
        "user_xyz": "SFO",
        "user_123": "LAX",
    }
    
    airport = user_airports.get(user_id, "unknown airport")
    return f"Your home airport is {airport}."


# ============================================================================
# STEP 3: Set up memory with a checkpointer
# ============================================================================
# Memory allows the agent to remember the conversation
# InMemorySaver stores conversation history in memory (use Redis/Postgres for production)

checkpointer = InMemorySaver()


# ============================================================================
# STEP 4: Create the advanced agent
# ============================================================================
# This agent has:
#   - Multiple tools
#   - A detailed system prompt
#   - Memory (via checkpointer)
#   - Context awareness (via context_schema)

advanced_agent = create_agent(
    model="openai:gpt-4o-mini",
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_home_airport, get_flight_details],
    context_schema=Context,
    checkpointer=checkpointer,
    name="advanced_flight_agent"
)


# ============================================================================
# STEP 5: Test the advanced agent with context and memory
# ============================================================================

# Each conversation needs a unique thread_id for memory management
config = {"configurable": {"thread_id": "flight_convo_1"}}

print("\n--- Interaction 1: User asks about their flight ---\n")
print("User: What's the status of my flight home today?")

response = advanced_agent.invoke(
    {"messages": [{"role": "user", "content": "What's the status of my flight home today?"}]},
    config=config,
    context=Context(user_id="user_abc")  # Provide the user's ID
)

print(f"Agent: {response['messages'][-1].content}")


print("\n--- Interaction 2: User provides flight number ---\n")
print("User: It's flight UA456.")

response = advanced_agent.invoke(
    {"messages": [{"role": "user", "content": "It's flight UA456."}]},
    config=config,
    context=Context(user_id="user_abc")
)

print(f"Agent: {response['messages'][-1].content}")


print("\n--- Interaction 3: User asks another question (memory test) ---\n")
print("User: What gate was that again?")

response = advanced_agent.invoke(
    {"messages": [{"role": "user", "content": "What gate was that again?"}]},
    config=config,
    context=Context(user_id="user_abc")
)

print(f"Agent: {response['messages'][-1].content}")


print("\n" + "="*70)
print("DEMO COMPLETE!")
print("="*70)
print("\nKey takeaways:")
print("1. Simple agents are created with create_agent() - just one function call")
print("2. Tools are Python functions with the @tool decorator")
print("3. System prompts guide the agent's behavior and tool usage")
print("4. Memory is enabled by adding a checkpointer")
print("5. Context awareness lets tools access user-specific data")
print("6. The agent automatically decides when to use each tool")
print("="*70 + "\n")

