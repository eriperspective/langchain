"""
LangSmith Integration Demo
==========================
Learn how to use LangSmith for tracing, debugging, and monitoring agents.

What this demonstrates:
1. Setting up LangSmith tracing
2. Creating a simple agent with a tool
3. Testing multiple query types
4. Understanding what appears in LangSmith traces

Prerequisites:
- OpenAI API key
- LangSmith account (sign up at smith.langchain.com)
- LangSmith API key

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/langsmith
Last Updated: October 30, 2025
"""

import os
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

# ============================================================================
# STEP 1: VERIFY API KEYS
# ============================================================================

print("="*70)
print("LANGSMITH INTEGRATION TEST - AGENT DEMO")
print("="*70)

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("\n ERROR: OPENAI_API_KEY environment variable must be set")
    print("   Set it with: export OPENAI_API_KEY='your-key-here'")
    exit(1)

# Verify LangSmith is configured
if not os.getenv("LANGSMITH_API_KEY"):
    print("\n WARNING: LANGSMITH_API_KEY not set. Tracing will not work.")
    print("   To enable tracing:")
    print("   1. Sign up at https://smith.langchain.com")
    print("   2. Get your API key from Settings > API Keys")
    print("   3. Set environment variables:")
    print("      export LANGSMITH_API_KEY='your-key-here'")
    print("      export LANGSMITH_TRACING=true")
    print("      export LANGSMITH_PROJECT='my-first-agent'")
    print("\n   Continuing without tracing...\n")
else:
    print("\n LangSmith configuration detected")
    print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print(f"   Tracing: {os.getenv('LANGSMITH_TRACING', 'false')}")
    print(f"   Endpoint: {os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')}")
    print()


# ============================================================================
# STEP 2: DEFINE TOOLS
# ============================================================================
# This tool provides weather information for different cities

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.
    
    Args:
        city: The name of the city to get weather for
        
    Returns:
        A string describing the weather
    """
    # Mock weather data for demonstration
    # In production, you'd call a real weather API
    weather_data = {
        "philadelphia": "Sunny and 72°F with light winds",
        "new york": "Cloudy with a chance of rain, 65°F",
        "san francisco": "Foggy and cool, 58°F",
        "seattle": "Rainy as always, 55°F",
        "boston": "Clear skies, 68°F",
        "chicago": "Windy and cold, 52°F",
    }
    
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"The weather in {city} is: {weather_data[city_lower]}"
    else:
        return f"Weather data for {city} is not available. It's probably nice though!"


# ============================================================================
# STEP 3: CREATE AND TEST THE AGENT
# ============================================================================

print("\nCreating weather agent...\n")

# Create the agent - THIS IS IT! Just one simple call.
# LangSmith automatically traces all operations if environment variables are set.
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    system_prompt=(
        "You are a helpful AI assistant. "
        "For weather-related questions, use the get_weather tool. "
        "For general questions, answer using your knowledge."
    ),
    name="weather_agent"
)

# Test cases that demonstrate different agent behaviors
test_queries = [
    "What is the weather in Philadelphia?",
    "What is the capital of France?",
    "Can you check the weather in Seattle and San Francisco?",
]

for i, query in enumerate(test_queries, 1):
    print("─" * 70)
    print(f"Test {i}: {query}")
    print("─" * 70)
    
    try:
        # Invoke the agent
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": f"test-{i}"}}
        )
        
        # Extract final response
        final_message = result["messages"][-1]
        print(f"\n Response: {final_message.content}\n")
        
    except Exception as e:
        print(f"\n Error: {str(e)}\n")

print("=" * 70)
print("TESTING COMPLETE!")
print("=" * 70)

if os.getenv("LANGSMITH_API_KEY"):
    print(f"\n View traces in LangSmith:")
    print(f"   https://smith.langchain.com/")
    print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
    print("\nIn LangSmith, you should see:")
    print("   • Full conversation history for each test")
    print("   • Tool invocations (get_weather) for queries 1 and 3")
    print("   • LLM calls and responses")
    print("   • Token usage and latency metrics")
    print("   • Execution flow visualization")
else:
    print("\n To see traces next time:")
    print("   1. Sign up at https://smith.langchain.com")
    print("   2. Get your API key")
    print("   3. Set environment variables:")
    print("      export LANGSMITH_API_KEY='your-key-here'")
    print("      export LANGSMITH_TRACING=true")
    print("      export LANGSMITH_PROJECT='my-first-agent'")
    print("   4. Run this script again")

print("=" * 70)

print("\n\nKey Concepts:")
print("\n1. AUTOMATIC TRACING:")
print("   Once environment variables are set, ALL LangChain operations")
print("   are automatically traced - no code changes needed!")

print("\n2. TOOL CALLING VS LLM RESPONSES:")
print(f"   • Test 1 & 3: Agent USES the tool (weather-related)")
print(f"   • Test 2: Agent ANSWERS DIRECTLY (general knowledge)")

print("\n3. SIMPLIFIED AGENT CREATION:")
print("   • This uses create_agent() - the v1.0 standard")
print("   • Simple one-line agent creation")
print("   • Handles all workflow complexity automatically")
print("   • Built-in state management and tool calling")

print("\n4. WHAT YOU SEE IN LANGSMITH:")
print("   • Input/Output for each invocation")
print("   • Tool calls with arguments and results")
print("   • LLM reasoning and decision-making")
print("   • Token counts and costs")
print("   • Latency for each step")
print("   • Visual graph of execution flow")

print("\n5. WHY LANGSMITH MATTERS:")
print("   • Debug complex agent behaviors")
print("   • Understand why agents make certain decisions")
print("   • Track costs and performance")
print("   • Monitor production applications")
print("   • Identify bottlenecks and errors")

print("\n" + "=" * 70 + "\n")

