"""
Multi-Agent System Demo
=======================
Learn how to build systems with multiple specialized agents working together.

Pattern: Tool Calling (Supervisor)
- A supervisor agent calls sub-agents as tools
- Sub-agents perform specific tasks and return results
- Centralized workflow control

Prerequisites:
- OpenAI API key

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/oss/python/langchain/multi-agent
Last Updated: October 30, 2025
"""

import os
from langchain.tools import tool
from langchain.agents import create_agent

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set your OPENAI_API_KEY environment variable")
    exit(1)

print("="*70)
print("MULTI-AGENT SYSTEM DEMO")
print("="*70)

# ============================================================================
# STEP 1: CREATE SUB-AGENTS
# ============================================================================
# Each sub-agent is a specialist in one domain

print("\n[1/3] Creating specialized sub-agents...")

# Calendar specialist
calendar_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],  # In production, give it calendar API tools
    system_prompt=(
        "You are a calendar specialist. "
        "Handle scheduling tasks like creating events, checking availability, and managing appointments. "
        "IMPORTANT: Include ALL details in your final response. "
        "The supervisor only sees your final message."
    ),
    name="calendar_agent"
)

# Email specialist
email_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],  # In production, give it email API tools
    system_prompt=(
        "You are an email specialist. "
        "Handle email-related tasks like sending emails, checking inbox, and managing drafts. "
        "IMPORTANT: Include ALL details in your final response. "
        "The supervisor only sees your final message."
    ),
    name="email_agent"
)

# Research specialist
research_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[],  # In production, give it web search tools
    system_prompt=(
        "You are a research specialist. "
        "Find and synthesize information from various sources. "
        "IMPORTANT: Include ALL findings in your final response. "
        "The supervisor only sees your final message."
    ),
    name="research_agent"
)

print(" Created 3 specialized agents:")
print("   • calendar_agent: Scheduling and appointments")
print("   • email_agent: Email management")
print("   • research_agent: Information gathering")

# ============================================================================
# STEP 2: WRAP SUB-AGENTS AS TOOLS
# ============================================================================
# The supervisor calls sub-agents through tools

print("\n[2/3] Wrapping sub-agents as tools...")

@tool
def schedule_event(request: str) -> str:
    """
    Schedule calendar events using natural language.
    Use for:
    - Creating meetings
    - Checking availability
    - Managing appointments
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content


@tool
def manage_email(request: str) -> str:
    """
    Handle email-related tasks.
    Use for:
    - Sending emails
    - Checking inbox
    - Managing drafts
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content


@tool
def research_topic(request: str) -> str:
    """
    Research and gather information on topics.
    Use for:
    - Finding facts
    - Gathering data
    - Synthesizing information
    """
    result = research_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content


print(" Sub-agents wrapped as tools")

# ============================================================================
# STEP 3: CREATE SUPERVISOR AGENT
# ============================================================================
# The supervisor coordinates all sub-agents

print("\n[3/3] Creating supervisor agent...")

supervisor = create_agent(
    model="openai:gpt-4o-mini",
    tools=[schedule_event, manage_email, research_topic],
    system_prompt=(
        "You are a helpful AI coordinator. "
        "You have access to specialized agents for calendar, email, and research tasks. "
        "Choose the appropriate agent based on the user's request. "
        "You may need to use multiple agents to complete complex tasks."
    ),
    name="supervisor"
)

print(" Supervisor agent ready\n")

# ============================================================================
# TEST THE MULTI-AGENT SYSTEM
# ============================================================================

print("="*70)
print("TESTING MULTI-AGENT SYSTEM")
print("="*70)

test_scenarios = [
    (
        "Schedule a team meeting for tomorrow at 2pm",
        "Should use: calendar_agent"
    ),
    (
        "Send an email to the team about the meeting",
        "Should use: email_agent"
    ),
    (
        "What are the benefits of agile methodology?",
        "Should use: research_agent"
    ),
    (
        "Schedule a meeting and send a calendar invite via email",
        "Should use: calendar_agent AND email_agent"
    ),
]

for i, (query, expected) in enumerate(test_scenarios, 1):
    print(f"\n{'─'*70}")
    print(f"Test {i}: {query}")
    print(f"Expected: {expected}")
    print("─"*70)
    
    response = supervisor.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    print(f"\nSupervisor: {response['messages'][-1].content}\n")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

print("\n" + "="*70)
print("INTERACTIVE MODE")
print("="*70)
print("Try tasks involving calendar, email, or research!")
print("Type 'quit' to exit.\n")

while True:
    query = input("You: ").strip()
    
    if query.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!\n")
        break
    
    if not query:
        continue
    
    response = supervisor.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    print(f"\nSupervisor: {response['messages'][-1].content}\n")

print("\n" + "="*70)
print("KEY CONCEPTS")
print("="*70)
print("""
MULTI-AGENT ARCHITECTURE:

1. SUB-AGENTS:
   • Each handles one specific domain
   • Have focused tools and expertise
   • Return results to supervisor

2. SUPERVISOR:
   • Coordinates all sub-agents
   • Decides which agent(s) to use
   • Combines results for final answer

3. TOOL CALLING PATTERN:
   • Sub-agents wrapped as tools
   • Centralized workflow control
   • Clear separation of concerns

4. CRITICAL SUCCESS FACTORS:
   • Clear domain boundaries (no overlap)
   • Specific tool descriptions
   • Sub-agent prompts emphasize final output
   • Test sub-agents independently first

5. WHEN TO USE MULTI-AGENT:
   • Multiple distinct domains
   • Each domain has complex logic/tools
   • Need centralized coordination
   • Sub-agents don't need to talk to users directly

6. WHEN NOT TO USE:
   • Simple single-domain tasks
   • Few tools (< 5)
   • No clear domain separation
""")
print("="*70 + "\n")

