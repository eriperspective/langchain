"""
LangChain Tools Demo
====================
Learn how to create tools that extend agent capabilities with real-world actions.

What we'll cover:
1. Basic tool definition
2. Custom tool properties (name, description)
3. Advanced schema definition
4. Accessing runtime context (State, Context, Store)

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/oss/python/langchain/tools
Last Updated: October 30, 2025
"""

import os
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing import Literal
from dataclasses import dataclass

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set your OPENAI_API_KEY environment variable")
    exit(1)


print("="*70)
print("PART 1: BASIC TOOL DEFINITION")
print("="*70)

# ============================================================================
# BASIC TOOL CREATION
# ============================================================================
# Tools are Python functions decorated with @tool
# The docstring is CRITICAL - the agent uses it to decide when to call the tool

@tool
def check_server_status(server_ip: str) -> str:
    """Checks if a server is online and returns its status.

    Args:
        server_ip: The IP address of the server to check.
    """
    # Mock server status check - in real use, you'd ping the server
    mock_servers = {
        "192.168.1.1": "ONLINE",
        "192.168.1.2": "OFFLINE",
        "192.168.1.3": "MAINTENANCE"
    }
    
    status = mock_servers.get(server_ip, "UNKNOWN")
    return f"Server at {server_ip} is {status}."


# Create agent with the tool
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[check_server_status],
    system_prompt="You are an IT support agent. Use tools to check server status when asked.",
    name="it_support_agent"
)

print("\n--- Test: Check server status ---")
response = agent.invoke({
    "messages": [{"role": "user", "content": "Is server 192.168.1.1 online?"}]
})
print(f"Response: {response['messages'][-1].content}\n")


print("="*70)
print("PART 2: CUSTOM TOOL PROPERTIES")
print("="*70)

# ============================================================================
# CUSTOMIZING TOOL NAME AND DESCRIPTION
# ============================================================================
# You can override the default name and provide a more helpful description

@tool(
    name="reboot_system_service",
    description="Restarts a specific service on a server. Use this as a first step to resolve common service issues."
)
def reboot_service(service_name: str, server_ip: str) -> str:
    """Reboots a given service on a specific server."""
    # Mock reboot operation
    return f"Successfully initiated reboot for '{service_name}' on server {server_ip}. Service should be back online in 30 seconds."


print(f"\nTool name: {reboot_service.name}")
print(f"Tool description: {reboot_service.description}\n")

# Test with agent
support_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[check_server_status, reboot_service],
    system_prompt="You are an IT support agent. Check server status first, then suggest rebooting services if needed.",
    name="advanced_it_agent"
)

print("--- Test: Multi-tool scenario ---")
response = support_agent.invoke({
    "messages": [{"role": "user", "content": "Server 192.168.1.2 seems down. Can you fix it?"}]
})
print(f"Response: {response['messages'][-1].content}\n")


print("="*70)
print("PART 3: ADVANCED SCHEMA DEFINITION")
print("="*70)

# ============================================================================
# USING PYDANTIC FOR COMPLEX INPUT SCHEMAS
# ============================================================================
# For tools with many parameters or complex validation, use Pydantic

class NewAccountInput(BaseModel):
    """Input schema for creating a new user account."""
    username: str = Field(description="The desired username for the new account.")
    department: Literal["Engineering", "Sales", "HR"] = Field(
        description="The department the user belongs to."
    )
    access_level: int = Field(
        default=1,
        description="The access level, from 1 (basic) to 3 (admin)."
    )


@tool(args_schema=NewAccountInput)
def create_user_account(username: str, department: str, access_level: int = 1) -> str:
    """Creates a new user account in the system."""
    return f"Account created:\n  Username: {username}\n  Department: {department}\n  Access Level: {access_level}"


# Test with agent
admin_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[create_user_account],
    system_prompt="You are an IT admin agent. Create user accounts based on requests.",
    name="admin_agent"
)

print("\n--- Test: Create user account ---")
response = admin_agent.invoke({
    "messages": [{"role": "user", "content": "Create an account for John in Engineering with admin access."}]
})
print(f"Response: {response['messages'][-1].content}\n")


print("="*70)
print("PART 4: ACCESSING RUNTIME CONTEXT")
print("="*70)

# ============================================================================
# TOOL RUNTIME - ACCESSING STATE, CONTEXT, AND STORE
# ============================================================================
# ToolRuntime gives tools access to conversation state and user context

# Define context schema
@dataclass
class HelpDeskContext:
    """Context for help desk operations."""
    user_id: str


# Mock user permissions database
USER_PERMISSIONS = {
    "user_123": "Admin",
    "user_456": "Standard",
    "user_789": "Guest"
}


# Tool that accesses State (conversation history)
@tool
def get_ticket_summary(runtime: ToolRuntime) -> str:
    """Summarizes the current IT support ticket conversation."""
    messages = runtime.state.get("messages", [])
    
    # Count user messages
    user_requests = [
        m for m in messages 
        if hasattr(m, 'type') and m.type == "human"
    ]
    
    return f"The user has made {len(user_requests)} requests in this ticket so far."


# Tool that accesses Context (immutable user data)
@tool
def check_user_permissions(runtime: ToolRuntime[HelpDeskContext]) -> str:
    """Checks the permission level of the current user."""
    user_id = runtime.context.user_id
    permissions = USER_PERMISSIONS.get(user_id, "Unknown")
    return f"User {user_id} has '{permissions}' permissions."


# Tool that accesses Store (long-term memory - simulated here)
@tool
def get_user_contact_info(user_id: str, runtime: ToolRuntime) -> str:
    """Looks up a user's contact info from the persistent store."""
    # Mock store lookup
    # In real use: user_info = runtime.store.get(("user_profiles",), user_id)
    
    mock_contacts = {
        "user_123": "john.doe@company.com, ext. 5551",
        "user_456": "jane.smith@company.com, ext. 5552",
    }
    
    contact = mock_contacts.get(user_id, "Contact info not found")
    return f"Contact info for {user_id}: {contact}"


# Tool that writes to State
@tool
def escalate_ticket(reason: str) -> Command:
    """Escalates the ticket to senior support."""
    # Use Command to update the agent's state
    return Command(
        update={"escalated": True, "escalation_reason": reason},
        result=f"Ticket escalated to senior support. Reason: {reason}"
    )


# Create context-aware agent
context_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[
        get_ticket_summary,
        check_user_permissions,
        get_user_contact_info,
        escalate_ticket
    ],
    context_schema=HelpDeskContext,
    system_prompt="You are a help desk agent. Use tools to check permissions and escalate when needed.",
    name="context_aware_agent"
)

print("\n--- Test: Check user permissions ---")
response = context_agent.invoke(
    {"messages": [{"role": "user", "content": "What permissions do I have?"}]},
    context=HelpDeskContext(user_id="user_123")
)
print(f"Response: {response['messages'][-1].content}\n")


print("\n--- Test: Get contact info ---")
response = context_agent.invoke(
    {"messages": [{"role": "user", "content": "What's the contact info for user_456?"}]},
    context=HelpDeskContext(user_id="user_123")
)
print(f"Response: {response['messages'][-1].content}\n")


print("="*70)
print("DEMO COMPLETE!")
print("="*70)
print("\nKey takeaways:")
print("1. Tools are Python functions with the @tool decorator")
print("2. Docstrings are critical - agents use them to decide when to call tools")
print("3. Type hints are required for defining the tool's schema")
print("4. Custom names and descriptions improve agent tool selection")
print("5. Pydantic models provide complex input validation")
print("6. ToolRuntime gives access to state, context, and store")
print("7. runtime.state - current conversation state")
print("8. runtime.context - immutable user/request data")
print("9. runtime.store - persistent long-term memory")
print("10. Command updates agent state from within tools")
print("="*70 + "\n")

