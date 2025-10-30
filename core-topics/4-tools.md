# Understanding Tools in LangChain

[LangChain Docs - Tools](https://docs.langchain.com/oss/python/langchain/tools)

While many AI applications are purely conversational, advanced use cases require models to interact with external systems like APIs, databases, or even a local file system. In LangChain, **Tools** are the components that allow an agent to perform these actions.

A tool is essentially a function that an agent can call. It provides a bridge between the model's reasoning capabilities and the outside world, extending its abilities far beyond simple text generation. By defining tools with clear inputs and outputs, you enable a language model to generate structured requests to interact with your systems, effectively giving it the ability to "act."

### Creating Tools

You can define tools in several ways, from simple decorated functions to those with complex, structured inputs.

#### Basic Tool Definition

The easiest way to create a tool is with the `@tool` decorator. LangChain automatically uses the function's name, type hints, and docstring to create a schema that the model can understand. The docstring is especially important, as it becomes the tool's description, which the model uses to decide when and how to use it.

```python
from langchain.tools import tool

@tool
def check_server_status(server_ip: str) -> str:
    """Checks if a server is online and returns its status.

    Args:
        server_ip: The IP address of the server to check.
    """
    # In a real application, this would ping the server.
    return f"Server at {server_ip} is ONLINE."
```

**Note:** Type hints are required because they define the input schema for the tool.

#### Customizing Tool Properties

You can override the default properties of a tool for better clarity and control.

  * **Custom Name**: By default, the tool's name is the function's name. You can provide a more descriptive name that is easier for the model to work with.
  * **Custom Description**: You can also provide a custom description, which is often more direct and helpful to the model than a multi-line docstring.

<!-- end list -->

```python
@tool(
    "reboot_system_service",
    description="Restarts a specific service on a server. Use this as a first step to resolve common issues."
)
def reboot_service(service_name: str, server_ip: str) -> str:
    """Reboots a given service on a specific server."""
    return f"Successfully initiated reboot for '{service_name}' on server {server_ip}."

print(reboot_service.name)
# Output: reboot_system_service
```

#### Advanced Schema Definition

For tools with complex or highly specific inputs, you can define the schema using a Pydantic model. This gives you fine-grained control over each argument, including descriptions and validation.

```python
from pydantic import BaseModel, Field
from typing import Literal

class NewAccountInput(BaseModel):
    """Input schema for creating a new user account."""
    username: str = Field(description="The desired username for the new account.")
    department: Literal["Engineering", "Sales", "HR"] = Field(description="The department the user belongs to.")
    access_level: int = Field(default=1, description="The access level, from 1 (basic) to 3 (admin).")

@tool(args_schema=NewAccountInput)
def create_user_account(username: str, department: str, access_level: int = 1) -> str:
    """Creates a new user account in the system."""
    return f"Account for {username} in {department} with access level {access_level} has been created."
```

### Accessing Context within Tools

Tools become truly powerful when they can access information about the current state of the conversation and the broader runtime environment. This allows them to perform context-aware actions, such as looking up information related to the specific user interacting with the agent.

#### ToolRuntime

LangChain provides a unified `ToolRuntime` parameter that can be added to any tool's function signature. This parameter is automatically injected by the system and is **not** visible to the model. It provides access to:

  * **State**: The current, mutable state of the agent's execution graph (e.g., conversation history).
  * **Context**: Immutable data for the current run (e.g., the user's ID).
  * **Store**: A connection to persistent, long-term memory.
  * **Stream Writer**: A function to stream real-time updates from the tool.

Here are examples of how to use it:

  * **Accessing State**
    A tool can inspect the agent's state, such as the list of messages, to understand the conversation's history.

    ```python
    from langchain.tools import ToolRuntime

    @tool
    def get_ticket_summary(runtime: ToolRuntime) -> str:
        """Summarizes the current IT support ticket conversation."""
        messages = runtime.state.get("messages", [])
        user_requests = [m.content for m in messages if m.role == "user"]
        return f"The user has made {len(user_requests)} requests in this ticket so far."
    ```

  * **Accessing Context**
    A tool can use immutable context, such as a user ID passed in when the agent is invoked, to perform personalized actions.

    ```python
    from dataclasses import dataclass

    USER_PERMISSIONS = {"user_123": "Admin", "user_456": "Standard"}

    @dataclass
    class HelpDeskContext:
        user_id: str

    @tool
    def check_user_permissions(runtime: ToolRuntime[HelpDeskContext]) -> str:
        """Checks the permission level of the current user."""
        user_id = runtime.context.user_id
        permissions = USER_PERMISSIONS.get(user_id, "Unknown")
        return f"User {user_id} has '{permissions}' permissions."
    ```

  * **Accessing Memory (Store)**
    The store provides access to long-term memory that persists across different conversations.

    ```python
    @tool
    def get_user_contact_info(user_id: str, runtime: ToolRuntime) -> str:
        """Looks up a user's contact info from the persistent store."""
        store = runtime.store
        # The key is a tuple identifying the data's location
        user_info = store.get(("user_profiles",), user_id)
        return str(user_info.value) if user_info else f"No contact info found for user {user_id}."
    ```

  * **Using the Stream Writer**
    For long-running tasks, a tool can use the stream writer to send real-time progress updates back to the user.

    ```python
    import time

    @tool
    def run_system_diagnostics(server_ip: str, runtime: ToolRuntime) -> str:
        """Runs a full diagnostic on the specified server."""
        writer = runtime.stream_writer
        
        writer(f"Starting diagnostics on {server_ip}...")
        time.sleep(2)
        writer("...Checking network connectivity.")
        time.sleep(2)
        writer("...Verifying disk integrity.")
        time.sleep(2)
        
        return f"Diagnostics complete for {server_ip}. All systems nominal."
    ```