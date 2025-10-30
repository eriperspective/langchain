# Quickstart Guide to LangChain v1.0

First, you'll need to install the `langchain`and `langchain-openai` package and set your OpenAI API key as an environment variable.

```bash
pip install langchain langchain-openai
# IF NOT ALREADY IN ENVIRONMENT Then, set your API key in your terminal
export OPENAI_API_KEY="your-api-key"
```

-----

### **Part 1: Your First, Simple Agent (Flights Edition)**

We'll start with a basic agent that can look up the status of a flight.

```python
from langchain.agents import create_agent

# This is our tool—a simple function to get a flight's status.
# @tool
def get_flight_status(flight_number: str) -> str:
    """Gets the status for a given flight number."""
    # This is a mock response for demonstration
    return f"Flight {flight_number} is on time."

# Now, we create the agent using an OpenAI model.
agent = create_agent(
    model="openai:gpt-4o",  # Using OpenAI's GPT-4o model
    tools=[get_flight_status],
    system_prompt="You are a helpful flight assistant.",
)

# Let's run it!
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the status of flight AA123?"}]}
)

print(response)
```

This agent now understands how to use the `get_flight_status` tool when you ask about a specific flight.

-----

### **Part 2: Building a "Real-World" Flight Agent** 

Now, let's build a more advanced agent that can handle more complex queries and remember your conversation.

#### **Step 1: Write a Detailed System Prompt**

A good prompt is key. We'll instruct our agent on its personality and how to use its tools effectively.

```python
SYSTEM_PROMPT = """Your goal is to provide passengers with accurate and helpful flight information. You should be professional, but with a friendly and slightly witty tone.

You have access to two tools:
- get_flight_details: Use this to get the status, gate, and departure time for a specific flight number.
- get_user_home_airport: If a user asks about flights from 'my airport' or 'home', use this to find their registered home airport.

Always confirm the flight number before providing details. If a user asks a general question, use their home airport to provide relevant examples."""
```

#### **Step 2: Create Smart Tools with Context**

Here, we'll create a tool that can access runtime context (like a `user_id`) to provide personalized information.

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

# A tool for specific flight details
@tool
def get_flight_details(flight_number: str) -> str:
    """Gets flight details like status, gate, and time for a given flight number."""
    # Mock data for the example
    return f"Flight {flight_number} is on time, departing from Gate B12 at 8:45 PM."

# Define the data structure for our runtime context
@dataclass
class Context:
    """Custom runtime context schema for user info."""
    user_id: str

# A tool that uses the runtime context to find the user's home airport
@tool
def get_user_home_airport(runtime: ToolRuntime[Context]) -> str:
    """Retrieves the user's home airport based on their user ID."""
    user_id = runtime.context.user_id
    # In a real app, you'd look this up in a database
    return "JFK" if user_id == "user_abc" else "SFO"
```

#### **Step 3: Configure the Language Model**

We'll select an OpenAI model and set its parameters.

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai:gpt-4o",
    temperature=0.3, # Leans toward more factual responses
    timeout=10,
)
```

#### **Step 4: Define a Structured Output (Optional)**

A structured response format is perfect for flight data, ensuring we always get the information we need in a clean, predictable way.

```python
from dataclasses import dataclass

@dataclass
class ResponseFormat:
    """Response schema for the flight agent."""
    friendly_response: str
    flight_details: str | None = None
```

#### **Step 5: Add Conversational Memory**

We'll use the same `InMemorySaver` to allow our agent to remember the conversation's history.

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
```

#### **Step 6: Assemble and Run Your Agent**

Finally, we combine all our components and run the agent.

```python
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_home_airport, get_flight_details],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

# Each conversation needs a unique thread_id
config = {"configurable": {"thread_id": "flight_convo_1"}}

# First interaction: User asks a general question
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the status of my flight home today?"}]},
    config=config,
    context=Context(user_id="user_abc") # Provide the runtime context
)
print(response['structured_response'])

# Second interaction: User provides the flight number
response = agent.invoke(
    {"messages": [{"role": "user", "content": "It's flight UA456."}]},
    config=config,
    context=Context(user_id="user_abc")
)
print(response['structured_response'])
```