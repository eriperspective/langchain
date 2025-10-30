# Understanding LangChain Agents

[LangChain Docs - Agents](https://docs.langchain.com/oss/python/langchain/agents)

An **agent** is like a smart assistant that uses a language model (its "brain") to reason through a task. What makes it special is its ability to use **tools**—like searching the web or checking a database—to gather information and take action. It works in a loop, thinking about a problem, using a tool, observing the result, and then thinking again until it reaches a final answer.

The `create_agent` function is the modern way to build these agents in LangChain. Under the hood, it constructs a "graph," which is like a blueprint that dictates how the agent moves from one step to the next.

-----

## Core Component 1: The Model

The **model** is the reasoning engine of your agent. You can set it up in two main ways: statically (fixed) or dynamically (changes based on the situation).

### Static Model

This is the most common approach. You choose a model when you create the agent, and it stays the same.

You can specify the model with a simple string. LangChain will figure out the provider automatically.

```python
from langchain.agents import create_agent

# 'tools' would be a list of functions defined elsewhere
# This creates an agent powered by OpenAI's gpt-4o
agent = create_agent(
    "openai:gpt-4o",
    tools=tools 
)
```

For more control, you can create a model instance directly. This lets you set specific parameters like `temperature` (for creativity) or `timeout`.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# Create a customized model instance
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2, # Low temperature for more predictable recipe steps
    max_tokens=1500,
    timeout=30
)

# Pass the instance to the agent
agent = create_agent(model, tools=tools)
```

### Dynamic Model

A dynamic setup allows the agent to switch models on the fly. This is great for optimizing cost and performance—for example, using a simpler, cheaper model for easy questions and a more powerful one for complex tasks.

You can achieve this with **middleware**, which is like a function that intercepts the agent's process. The `@wrap_model_call` decorator lets you create a function to swap the model based on the conversation's state.

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

# Define two different models
simple_model = ChatOpenAI(model="gpt-4o-mini")
expert_chef_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_router(request: ModelRequest, handler) -> ModelResponse:
    """Selects a model based on query complexity using multiple heuristics."""
    user_query = request.state["messages"][-1].content.lower()
    
    # Heuristic 1: Check for complex recipe names
    complex_recipes = ["beef wellington", "soufflé", "soufflé", "coq au vin", 
                       "bouillabaisse", "consommé", "croissant", "macarons"]
    has_complex_recipe = any(recipe in user_query for recipe in complex_recipes)
    
    # Heuristic 2: Count ingredients (comma-separated list)
    has_many_ingredients = user_query.count(',') > 4
    
    # Heuristic 3: Check for advanced cooking techniques
    advanced_techniques = ["sous vide", "flambé", "confit", "molecular", 
                          "emulsify", "temper", "clarify"]
    has_advanced_technique = any(technique in user_query for technique in advanced_techniques)
    
    # Heuristic 4: Query length (longer queries often indicate complexity)
    is_long_query = len(user_query.split()) > 15
    
    # Use expert model if ANY complexity indicator is present
    if has_complex_recipe or has_many_ingredients or has_advanced_technique or is_long_query:
        request.model = expert_chef_model
    else:
        request.model = simple_model
        
    return handler(request)

agent = create_agent(
    model=simple_model, # The default model to start with
    tools=tools,
    middleware=[dynamic_model_router]
)
```

-----

## Core Component 2: Tools

**Tools** are what give your agent superpowers. They are simply functions that the agent can decide to call to get information or perform an action.

### Defining Tools

You can turn any Python function into a tool using the `@tool` decorator. Just pass a list of these functions when creating your agent.

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool(name="search_recipe")
def find_recipe(dish_name: str) -> str:
    """
    Searches for a recipe for a specific dish.

    Params:
        dish_name (str): The name of the dish to search for.

    Returns:
        str: A message indicating that a recipe has been found for the specified dish.
    """
    # In a real app, this would query a database or API
    return f"Found a classic recipe for {dish_name}."

@tool
def check_pantry(ingredient: str) -> str:
    """Checks if an ingredient is available in the pantry."""
    pantry = ["flour", "sugar", "eggs"]
    if ingredient.lower() in pantry:
        return f"Yes, you have {ingredient} in the pantry."
    return f"Sorry, you don't have {ingredient}."

# Provide the tools to the agent
agent = create_agent(model, tools=[find_recipe, check_pantry])
```

### Handling Tool Errors

You can also create middleware to manage what happens when a tool fails. The `@wrap_tool_call` decorator can catch errors and return a custom, helpful message to the model so it can try something else.

```python
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

@wrap_tool_call
def custom_tool_error_handler(request, handler):
    """Handles errors during tool execution with a custom message."""
    try:
        return handler(request)
    except Exception as e:
        # Return a clear error message to the model
        return ToolMessage(
            content=f"There was an issue with the tool. Please check your query. Error: {str(e)}",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="openai:gpt-4o",
    tools=[find_recipe, check_pantry],
    middleware=[custom_tool_error_handler]
)
```

-----

## Core Component 3: The System Prompt

The **system prompt** is your instruction manual for the agent. It sets the agent's personality, goals, and rules of engagement.

### Static System Prompt

You can provide a simple string to guide the agent's behavior.

```python
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful culinary assistant. Always provide clear, step-by-step instructions for recipes."
)
```

### Dynamic System Prompt

For more tailored interactions, a dynamic prompt can change based on the user's needs. The `@dynamic_prompt` decorator lets you create middleware that builds a prompt on the fly using runtime context.

```python
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class Context(TypedDict):
    dietary_preference: str

@dynamic_prompt
def dietary_prompt_builder(request: ModelRequest) -> str:
    """Generate a system prompt based on the user's dietary preference."""
    preference = request.runtime.context.get("dietary_preference", "none")
    base_prompt = "You are a helpful culinary assistant."
    
    if preference == "vegan":
        return f"{base_prompt} All recipes must be 100% plant-based."
    elif preference == "gluten-free":
        return f"{base_prompt} All recipes must be gluten-free."
    
    return base_prompt

agent = create_agent(
    model="openai:gpt-4o",
    tools=[find_recipe],
    middleware=[dietary_prompt_builder],
    context_schema=Context
)
```

-----

## Invoking the Agent

To run your agent, you simply use the `invoke` method and pass it a message.

```python
# Invoke the dynamic prompt agent with a specific context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "I want to make lasagna."}]},
    context={"dietary_preference": "vegan"}
)

print(result)
```