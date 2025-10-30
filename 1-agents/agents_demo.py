"""
LangChain Agents Demo
=====================
Learn how agents work and how to customize models, tools, and system prompts.

What we'll cover:
1. Static vs Dynamic Models
2. Tool creation and error handling
3. Static vs Dynamic System Prompts

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/oss/python/langchain/agents
Last Updated: October 30, 2025
"""

import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    dynamic_prompt,
    wrap_tool_call
)
from langchain_core.messages import ToolMessage
from typing import Callable, TypedDict

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set your OPENAI_API_KEY environment variable")
    exit(1)


print("="*70)
print("PART 1: STATIC MODEL (MOST COMMON APPROACH)")
print("="*70)

# ============================================================================
# STATIC MODEL
# ============================================================================
# Static models are set when creating the agent and don't change.
# This is the most common and simplest approach.

# Method 1: Simple string (LangChain figures out the provider)
print("\n--- Method 1: Simple string ---")

@tool
def search_recipe(dish_name: str) -> str:
    """
    Searches for a recipe for a specific dish.
    
    Args:
        dish_name: The name of the dish to search for.
        
    Returns:
        A recipe description.
    """
    # Mock recipe database
    recipes = {
        "pasta": "Classic pasta: Boil water, add pasta, cook 10 minutes, drain, add sauce.",
        "pancakes": "Pancakes: Mix flour, eggs, milk. Pour on hot griddle, flip when bubbles form.",
    }
    return recipes.get(dish_name.lower(), f"Recipe for {dish_name} not found in our database.")


agent_simple = create_agent(
    model="openai:gpt-4o-mini",  # Simple string format
    tools=[search_recipe],
    system_prompt="You are a helpful cooking assistant.",
    name="simple_recipe_agent"
)

response = agent_simple.invoke({
    "messages": [{"role": "user", "content": "How do I make pasta?"}]
})
print(f"Response: {response['messages'][-1].content}\n")


# Method 2: Model instance with custom parameters
print("--- Method 2: Custom model instance ---")

# Create a customized model instance for more control
custom_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,     # Low temperature for more predictable, factual responses
    max_tokens=1500,
    timeout=30
)

agent_custom = create_agent(
    model=custom_model,
    tools=[search_recipe],
    system_prompt="You are a precise cooking instructor. Provide clear, step-by-step instructions.",
    name="custom_recipe_agent"
)

response = agent_custom.invoke({
    "messages": [{"role": "user", "content": "How do I make pancakes?"}]
})
print(f"Response: {response['messages'][-1].content}\n")


print("="*70)
print("PART 2: DYNAMIC MODEL (ADVANCED)")
print("="*70)

# ============================================================================
# DYNAMIC MODEL
# ============================================================================
# Dynamic models can switch based on query complexity.
# This optimizes cost (use cheap model) and performance (use powerful model).

# Define two models with different capabilities
simple_model = ChatOpenAI(model="gpt-4o-mini")      # Fast and cheap
expert_model = ChatOpenAI(model="gpt-4o")            # Powerful but expensive

# Create middleware to route between models based on query complexity
@wrap_model_call
def dynamic_model_router(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """
    Selects a model based on query complexity using multiple heuristics.
    
    This checks for:
    - Complex recipe names (e.g., "beef wellington", "souffle")
    - Many ingredients (comma-separated list)
    - Advanced cooking techniques (e.g., "sous vide", "flambe")
    - Long queries (more words = more complex)
    """
    user_query = request.state["messages"][-1].content.lower()
    
    # Heuristic 1: Check for complex recipe names
    complex_recipes = [
        "beef wellington", "souffle", "soufflé", "coq au vin",
        "bouillabaisse", "consomme", "croissant", "macarons"
    ]
    has_complex_recipe = any(recipe in user_query for recipe in complex_recipes)
    
    # Heuristic 2: Count ingredients (comma-separated list)
    has_many_ingredients = user_query.count(',') > 4
    
    # Heuristic 3: Check for advanced cooking techniques
    advanced_techniques = [
        "sous vide", "flambe", "confit", "molecular",
        "emulsify", "temper", "clarify"
    ]
    has_advanced_technique = any(technique in user_query for technique in advanced_techniques)
    
    # Heuristic 4: Query length (longer queries often indicate complexity)
    is_long_query = len(user_query.split()) > 15
    
    # Use expert model if ANY complexity indicator is present
    if has_complex_recipe or has_many_ingredients or has_advanced_technique or is_long_query:
        print("  [Router] Query is complex, using expert model (GPT-4o)")
        request = request.override(model=expert_model)
    else:
        print("  [Router] Query is simple, using fast model (GPT-4o-mini)")
        request = request.override(model=simple_model)
    
    return handler(request)


# Create agent with dynamic model routing
dynamic_agent = create_agent(
    model=simple_model,  # Default/starting model
    tools=[search_recipe],
    middleware=[dynamic_model_router],
    system_prompt="You are a knowledgeable cooking assistant.",
    name="dynamic_recipe_agent"
)

# Test with simple query (should use simple model)
print("\n--- Test 1: Simple query ---")
print("Query: How do I make pasta?")
response = dynamic_agent.invoke({
    "messages": [{"role": "user", "content": "How do I make pasta?"}]
})
print(f"Response: {response['messages'][-1].content[:100]}...\n")

# Test with complex query (should use expert model)
print("--- Test 2: Complex query ---")
print("Query: How do I make beef wellington with a perfect pastry crust?")
response = dynamic_agent.invoke({
    "messages": [{"role": "user", "content": "How do I make beef wellington with a perfect pastry crust?"}]
})
print(f"Response: {response['messages'][-1].content[:100]}...\n")


print("="*70)
print("PART 3: TOOLS AND ERROR HANDLING")
print("="*70)

# ============================================================================
# TOOL CREATION AND ERROR HANDLING
# ============================================================================

# Define a tool that might fail
@tool
def check_pantry(ingredient: str) -> str:
    """Checks if an ingredient is available in the pantry."""
    pantry = ["flour", "sugar", "eggs", "milk", "butter"]
    
    # Simulate potential error
    if ingredient.lower() == "error":
        raise Exception("Pantry database connection failed!")
    
    if ingredient.lower() in pantry:
        return f"Yes, you have {ingredient} in the pantry."
    return f"Sorry, you don't have {ingredient}."


# Create error handler middleware
@wrap_tool_call
def custom_tool_error_handler(request, handler: Callable) -> ToolMessage:
    """Handles errors during tool execution with a custom message."""
    try:
        return handler(request)
    except Exception as e:
        # Return a clear error message to the model so it can try something else
        return ToolMessage(
            content=f"Tool error occurred: {str(e)}. Please try a different approach.",
            tool_call_id=request.tool_call["id"]
        )


# Create agent with error handling
error_handling_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_recipe, check_pantry],
    middleware=[custom_tool_error_handler],
    system_prompt="You are a helpful cooking assistant. If a tool fails, explain what happened and suggest alternatives.",
    name="error_handling_agent"
)

# Test normal tool use
print("\n--- Test 1: Normal tool use ---")
response = error_handling_agent.invoke({
    "messages": [{"role": "user", "content": "Do I have flour in my pantry?"}]
})
print(f"Response: {response['messages'][-1].content}\n")

# Test error handling
print("--- Test 2: Tool error handling ---")
response = error_handling_agent.invoke({
    "messages": [{"role": "user", "content": "Check for 'error' in my pantry."}]
})
print(f"Response: {response['messages'][-1].content}\n")


print("="*70)
print("PART 4: DYNAMIC SYSTEM PROMPTS")
print("="*70)

# ============================================================================
# DYNAMIC SYSTEM PROMPTS
# ============================================================================
# System prompts can change based on runtime context (like user preferences)

class Context(TypedDict):
    """Context schema for dietary preferences."""
    dietary_preference: str


@dynamic_prompt
def dietary_prompt_builder(request: ModelRequest) -> str:
    """Generate a system prompt based on the user's dietary preference."""
    preference = request.runtime.context.get("dietary_preference", "none")
    
    base_prompt = "You are a helpful culinary assistant."
    
    if preference == "vegan":
        return f"{base_prompt} All recipes must be 100% plant-based. No animal products."
    elif preference == "gluten-free":
        return f"{base_prompt} All recipes must be gluten-free. No wheat, barley, or rye."
    elif preference == "keto":
        return f"{base_prompt} All recipes must be keto-friendly. Low carb, high fat."
    
    return base_prompt


# Create agent with dynamic prompts
dietary_agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_recipe],
    middleware=[dietary_prompt_builder],
    context_schema=Context,
    name="dietary_aware_agent"
)

# Test with vegan preference
print("\n--- Test 1: Vegan preference ---")
response = dietary_agent.invoke(
    {"messages": [{"role": "user", "content": "Suggest a breakfast recipe."}]},
    context={"dietary_preference": "vegan"}
)
print(f"Response: {response['messages'][-1].content}\n")

# Test with gluten-free preference
print("--- Test 2: Gluten-free preference ---")
response = dietary_agent.invoke(
    {"messages": [{"role": "user", "content": "Suggest a breakfast recipe."}]},
    context={"dietary_preference": "gluten-free"}
)
print(f"Response: {response['messages'][-1].content}\n")


print("="*70)
print("DEMO COMPLETE!")
print("="*70)
print("\nKey takeaways:")
print("1. Static models (simple string or instance) work for most cases")
print("2. Dynamic models optimize cost vs performance based on complexity")
print("3. Middleware with @wrap_model_call controls model selection")
print("4. Tools are Python functions that extend agent capabilities")
print("5. @wrap_tool_call middleware handles tool errors gracefully")
print("6. Dynamic prompts with @dynamic_prompt adapt to user context")
print("7. Always name your agents for better debugging and tracing")
print("="*70 + "\n")

