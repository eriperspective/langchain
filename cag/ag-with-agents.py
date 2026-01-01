"""
CAG Tutorial 5: Context + Agents
Combine structured context with agent reasoning.
"""
from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI

tools = [
    Tool(
        name="Greet",
        func=lambda name: f"Hello, {name}!",
        description="Greets a user by name"
    )
]

context = {"user": "Alice", "role": "admin"}
agent = initialize_agent(tools, OpenAI(), agent="zero-shot-react-description")

query = f"Greet the user {context['user']}."
print(agent.run(query))
