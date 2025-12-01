from typing import TypedDict

class AgentState(TypedDict):
    user_input: str
    response: str

state: AgentState = {
    "user_input": "What is agentic AI?",
    "response": "Agentic AI can make decisions and take actions."
}

print(state)
