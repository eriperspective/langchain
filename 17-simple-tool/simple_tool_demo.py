from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b

llm = ChatOpenAI(model="gpt-4o-mini")

result = add_numbers.invoke({"a": 5, "b": 7})

print(result)
