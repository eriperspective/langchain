from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    return a - b

tools = [multiply, subtract]

print(multiply.invoke({"a": 4, "b": 3}))
print(subtract.invoke({"a": 10, "b": 6}))
