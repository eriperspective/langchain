from langchain_core.tools import tool

@tool
def divide(a: int, b: int) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

try:
    result = divide.invoke({"a": 10, "b": 0})
    print(result)
except Exception as e:
    print("Tool error:", e)
