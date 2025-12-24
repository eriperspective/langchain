from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_definition(term: str) -> str:
    return f"{term} is a core concept in AI systems."

llm = ChatOpenAI(model="gpt-4o-mini")

result = get_definition.invoke({"term": "Agentic AI"})

print(result)
