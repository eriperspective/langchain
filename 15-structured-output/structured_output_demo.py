from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class AIResponse(BaseModel):
    summary: str
    use_case: str

llm = ChatOpenAI(model="gpt-4o-mini").with_structured_output(AIResponse)

result = llm.invoke(
    "Explain LangChain and give one real-world use case."
)

print(result)
