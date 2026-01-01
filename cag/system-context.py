"""
CAG: System Context
Inject system-level instructions or constraints for reasoning.
"""
from langchain.prompts import PromptTemplate

system_prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer this: {question}"
)

print(system_prompt.format(question="What is LangChain?"))
