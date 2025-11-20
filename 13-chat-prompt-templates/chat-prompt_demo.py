from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor."),
    ("human", "Explain {concept} in one paragraph.")
])

messages = prompt.format_messages(concept="agentic AI")

response = llm.invoke(messages)

print(response.content)

