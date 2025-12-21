from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

question = "How should I learn LangChain?"

thought = llm.invoke(f"Think step by step:\n{question}")
action = llm.invoke(f"Give one clear action:\n{thought.content}")

print(action.content)
