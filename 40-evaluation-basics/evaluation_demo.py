from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

expected = "LangChain helps build LLM applications."
actual = llm.invoke("What is LangChain?").content

print("Pass:", expected.lower() in actual.lower())

