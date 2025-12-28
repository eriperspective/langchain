from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=50)

response = llm.invoke("Explain LangChain briefly.")

print(response.content)
print("Token usage limited for cost control.")
