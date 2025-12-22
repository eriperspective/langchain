from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

answer = llm.invoke("Explain LangChain briefly.")
reflection = llm.invoke(
    f"Critique this answer and improve it:\n{answer.content}"
)

print("Improved Answer:")
print(reflection.content)
