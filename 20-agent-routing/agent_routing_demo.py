from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

question = "Should I use LangChain or plain OpenAI API?"

response = llm.invoke(
    f"Answer this question as a software architect:\n{question}"
)

print(response.content)
