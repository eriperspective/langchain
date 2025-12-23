from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

plan = llm.invoke(
    "Create a 3-step plan to learn agentic AI."
)

execution = llm.invoke(
    f"Execute step 1 from this plan:\n{plan.content}"
)

print(execution.content)
