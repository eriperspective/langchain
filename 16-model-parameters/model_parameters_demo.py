from langchain_openai import ChatOpenAI

creative_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.9
)

precise_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("Creative:")
print(creative_llm.invoke("Write a metaphor for AI.").content)

print("\nPrecise:")
print(precise_llm.invoke("Define artificial intelligence.").content)
