from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
memory = ConversationSummaryMemory(llm=llm)

memory.save_context(
    {"input": "Tell me about LangChain."},
    {"output": "LangChain helps build AI apps."}
)

memory.save_context(
    {"input": "Why is it popular?"},
    {"output": "It simplifies agents, tools, and memory."}
)

print(memory.load_memory_variables({})["history"])
