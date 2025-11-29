from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

memory.save_context(
    {"input": "Hello"},
    {"output": "Hi there!"}
)

memory.save_context(
    {"input": "What is LangChain?"},
    {"output": "A framework for building LLM apps."}
)

print(memory.buffer)

