from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model="gpt-4o-mini")
memory = ConversationBufferMemory(return_messages=True)

def assistant(user_input):
    memory.chat_memory.add_user_message(user_input)
    response = llm.invoke(memory.chat_memory.messages)
    memory.chat_memory.add_ai_message(response.content)
    return response.content

print(assistant("What is agentic AI?"))
print(assistant("Why is it useful in production systems?"))
