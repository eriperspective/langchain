from langchain.memory import ConversationBufferMemory

sessions = {}

def get_memory(session_id):
    if session_id not in sessions:
        sessions[session_id] = ConversationBufferMemory()
    return sessions[session_id]

mem1 = get_memory("user_1")
mem2 = get_memory("user_2")

mem1.save_context({"input": "Hi"}, {"output": "Hello user 1"})
mem2.save_context({"input": "Hi"}, {"output": "Hello user 2"})

print(mem1.buffer)
print(mem2.buffer)
