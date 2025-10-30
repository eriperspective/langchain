# Agentic RAG - Let the Agent Decide When to Retrieve

**LangChain Version:** v1.0+  
**Documentation:** https://docs.langchain.com/oss/python/langchain/retrieval#agentic-rag  
**Last Updated:** October 23, 2025

Build a RAG system where the agent decides when to retrieve information.

---

## The Key Difference

| Feature      | 2-Step RAG              | Agentic RAG                   |
|--------------|-------------------------|-------------------------------|
| Retrieval    | Always happens          | Agent decides when            |
| Flexibility  | Fixed workflow          | Dynamic reasoning             |
| Best for     | FAQs, simple docs       | Research, complex tasks       |

**Simple analogy:** 
- **2-Step RAG** = Always looking in the manual before answering
- **Agentic RAG** = Only looking in the manual when you need to

---

## Prerequisites

You need:
- Vector database at `./chroma_db` (from [Document Loaders](./7-document-loaders-and-text-splitter.md))
- OpenAI API key

---

## Basic Example

```python
"""
Agentic RAG - Agent Decides When to Search
"""
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Create search tool
@tool
def search_documents(query: str) -> str:
    """Search documents for specific information."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in docs)

# THE KEY DIFFERENCE: System prompt gives agent autonomy
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_documents],
    system_prompt=(
        "Use search_documents when you need specific document information. "
        "If you can answer from general knowledge, do so."
    ),
    name="agentic_rag"
)

# Test 1: Needs document (will search)
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is this document about?"}]
})
print(response["messages"][-1].content)

# Test 2: General knowledge (probably won't search)
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is 2+2?"}]
})
print(response["messages"][-1].content)
```

---

## Compare the System Prompts

### 2-Step RAG (Always Retrieves)
```python
system_prompt = "Use search_documents to answer questions."
# Agent will ALWAYS search
```

### Agentic RAG (Decides Dynamically)
```python
system_prompt = (
    "Use search_documents when you need specific document information. "
    "Answer from general knowledge when possible."
)
# Agent will ONLY search when needed
```

That's the whole difference!

---

## Why Use Agentic RAG?

### When Multiple Tools are Available

```python
@tool
def search_local_docs(query: str) -> str:
    """Search uploaded documents."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def calculate(expression: str) -> str:
    """Do math calculations."""
    return str(eval(expression))

# Agent picks the right tool for each question
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_local_docs, calculate],
    system_prompt="Use the appropriate tool based on the question.",
    name="multi_tool_agent"
)

# "What's in my document?" -> uses search_local_docs
# "What's 15 * 37?" -> uses calculate
# "What's the capital of France?" -> uses neither
```

**The agent reasons about which tool to use (or none at all).**

---

## When to Use Each Approach

### Use 2-Step RAG:
- Every question needs document context
- Building simple doc chatbots
- Want predictable, fast responses

### Use Agentic RAG:
- Questions vary (general + document-specific)
- Need multiple tools (docs + web + APIs + calculations)
- Agent should reason about what it needs

---

## Best Practices

```python
# Good: Clear tool description
@tool
def search_documents(query: str) -> str:
    """Search documents for specific information about [YOUR TOPIC]."""
    pass

# Good: Give agent autonomy
system_prompt = "Use search_documents when you need document details."

# Bad: Forcing the agent (defeats the purpose)
system_prompt = "You MUST ALWAYS use search_documents."

# Bad: Vague description
@tool
def tool1(x: str) -> str:
    """Does stuff."""  # Agent won't know when to use this
    pass
```

---

## Quick Reference

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Tool
@tool
def search_documents(query: str) -> str:
    """Search documents for specific information."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in docs)

# Agent (with autonomy)
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_documents],
    system_prompt="Use search_documents when you need document information.",
    name="agentic_rag"
)

# Use
response = agent.invoke({"messages": [{"role": "user", "content": "query"}]})
print(response["messages"][-1].content)
```

---

## Key Takeaways

- **Agentic RAG** = Agent decides when to retrieve (not always)
- **System prompt** is key: give guidance, not commands
- **Perfect for** multiple tools or mixed question types
- **The agent reasons** about what information it needs

---

## Resources

- [LangChain Agentic RAG Docs](https://docs.langchain.com/oss/python/langchain/retrieval#agentic-rag)
- [Previous: 2-Step RAG](./9-two_step_rag.md)
- [Previous: Retrieval](./8-retrieval.md)
