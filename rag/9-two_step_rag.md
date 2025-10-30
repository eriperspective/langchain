# 2-Step RAG (Retrieval-Augmented Generation)

**LangChain Version:** v1.0+  
**Documentation:** https://docs.langchain.com/oss/python/langchain/retrieval#2-step-rag  
**Last Updated:** October 23, 2025

Build a simple RAG system where retrieval always happens before generation.

---

## What is 2-Step RAG?

**2-Step RAG** = Fixed two-step process:

1. **Retrieve** relevant documents from your knowledge base
2. **Generate** answer using those documents as context

| Feature     | 2-Step RAG                            |
|-------------|---------------------------------------|
| Control     | High - predictable execution          |
| Latency     | Fast and predictable                  |
| Use Cases   | FAQs, documentation bots, simple Q&A  |

**When to use:** When you always need context from your knowledge base for every query.

---

## Prerequisites

Complete these first:
1. [Document Loaders](./7-document-loaders-and-text-splitter.md) - Build knowledge base
2. [Retrieval](./8-retrieval.md) - Learn querying

You need:
- Vector database at `./chroma_db`
- OpenAI API key

---

## Simple Implementation

```python
"""
Simple 2-Step RAG with Agent and Tool
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

# Define search tool
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG agent
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    system_prompt="Use search_knowledge_base to find information before answering.",
    name="rag_agent"
)

# Ask a question
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is this document about?"}]
})

print(response["messages"][-1].content)
```

That's it! The agent will automatically:
1. Search the knowledge base using your tool
2. Generate an answer using the retrieved context

---

## With Source Citations

```python
"""
2-Step RAG with Source Information
"""
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    docs = vectorstore.similarity_search(query, k=3)
    
    # Format with source info
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        results.append(
            f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
        )
    
    return "\n\n".join(results)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    system_prompt="Use search_knowledge_base to find information. Cite sources in your answer.",
    name="rag_with_sources"
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "What are the key points?"}]
})

print(response["messages"][-1].content)
```

---

## Interactive Chat

```python
"""
Interactive RAG Chat Loop
"""
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in docs)

agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    system_prompt="Use search_knowledge_base to answer questions accurately.",
    name="rag_chat"
)

print("RAG Chat - Type 'quit' to exit\n")

while True:
    query = input("You: ").strip()
    
    if query.lower() in ['quit', 'exit', 'q']:
        break
    
    if not query:
        continue
    
    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    print(f"\nAssistant: {response['messages'][-1].content}\n")
```

---

## Best Practices

### Do This

```python
# Retrieve 3-5 documents (good balance)
docs = vectorstore.similarity_search(query, k=3)

# Clear tool descriptions
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    pass

# Provide clear system prompts
system_prompt = "Use search_knowledge_base before answering."
```

### Avoid This

```python
# Too many documents (wastes tokens)
docs = vectorstore.similarity_search(query, k=20)

# Vague tool names
@tool
def tool1(x: str) -> str:
    pass

# No system prompt
agent = create_agent(model, tools=[search])
```

---

## Troubleshooting

**Agent doesn't use tool:**
```python
# Make system prompt more explicit
system_prompt = "You MUST use search_knowledge_base before answering ANY question."
```

**Poor results:**
```python
# Check similarity scores
results = vectorstore.similarity_search_with_score(query, k=5)
for doc, score in results:
    print(f"Score: {score:.4f}")  # Lower = better

# Try more documents
docs = vectorstore.similarity_search(query, k=5)
```

---

## Quick Reference

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Create search tool
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in docs)

# Create agent
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    system_prompt="Use search_knowledge_base to answer questions.",
    name="rag_agent"
)

# Use it
response = agent.invoke({"messages": [{"role": "user", "content": "query"}]})
print(response["messages"][-1].content)
```

---

## Key Takeaways

- **2-Step RAG** = Simple retrieval-then-generation pattern
- **Use `create_agent()`** with a search tool
- **Retrieve 3-5 documents** for best results
- **Clear system prompts** help agent use tools correctly
- **Check similarity scores** if results are poor

---

## Next Steps

- Try [Agentic RAG](./10-agentic-rag.md) for more complex scenarios
- Experiment with different `k` values
- Add metadata filtering for targeted searches

---

## Resources

- [LangChain Retrieval Docs](https://docs.langchain.com/oss/python/langchain/retrieval)
- [Previous: Retrieval](./8-retrieval.md)
- [Previous: Document Loaders](./7-document-loaders-and-text-splitter.md)
