# Retrieval - Querying Your Knowledge Base

**LangChain Version:** v1.0+  
**Documentation:** https://docs.langchain.com/oss/python/langchain/retrieval#overview  
**Last Updated:** October 23, 2025

Learn how to query your vector database using different retrieval strategies to find relevant documents.

---

## Prerequisites

**IMPORTANT:** Complete [0-document-loaders-and-text-splitter.md](./0-document-loaders-and-text-splitter.md) first. 

This guide assumes you have:
- A ChromaDB vector store at `./chroma_db`
- Documents loaded, chunked, and embedded
- OpenAI API key configured

---

## What is Retrieval?

LLMs have two key limitations:
- **Finite context** — can't ingest entire corpora at once
- **Static knowledge** — training data is frozen in time

**Retrieval** solves this by fetching relevant external knowledge at query time. This is the foundation of **RAG** (Retrieval-Augmented Generation).

---

## Load Your Vector Store

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Load existing vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

**Note:** Use the **same embedding model** you used when creating the vector store.

---

## Retrieval Methods

### 1. Basic Similarity Search

```python
query = "What is this document about?"
docs = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content[:150]}...")
```

### 2. Similarity Search with Scores

```python
# Lower scores = more similar (0.0 = perfect match)
results = vectorstore.similarity_search_with_score(query, k=3)

for doc, score in results:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}...")
```

**Score Guidelines:**
- `< 0.5` = Excellent match
- `0.5-1.0` = Good match
- `1.0-1.5` = Fair match
- `> 1.5` = Probably not relevant

### 3. Using Retriever Interface

```python
# Create retriever (standardized interface for RAG)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

docs = retriever.invoke(query)
```

---

## Advanced Retrieval Strategies

### Maximum Marginal Relevance (MMR)

Balances relevance with diversity - avoids returning duplicate or very similar results.

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,              # Return 5 results
        "fetch_k": 20,       # Consider top 20 candidates
        "lambda_mult": 0.7   # 0=max diversity, 1=max relevance
    }
)
```

### Score Threshold (Quality Control)

Only return results above a similarity threshold.

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,  # Only high-quality matches
        "k": 5
    }
)
```

### Metadata Filtering

Filter results by document properties.

```python
# Search specific page
docs = vectorstore.similarity_search(
    query,
    k=5,
    filter={"page": 1}
)

# Search specific document
docs = vectorstore.similarity_search(
    query,
    k=5,
    filter={"source": "sample-doc.pdf"}
)
```

---

## Complete Example

```python
"""
Complete Retrieval Example
Prerequisites: Run 0-document-loaders-and-text-splitter.md first
"""
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Load vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Test query
query = "What is the main topic of this document?"

# Method 1: Direct search
print("=== Basic Search ===")
docs = vectorstore.similarity_search(query, k=3)
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.page_content[:100]}...\n")

# Method 2: With scores
print("=== With Scores ===")
results = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in results:
    print(f"Score: {score:.4f} | {doc.page_content[:80]}...\n")

# Method 3: Retriever interface
print("=== Using Retriever ===")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents")
```

---

## Best Practices

### Do This

```python
# Use specific queries
query = "What are the benefits of the proposed system?"

# Choose appropriate k (3-5 is usually good)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Use MMR for diverse results
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.7}
)

# Check scores to validate quality
results = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in results:
    if score < 1.0:  # Good quality threshold
        print(doc.page_content)
```

### Avoid This

```python
# Vague queries
query = "stuff"

# Too many results (wastes tokens, adds noise)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Too few results (might miss context)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
```

---

## Troubleshooting

### Vector Store Not Found
```python
import os
if not os.path.exists("./chroma_db"):
    print("ERROR: Run 0-document-loaders-and-text-splitter.md first!")
```

### Poor Results
- Use more specific queries
- Increase `k` to get more results
- Try MMR for diversity
- Check similarity scores (if all > 1.5, query doesn't match content)

### Wrong Embedding Model Error
```python
# Must use the SAME model as when creating the store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

---

## Quick Reference

```python
# Load store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Basic search
docs = vectorstore.similarity_search("query", k=3)

# With scores
results = vectorstore.similarity_search_with_score("query", k=3)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke("query")

# MMR (diverse results)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.7}
)

# Score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.8, "k": 5}
)

# Metadata filter
docs = vectorstore.similarity_search("query", k=5, filter={"page": 1})
```

---

## Key Takeaways

- **Retrieval fetches relevant context** at query time to overcome LLM limitations

- **Three main methods** - direct search, with scores, retriever interface

- **Advanced strategies** - MMR for diversity, score thresholds for quality

- **Lower scores = better** - aim for scores < 1.0

- **Query quality matters** - specific queries get better results

- **Use retrievers for RAG** - provides standardized interface

---

## Next Steps

- **Try 2-Step RAG** (`2-two_step_rag.md`) - Simple retrieval + generation
- **Try Agentic RAG** (`3-agentic-rag.md`) - Agent decides when to retrieve
- **Experiment with MMR** and score thresholds
- **Test different queries** to understand quality

---

## Resources

- [LangChain Retrieval Docs](https://docs.langchain.com/oss/python/langchain/retrieval)
- [Previous Guide: Document Loaders](./0-document-loaders-and-text-splitter.md)
- [ChromaDB Docs](https://docs.trychroma.com/)
