# API Context with Context-Aware Generation (CAG) - Policy Service

**LangChain Version:** v1.0+  
**Documentation Reference:** https://docs.langchain.com/oss/python/langchain/context-engineering  
**Last Updated:** October 27, 2025

This guide shows you how to build a simple FastAPI policy service agent. The agent loads policy documents at startup into memory and uses Context-Aware Generation (CAG) to answer questions without needing a retrieval tool.

---

## What You'll Learn

1. Build a simple policy service API with FastAPI
2. Load policy documents into memory at startup
3. Use in-memory context for fast responses
4. Answer policy questions via HTTP endpoints
5. Keep the implementation simple and maintainable

---

## Prerequisites

- Python 3.8+
- OpenAI API key configured in environment
- ChromaDB vector store already created (see [7-document-loaders-and-text-splitter.md](../rag/7-document-loaders-and-text-splitter.md))
- Basic understanding of FastAPI

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  FastAPI Application Startup                            │
│  ↓                                                       │
│  Load ChromaDB from disk                                │
│  ↓                                                       │
│  Extract ALL chunks into memory (one-time)              │
│  ↓                                                       │
│  Create agent with policy context in system prompt      │
│  ↓                                                       │
│  Ready to serve requests                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  HTTP Request: "What is the refund policy?"             │
│  ↓                                                       │
│  Agent already has ALL policy context in memory         │
│  ↓                                                       │
│  Agent answers based on pre-loaded context              │
│  ↓                                                       │
│  Return response as JSON                                │
└─────────────────────────────────────────────────────────┘
```

**Key Benefits:**
- Simple: No retrieval needed - all context loaded once
- Fast: All context loaded once at startup, no database queries per request
- Uses existing ChromaDB: Reuses your already-created vector store
- Easy to understand and maintain

---

## Step 1: Install Dependencies

```bash
pip install fastapi uvicorn langchain langchain-chroma langchain-openai
```

Note: We use ChromaDB to load existing chunks, but we don't query it per request!

---

## Step 2: Project Structure

```
langchain/
├── chroma_db/               # Your existing ChromaDB vector store
└── context/
    ├── policy_service.py    # Simple policy service API
    └── 15-api-context-with-cag.md  # This guide
```

---

## Step 3: Complete Implementation

Here's the simple policy service application:

```python
"""
Policy Service Agent - Simple Context-Aware Generation

This application demonstrates:
1. Loading policy documents at startup (one-time operation)
2. Storing all context in memory (no database needed)
3. Answering policy questions via HTTP endpoints
4. Simple, maintainable code

LangChain Version: v1.0+
Documentation Reference: https://docs.langchain.com/oss/python/langchain/context-engineering
Last Verified: October 27, 2025
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ============================================================================
# Configuration
# ============================================================================

CHROMA_DB_PATH = "./chroma_db"  # Path to existing ChromaDB
LLM_MODEL = "openai:gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# ============================================================================
# Global State (Loaded at Startup)
# ============================================================================

policy_context = ""  # All policy text stored here (in memory)
agent = None         # Agent will be created here

# ============================================================================
# Request/Response Models
# ============================================================================

class PolicyQuery(BaseModel):
    """Request model for policy questions."""
    question: str

class PolicyResponse(BaseModel):
    """Response model with policy answer."""
    answer: str
    source: str = "Policy Document"

# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load ChromaDB and extract all chunks into memory at startup.
    
    This happens ONCE, not on every request. All policy context is loaded
    into memory and included in the agent's system prompt.
    """
    global policy_context, agent
    
    print("=" * 70)
    print("STARTUP: Loading ChromaDB and creating agent...")
    print("=" * 70)
    
    # Step 1: Load ChromaDB
    try:
        print(f"\n[1/3] Loading ChromaDB from: {CHROMA_DB_PATH}")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
        
        # Get all documents from the collection
        collection = vectorstore._collection
        all_docs = collection.get(include=['documents'])
        doc_count = len(all_docs['documents'])
        print(f"      ✓ Loaded ChromaDB with {doc_count} chunks")
        
        # Step 2: Combine all chunks into one context string
        print(f"\n[2/3] Loading all chunks into memory...")
        policy_context = "\n\n".join([
            f"[Section {i+1}]\n{doc}" 
            for i, doc in enumerate(all_docs['documents'])
        ])
        print(f"      ✓ Policy context loaded ({len(policy_context)} characters)")
    
    except Exception as e:
        print(f"      ✗ Failed to load ChromaDB: {e}")
        print("      Agent will run WITHOUT policy context")
        policy_context = ""
    
    # Step 3: Create agent with policy context in system prompt
    print(f"\n[3/3] Creating policy service agent...")
    agent = create_agent(
        model=LLM_MODEL,
        tools=[],  # No tools needed - context is in system prompt
        system_prompt=f"""You are a helpful policy service agent.

Your job is to answer questions about company policies based on the policy document below.

IMPORTANT INSTRUCTIONS:
- Answer questions accurately based on the policy information provided
- If a question is not covered in the policies, clearly state that
- Be concise but thorough in your responses
- Quote specific policy sections when relevant

POLICY DOCUMENT:
{policy_context}

Now, please answer the user's policy questions based on the above information.""",
        name="policy_agent"
    )
    print("      ✓ Policy agent created with full context in memory")
    print("\n" + "=" * 70)
    print("Policy Service API is ready to answer questions!")
    print("=" * 70)
    
    yield  # Application runs here
    
    # Shutdown (optional cleanup)
    print("\nShutting down Policy Service API...")

app = FastAPI(
    title="Policy Service Agent",
    description="Answer questions about company policies using in-memory context",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Policy Service Agent",
        "status": "running",
        "policy_loaded": len(policy_context) > 0,
        "agent_ready": agent is not None
    }

@app.post("/ask", response_model=PolicyResponse)
async def ask_policy_question(query: PolicyQuery):
    """
    Ask a question about company policies.
    
    The agent has ALL policy context loaded in memory and will answer
    based on that information.
    """
    
    if not agent:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Check server logs."
        )
    
    if not policy_context:
        raise HTTPException(
            status_code=503,
            detail="Policy document not loaded. Check server logs."
        )
    
    try:
        # Invoke agent with policy question
        result = agent.invoke({
            "messages": [{"role": "user", "content": query.question}]
        })
        
        # Extract assistant's response
        assistant_message = result["messages"][-1]
        answer = assistant_message.content
        
        return PolicyResponse(
            answer=answer,
            source="Policy Document"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/info")
async def get_info():
    """Get information about the loaded policy document."""
    return {
        "chroma_db_path": CHROMA_DB_PATH,
        "context_size_chars": len(policy_context),
        "context_loaded": len(policy_context) > 0,
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL
    }

# ============================================================================
# Run Instructions
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("Starting Policy Service Agent API...")
    print("=" * 70)
    print("\nMake sure you have:")
    print(f"  1. ChromaDB at: {CHROMA_DB_PATH}")
    print("  2. OPENAI_API_KEY environment variable set")
    print("\nEndpoints:")
    print("  GET  /       - Health check")
    print("  POST /ask    - Ask a policy question")
    print("  GET  /info   - Policy document information")
    print("=" * 70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Step 4: Understanding the Implementation

### How It Works

This is a simple approach to Context-Aware Generation using existing ChromaDB:

#### 1. Load Everything from ChromaDB at Startup
```python
# At startup, we:
# 1. Load ChromaDB from disk
# 2. Extract ALL chunks from the collection
# 3. Combine ALL chunks into one big string
# 4. Put that string directly into the agent's system prompt
```

No retrieval queries per request, no middleware complexity. Just load everything once from ChromaDB and keep it in memory.

#### 2. Create Agent with Full Context
```python
agent = create_agent(
    model=LLM_MODEL,
    tools=[],
    system_prompt=f"""You are a policy agent.
    
    POLICY DOCUMENT:
    {policy_context}  # All policy text is right here!
    
    Answer questions based on the above.""",
    name="policy_agent"
)
```

The agent has all policy information in its system prompt. Every question is answered using this context.

#### 3. Simple Endpoint
```python
@app.post("/ask")
async def ask_policy_question(query: PolicyQuery):
    # Just pass the question to the agent
    # Agent already has all the context it needs
    result = agent.invoke({
        "messages": [{"role": "user", "content": query.question}]
    })
    return result
```

No retrieval step, no context injection. The agent already knows everything!

---

## Step 5: Running the Application

Make sure your OPENAI_API_KEY is set:
```bash
export OPENAI_API_KEY="your-key-here"
```

Then run the service:
```bash
cd context
python policy_service.py
```

You'll see the startup sequence:
```
======================================================================
STARTUP: Loading ChromaDB and creating agent...
======================================================================

[1/3] Loading ChromaDB from: ./chroma_db
      ✓ Loaded ChromaDB with 12 chunks

[2/3] Loading all chunks into memory...
      ✓ Policy context loaded (8547 characters)

[3/3] Creating policy service agent...
      ✓ Policy agent created with full context in memory

======================================================================
Policy Service API is ready to answer questions!
======================================================================
```

The server is now running at `http://localhost:8000`

---

## Step 6: Testing the API

### Test 1: Health Check
```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "service": "Policy Service Agent",
  "status": "running",
  "policy_loaded": true,
  "agent_ready": true
}
```

### Test 2: Ask a Policy Question
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

Expected response:
```json
{
  "answer": "According to the policy document, refunds are processed within 30 days of the request. Full refunds are available for unused products within 90 days of purchase.",
  "source": "Policy Document"
}
```

### Test 3: Check Information
```bash
curl http://localhost:8000/info
```

Expected response:
```json
{
  "chroma_db_path": "./chroma_db",
  "context_size_chars": 8547,
  "context_loaded": true,
  "llm_model": "openai:gpt-4o-mini",
  "embedding_model": "text-embedding-3-small"
}
```

### Test 4: Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What are the working hours?"}
)

data = response.json()
print(data["answer"])
```

---

## Step 7: Why This Approach?

### Advantages
- **Simple**: No retrieval logic needed, no per-request database queries
- **Fast**: Context loaded once, no database lookups per request
- **Reuses ChromaDB**: Uses your existing vector store
- **Maintainable**: Easy to understand, easy to modify
- **Perfect for small documents**: Policy docs, FAQs, manuals

### Limitations
- **Context window size**: Limited by model's context window (GPT-4o-mini: 128k tokens)
- **Not for large corpora**: Use retrieval approach for 100+ page documents
- **No semantic search**: Entire context given, not just relevant parts
- **Fixed at startup**: Need to restart to pick up ChromaDB updates

### When to Use This vs Retrieval Approach

**Use this simple approach (load all context) when:**
- Document is small (< 50 pages)
- Want fastest possible responses
- All context fits comfortably in model's context window
- Want simplest implementation

**Use retrieval approach (query per request) when:**
- Large document corpus (100s of pages)
- Context exceeds model's window
- Want to minimize tokens per query
- Need semantic search precision

---

## Step 8: How CAG Works (Simple Version)

### Traditional RAG (Retrieval per Request)
```
User: "What's the refund policy?"
  → Agent decides to search
  → Calls retrieval tool
  → Queries ChromaDB for relevant chunks
  → Gets top-k results
  → Answers question
```

### Context-Aware Generation (This Approach)
```
Startup: Load ALL chunks from ChromaDB into memory
User: "What's the refund policy?"
  → Agent already has all policies in context
  → Answers immediately (no database query)
```

**Why This Is Better for Small Policy Documents:**
- No per-request retrieval delay
- No retrieval tool needed
- Agent always has full context
- Simpler code, fewer moving parts

---

## Step 9: Extending the Service

### Load from Multiple ChromaDB Collections
```python
from langchain_chroma import Chroma

CHROMA_COLLECTIONS = [
    {"path": "./chroma_db", "name": "policies"},
    {"path": "./hr_chroma_db", "name": "hr"},
]

# At startup, load all collections
all_chunks = []
for config in CHROMA_COLLECTIONS:
    vectorstore = Chroma(
        persist_directory=config["path"],
        embedding_function=embeddings
    )
    collection = vectorstore._collection
    docs = collection.get(include=['documents'])
    all_chunks.extend(docs['documents'])

# Then combine as before
```

### Add Authentication
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/ask")
async def ask_policy_question(
    query: PolicyQuery,
    credentials = Depends(security)
):
    # Verify token
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Process query...
```

### Update Policies Without Restart
Add an endpoint to reload from ChromaDB:
```python
@app.post("/admin/reload")
async def reload_policies():
    """Reload policy documents from ChromaDB (admin only)."""
    global policy_context, agent
    # Re-run the loading logic
    # ... load ChromaDB, extract chunks, recreate agent
    return {"status": "policies reloaded"}
```

---

## Troubleshooting

### ChromaDB Not Found
```
✗ Failed to load ChromaDB: [Errno 2] No such file or directory: './chroma_db'
```

**Solution:** Make sure you've created the ChromaDB first. Run the document loader script from [7-document-loaders-and-text-splitter.md](../rag/7-document-loaders-and-text-splitter.md)

### Context Too Large
```
Error: Context exceeds model's context window
```

**Solutions:**
- Use a model with larger context window (GPT-4o: 128k tokens)
- Switch to retrieval approach (query ChromaDB per request instead of loading all)
- Reduce the number of documents in ChromaDB

### Agent Not Using Policy Context
Agent gives generic answers instead of policy-specific ones.

**Checklist:**
- Check startup logs - did policy load successfully?
- Verify `policy_context` is not empty
- Check `/info` endpoint for context size
- Review system prompt - is context included?

### Slow Startup
Application takes long time to start.

**This is normal!** Loading ChromaDB and extracting all chunks takes time. This is a one-time cost at startup, making subsequent queries very fast.

---

## Summary

You've built a simple policy service agent that:

- Loads all chunks from ChromaDB at startup into memory
- Answers policy questions via FastAPI endpoints
- Uses Context-Aware Generation (no retrieval per request)
- Keeps the implementation simple and maintainable

### Key Takeaways

1. **Load once, query never**: Extract all chunks at startup, no per-request retrieval
2. **Reuse ChromaDB**: Leverage your existing vector store
3. **System prompt as knowledge base**: All context goes in the system prompt
4. **Perfect for small documents**: Policy docs, FAQs, manuals fit this pattern

### When to Use This Pattern

**YES - Use this approach (load all at startup) for:**
- Policy documents (< 50 pages)
- FAQs and help documentation  
- Product manuals
- Company handbooks
- Any static, small document set that fits in context window

**NO - Use retrieval approach (query per request) for:**
- Large document collections (100s of pages)
- When context exceeds model limits
- When you need semantic search precision
- When you want to minimize tokens per query

---

## Next Steps

1. **Run the demo**: Test with your existing ChromaDB
2. **Add more policies**: Load from multiple ChromaDB collections
3. **Add authentication**: Protect your API
4. **Monitor with LangSmith**: See [6-langsmith-setup.md](../langsmith/6-langsmith-setup.md)
5. **Compare with retrieval**: Try [10-agentic-rag.md](../rag/10-agentic-rag.md) for larger documents

---

## Related Guides

- **Context Engineering:** [12-model-context.md](./12-model-context.md), [13-tool-context.md](./13-tool-context.md), [14-life-cycle-context.md](./14-life-cycle-context.md)
- **RAG with Vector DB:** [9-two_step_rag.md](../rag/9-two_step_rag.md), [10-agentic-rag.md](../rag/10-agentic-rag.md)
- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **LangChain Context Engineering:** https://docs.langchain.com/oss/python/langchain/context-engineering

