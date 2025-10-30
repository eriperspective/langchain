"""
Agentic RAG Demo
================
Build a RAG system where the agent decides when to retrieve information.

Key Difference from 2-Step RAG:
- 2-Step RAG: ALWAYS retrieves
- Agentic RAG: Agent DECIDES when to retrieve

Prerequisites:
- Run 7-document-loaders first to create the vector database
- OpenAI API key

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/oss/python/langchain/retrieval#agentic-rag
Last Updated: October 30, 2025
"""

import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set your OPENAI_API_KEY environment variable")
    exit(1)

# Check if vector database exists
if not os.path.exists("./chroma_db"):
    print("="*70)
    print("ERROR: Vector database not found!")
    print("="*70)
    print("\nPlease run 7-document-loaders/document_loaders_demo.py first\n")
    exit(1)

print("="*70)
print("AGENTIC RAG DEMO")
print("="*70)

# ============================================================================
# LOAD VECTOR STORE
# ============================================================================

print("\n[1/3] Loading vector store...")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

print(" Vector store loaded")

# ============================================================================
# CREATE SEARCH TOOL
# ============================================================================

print("\n[2/3] Creating search tool...")

@tool
def search_documents(query: str) -> str:
    """Search documents for specific information."""
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in docs)


print(" Search tool created")

# ============================================================================
# CREATE AGENTIC RAG AGENT
# ============================================================================

print("\n[3/3] Creating agentic RAG agent...")

# THE KEY DIFFERENCE: System prompt gives agent AUTONOMY
# It can choose whether to search or answer directly
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_documents],
    system_prompt=(
        "Use search_documents when you need specific document information. "
        "If you can answer from general knowledge, do so."
    ),
    name="agentic_rag"
)

print(" Agentic RAG agent ready\n")

# ============================================================================
# TEST THE AGENT
# ============================================================================

print("="*70)
print("TESTING AGENTIC RAG")
print("="*70)
print("\nWatch how the agent decides when to search and when to answer directly.\n")

# Mix of document-specific and general questions
test_queries = [
    ("What is this document about?", "Should search"),
    ("What is 2+2?", "Should answer directly"),
    ("Summarize the key findings.", "Should search"),
    ("What is the capital of France?", "Should answer directly"),
    ("What are the main conclusions?", "Should search"),
]

for i, (query, expected_behavior) in enumerate(test_queries, 1):
    print("─"*70)
    print(f"Test {i}: {query}")
    print(f"Expected: {expected_behavior}")
    print("─"*70)
    
    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    print(f"\nAnswer: {response['messages'][-1].content}\n")

# ============================================================================
# COMPARISON WITH 2-STEP RAG
# ============================================================================

print("\n" + "="*70)
print("KEY DIFFERENCES: 2-STEP VS AGENTIC RAG")
print("="*70)

print("""
2-STEP RAG:
• System prompt: "Use search_knowledge_base to find information."
• Behavior: ALWAYS searches the knowledge base
• Best for: Document-only Q&A, FAQs

AGENTIC RAG:
• System prompt: "Use search_documents when you need document info."
• Behavior: Agent DECIDES when to search
• Best for: Mixed queries (general + document-specific)

Example:
  Question: "What is 2+2?"
  2-Step RAG: Searches documents (unnecessary)
  Agentic RAG: Answers directly (efficient)
  
  Question: "What does the document say?"
  2-Step RAG: Searches documents (correct)
  Agentic RAG: Searches documents (correct)
""")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

print("\n" + "="*70)
print("INTERACTIVE MODE")
print("="*70)
print("Ask both document and general questions!")
print("Watch the agent decide when to search.")
print("Type 'quit' to exit.\n")

while True:
    query = input("You: ").strip()
    
    if query.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!\n")
        break
    
    if not query:
        continue
    
    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    print(f"\nAssistant: {response['messages'][-1].content}\n")

print("\n" + "="*70)
print("KEY TAKEAWAYS:")
print("="*70)
print("""
1. Agentic RAG gives agents AUTONOMY to decide when to retrieve
2. More efficient - doesn't search unnecessarily
3. Better user experience - faster for general questions
4. System prompt is critical: "Use tool WHEN needed" vs "Use tool ALWAYS"
5. Perfect for applications with mixed query types
""")
print("="*70 + "\n")

