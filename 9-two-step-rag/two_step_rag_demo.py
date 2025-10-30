"""
2-Step RAG Demo
===============
Build a simple RAG system where retrieval always happens before generation.

What is 2-Step RAG?
- Step 1: Retrieve relevant documents from knowledge base
- Step 2: Generate answer using those documents as context

Prerequisites:
- Run 7-document-loaders first to create the vector database
- OpenAI API key

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/oss/python/langchain/retrieval#2-step-rag
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
print("2-STEP RAG DEMO")
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
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    # Retrieve top 3 most relevant chunks
    docs = vectorstore.similarity_search(query, k=3)
    
    # Combine the chunks into one context string
    return "\n\n".join(doc.page_content for doc in docs)


print(" Search tool created")

# ============================================================================
# CREATE RAG AGENT
# ============================================================================

print("\n[3/3] Creating RAG agent...")

# THE KEY: System prompt tells agent to ALWAYS use the search tool
agent = create_agent(
    model="openai:gpt-4o-mini",
    tools=[search_knowledge_base],
    system_prompt="Use search_knowledge_base to find information before answering questions.",
    name="rag_agent"
)

print(" RAG agent ready\n")

# ============================================================================
# TEST THE AGENT
# ============================================================================

print("="*70)
print("TESTING 2-STEP RAG")
print("="*70)

test_queries = [
    "What is this document about?",
    "Summarize the main points.",
    "What are the key findings?",
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'─'*70}")
    print(f"Question {i}: {query}")
    print("─"*70)
    
    response = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    print(f"\nAnswer: {response['messages'][-1].content}\n")

# ============================================================================
# INTERACTIVE MODE
# ============================================================================

print("\n" + "="*70)
print("INTERACTIVE MODE")
print("="*70)
print("Ask questions about your document!")
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

