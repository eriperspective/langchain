"""
Retrieval Demo
==============
Learn different ways to query your vector database.

What this demonstrates:
1. Basic similarity search
2. Similarity search with scores
3. Using the retriever interface
4. Maximum Marginal Relevance (MMR)
5. Score thresholds
6. Metadata filtering

Prerequisites:
- Run 7-document-loaders first to create the vector database
- OpenAI API key

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/oss/python/langchain/retrieval
Last Updated: October 30, 2025
"""

import os

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set your OPENAI_API_KEY environment variable")
    exit(1)

# Check if vector database exists
if not os.path.exists("./chroma_db"):
    print("="*70)
    print("ERROR: Vector database not found!")
    print("="*70)
    print("\nPlease run 7-document-loaders/document_loaders_demo.py first")
    print("to create the vector database.\n")
    exit(1)

print("="*70)
print("RETRIEVAL DEMO - QUERYING YOUR KNOWLEDGE BASE")
print("="*70)

# ============================================================================
# LOAD THE VECTOR STORE
# ============================================================================

print("\n[1/7] Loading vector store...")

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# IMPORTANT: Use the SAME embedding model as when creating the store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

print(" Vector store loaded successfully")

# Test query
query = "What is this document about?"
print(f"\nTest query: '{query}'\n")

# ============================================================================
# METHOD 1: BASIC SIMILARITY SEARCH
# ============================================================================

print("="*70)
print("[2/7] METHOD 1: BASIC SIMILARITY SEARCH")
print("="*70)
print("Returns the k most similar documents to your query.\n")

docs = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(docs, 1):
    preview = doc.page_content[:150].replace('\n', ' ')
    page = doc.metadata.get('page', 'N/A')
    print(f"{i}. (Page {page}) {preview}...\n")

# ============================================================================
# METHOD 2: SIMILARITY SEARCH WITH SCORES
# ============================================================================

print("="*70)
print("[3/7] METHOD 2: WITH SCORES")
print("="*70)
print("Shows how similar each result is to your query.")
print("Lower scores = more similar (0.0 = perfect match)\n")

results = vectorstore.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results, 1):
    preview = doc.page_content[:100].replace('\n', ' ')
    print(f"{i}. Score: {score:.4f} | {preview}...\n")

print("Score interpretation:")
print("  • < 0.5 = Excellent match")
print("  • 0.5-1.0 = Good match")
print("  • 1.0-1.5 = Fair match")
print("  • > 1.5 = Probably not relevant\n")

# ============================================================================
# METHOD 3: USING RETRIEVER INTERFACE
# ============================================================================

print("="*70)
print("[4/7] METHOD 3: RETRIEVER INTERFACE")
print("="*70)
print("Standardized interface for RAG systems.\n")

# Create retriever (used in RAG applications)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents\n")

for i, doc in enumerate(docs, 1):
    preview = doc.page_content[:100].replace('\n', ' ')
    print(f"{i}. {preview}...\n")

# ============================================================================
# METHOD 4: MAXIMUM MARGINAL RELEVANCE (MMR)
# ============================================================================

print("="*70)
print("[5/7] METHOD 4: MAXIMUM MARGINAL RELEVANCE (MMR)")
print("="*70)
print("Balances relevance with diversity - avoids duplicate results.\n")

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,              # Return 5 results
        "fetch_k": 20,       # Consider top 20 candidates
        "lambda_mult": 0.7   # 0=max diversity, 1=max relevance
    }
)

print(f"Configuration:")
print(f"  • k=5: Return 5 results")
print(f"  • fetch_k=20: Consider top 20 candidates")
print(f"  • lambda_mult=0.7: Balance relevance/diversity\n")

docs = mmr_retriever.invoke(query)
print(f"Retrieved {len(docs)} diverse documents\n")

# ============================================================================
# METHOD 5: SCORE THRESHOLD
# ============================================================================

print("="*70)
print("[6/7] METHOD 5: SCORE THRESHOLD")
print("="*70)
print("Only return results above a similarity threshold.\n")

threshold_retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,  # Only high-quality matches
        "k": 5
    }
)

print("Configuration:")
print("  • score_threshold=0.8: Only return scores < 0.8")
print("  • k=5: Return up to 5 results\n")

docs = threshold_retriever.invoke(query)
print(f"Retrieved {len(docs)} high-quality documents")

if len(docs) == 0:
    print("  (No results met the threshold - try a lower value)\n")
else:
    print()

# ============================================================================
# METHOD 6: METADATA FILTERING
# ============================================================================

print("="*70)
print("[7/7] METHOD 6: METADATA FILTERING")
print("="*70)
print("Filter results by document properties.\n")

# Search specific page
print("Example 1: Search only page 0")
docs = vectorstore.similarity_search(
    query,
    k=5,
    filter={"page": 0}
)
print(f"  Found {len(docs)} results from page 0\n")

# Search specific document (if you have multiple PDFs)
print("Example 2: Search by source file")
docs = vectorstore.similarity_search(
    query,
    k=5,
    filter={"source": "sample-doc.pdf"}
)
print(f"  Found {len(docs)} results from sample-doc.pdf\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("="*70)
print("DEMO COMPLETE!")
print("="*70)

print("\nKEY CONCEPTS:")

print("\n1. RETRIEVAL METHODS:")
print("   • similarity_search(): Basic relevance-based search")
print("   • similarity_search_with_score(): Shows similarity scores")
print("   • as_retriever(): Standardized interface for RAG")
print("   • MMR: Balances relevance with diversity")
print("   • Score threshold: Quality control")
print("   • Metadata filtering: Search specific documents/pages")

print("\n2. WHEN TO USE EACH:")
print("   • Basic search: General queries")
print("   • With scores: When you need to validate quality")
print("   • Retriever: Building RAG applications")
print("   • MMR: When results are too similar/redundant")
print("   • Threshold: When quality matters more than quantity")
print("   • Filtering: Multi-document databases")

print("\n3. BEST PRACTICES:")
print("   • Use specific queries for better results")
print("   • k=3-5 is usually a good balance")
print("   • Check scores to validate result quality")
print("   • Use MMR if results are too repetitive")
print("   • Lower scores = more similar")

print("\n4. QUICK REFERENCE:")
print("""
# Basic search
docs = vectorstore.similarity_search(query, k=3)

# With scores
results = vectorstore.similarity_search_with_score(query, k=3)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# MMR
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
docs = vectorstore.similarity_search(query, k=5, filter={"page": 0})
""")

print("\n" + "="*70)
print("NEXT STEPS:")
print("• Try 9-two-step-rag to build a Q&A system")
print("• Try 10-agentic-rag for an agent that decides when to search")
print("="*70 + "\n")

