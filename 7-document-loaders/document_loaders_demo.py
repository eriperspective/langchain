"""
Document Loaders and Vector Store Demo
=======================================
Learn how to load documents, split them into chunks, and store them in a vector database.

What this demonstrates:
1. Loading PDF documents with PyPDFLoader
2. Splitting text into chunks with RecursiveCharacterTextSplitter
3. Creating embeddings with OpenAI
4. Storing in ChromaDB vector database
5. Querying with similarity search

Prerequisites:
- OpenAI API key
- A PDF file to process (or use the included sample-doc.pdf)

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/oss/python/langchain/knowledge-base
Last Updated: October 30, 2025
"""

import os

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set your OPENAI_API_KEY environment variable")
    print("Example: export OPENAI_API_KEY='your-key-here'")
    exit(1)

print("="*70)
print("DOCUMENT PROCESSING AND VECTOR STORE DEMO")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

PDF_PATH = "./sample-doc.pdf"  # Path to the PDF we want to process
PERSIST_DIR = "./chroma_db"     # Where to save our vector database
TEST_QUERY = "What is the reason for this document?"  # Question to test our database

# ============================================================================
# STEP 1: LOAD PDF
# ============================================================================
# PyPDFLoader reads PDF files and extracts text
# It returns one Document object per page

print("\n[1/5] Loading PDF...")
print(f"   File: {PDF_PATH}")

try:
    from langchain_community.document_loaders import PyPDFLoader
    
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    
    print(f" Loaded {len(documents)} pages")
    print(f"   First page preview: {documents[0].page_content[:100]}...")
    
except FileNotFoundError:
    print(f"\n ERROR: PDF file not found at {PDF_PATH}")
    print("   Please make sure sample-doc.pdf exists in the current directory")
    print("   or update PDF_PATH in the script to point to your PDF file.")
    exit(1)
except Exception as e:
    print(f"\n ERROR loading PDF: {e}")
    exit(1)

# ============================================================================
# STEP 2: SPLIT INTO CHUNKS
# ============================================================================
# RecursiveCharacterTextSplitter breaks documents into smaller pieces
# This improves retrieval accuracy and fits within embedding limits

print("\n[2/5] Splitting into chunks...")
print(f"   Chunk size: 1000 characters")
print(f"   Chunk overlap: 200 characters")

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Maximum characters per chunk
    chunk_overlap=200     # Overlap between chunks to preserve context
)

chunks = text_splitter.split_documents(documents)
print(f" Created {len(chunks)} chunks")

if len(chunks) > 0:
    print(f"   First chunk preview: {chunks[0].page_content[:100]}...")

# ============================================================================
# STEP 3: INITIALIZE EMBEDDINGS
# ============================================================================
# Embeddings convert text into numerical vectors
# OpenAI's text-embedding-3-small is cost-effective and high quality

print("\n[3/5] Initializing OpenAI embeddings...")
print(f"   Model: text-embedding-3-small")
print(f"   Dimensions: 1536")

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print(" Ready")

# ============================================================================
# STEP 4: CREATE VECTOR DATABASE
# ============================================================================
# ChromaDB stores the chunks and their embeddings
# The database is saved to disk for reuse

print("\n[4/5] Creating vector database...")
print(f"   Database location: {PERSIST_DIR}")
print(f"   This may take a moment...")

from langchain_community.vectorstores import Chroma

try:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f" Stored {len(chunks)} chunks in ChromaDB")
    
except Exception as e:
    print(f"\n ERROR creating vector store: {e}")
    print("\n   Note: ChromaDB requires compilation. If you see build errors,")
    print("   it's because ChromaDB needs C++ build tools which weren't installed.")
    print("   The other tutorials will still work!")
    exit(1)

# ============================================================================
# STEP 5: QUERY TO VERIFY
# ============================================================================
# Test the database with a similarity search

print("\n[5/5] Testing query...")
print(f"   Query: '{TEST_QUERY}'")
print(f"   Retrieving top 3 results...")

results = vectorstore.similarity_search(TEST_QUERY, k=3)
print(f" Found {len(results)} results\n")

for i, doc in enumerate(results, 1):
    page = doc.metadata.get('page', 'N/A')
    preview = doc.page_content[:200].replace('\n', ' ')
    print(f"Result {i} (Page {page}):")
    print(f"   {preview}...\n")

# Show similarity scores
print("Checking similarity scores (lower = more similar)...")
results_with_scores = vectorstore.similarity_search_with_score(TEST_QUERY, k=3)
for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"   Result {i} score: {score:.4f}")

print("\n" + "="*70)
print("SUCCESS! VECTOR DATABASE CREATED")
print("="*70)
print(f"\nDatabase location: {PERSIST_DIR}")
print("\nTo use this database in other scripts:")
print("""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Query it
results = vectorstore.similarity_search("your query", k=3)
""")

print("\n" + "="*70)
print("\nKEY CONCEPTS:")
print("\n1. DOCUMENT LOADING:")
print("   • PyPDFLoader extracts text from PDFs")
print("   • Returns one Document per page")
print("   • Each Document has .page_content (text) and .metadata (info)")

print("\n2. TEXT SPLITTING:")
print("   • Breaks documents into smaller chunks")
print("   • chunk_size=1000: Max characters per chunk")
print("   • chunk_overlap=200: Overlap preserves context")
print("   • Tries to split at natural boundaries (paragraphs, sentences)")

print("\n3. EMBEDDINGS:")
print("   • Convert text into numerical vectors (1536 dimensions)")
print("   • Similar text = similar vectors")
print("   • text-embedding-3-small: Cost-effective, high quality")
print("   • Cost: ~$0.02 per 1M tokens (very cheap!)")

print("\n4. CHROMADB:")
print("   • Vector database for storing embeddings")
print("   • Saves to disk (./chroma_db)")
print("   • Can be reloaded without re-embedding")
print("   • Supports similarity search and filtering")

print("\n5. SIMILARITY SEARCH:")
print("   • Finds chunks most similar to your query")
print("   • k=3 means return top 3 results")
print("   • Score interpretation:")
print("     - < 0.5: Excellent match")
print("     - 0.5-1.0: Good match")
print("     - > 1.5: Probably not relevant")

print("\n" + "="*70)
print("\nNEXT STEPS:")
print("• See 8-retrieval for advanced querying techniques")
print("• See 9-two-step-rag to build a Q&A system with this database")
print("• See 10-agentic-rag for an agent that decides when to search")
print("="*70 + "\n")

