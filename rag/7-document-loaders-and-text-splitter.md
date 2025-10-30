# Document Loaders and Text Splitting with ChromaDB

**LangChain Version:** v1.0+  
**Documentation:** https://docs.langchain.com/oss/python/langchain/knowledge-base  
**Last Updated:** October 23, 2025

This guide shows you how to load PDFs, split them into chunks, generate embeddings with OpenAI, store them in ChromaDB, and query the database.

---

## What You'll Learn

1. Load PDF documents with `PyPDFLoader`
2. Split text into chunks with `RecursiveCharacterTextSplitter`
3. Generate embeddings with OpenAI's `text-embedding-3-small`
4. Store embeddings in ChromaDB vector database
5. Query the database with similarity search

---

## Prerequisites

- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- A PDF file to test

---

## Step 1: Install Dependencies

```bash
pip install langchain langchain-community langchain-openai chromadb pypdf
```

---

## Step 2: Set OpenAI API Key

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

---

## Step 3: Run the Demo Script

Use the included `pdf_to_chromadb_demo.py` script:

1. Edit the script and set your PDF path
2. Run: `python pdf_to_chromadb_demo.py`

That's it! The script will load, split, embed, store, and query your PDF.

---

## Step 4: Complete Code Example

Here's the complete code (also available in `pdf_to_chromadb_demo.py`):

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
PDF_PATH = "./sample-doc.pdf"  # Path to the PDF we want to process
PERSIST_DIR = "./chroma_db"     # Where to save our vector database
TEST_QUERY = "What is reason for this document?"  # Question to test our database

# Step 1: Load PDF
print("[1/5] Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"✓ Loaded {len(documents)} pages")

# Step 2: Split into chunks
print("[2/5] Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"✓ Created {len(chunks)} chunks")

# Step 3: Initialize embeddings
print("[3/5] Initializing OpenAI embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("✓ Ready")

# Step 4: Store in ChromaDB
print("[4/5] Creating vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR
)
print(f"✓ Stored {len(chunks)} chunks")

# Step 5: Query to verify
print("[5/5] Testing query...")
results = vectorstore.similarity_search(TEST_QUERY, k=3)
print(f"✓ Found {len(results)} results\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content[:200]}...")
    print(f"(Page {doc.metadata.get('page', 'N/A')})\n")

# Show similarity scores
results_with_scores = vectorstore.similarity_search_with_score(TEST_QUERY, k=3)
for i, (_, score) in enumerate(results_with_scores, 1):
    print(f"Result {i} score: {score:.4f}")

print(f"\n✓ Success! Database at: {PERSIST_DIR}")
```

---

## Understanding the Key Components

### 1. PyPDFLoader
- Loads PDF files and extracts text
- Returns one `Document` object per page
- Each document has `.page_content` (text) and `.metadata` (page number, source)

### 2. RecursiveCharacterTextSplitter
- Splits documents into smaller chunks for better retrieval
- `chunk_size=1000`: Max characters per chunk
- `chunk_overlap=200`: Overlap between chunks (preserves context)
- Tries to split at natural boundaries (paragraphs, sentences)

### 3. OpenAIEmbeddings
- Model: `text-embedding-3-small` (cost-effective, 1536 dimensions)
- Converts text into numerical vectors
- Alternative: `text-embedding-3-large` (better quality, more expensive)

### 4. ChromaDB
- Persistent vector database
- Automatically generates and stores embeddings
- Saved locally in `./chroma_db`
- Can be reloaded without re-embedding

### 5. Similarity Search
- Finds the `k` most similar chunks to your query
- Uses cosine similarity between embeddings
- Lower scores = more similar

---

## Loading an Existing Database

To reuse a database you've already created:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Query it
results = vectorstore.similarity_search("your query", k=3)
```

---

## Advanced Querying

### Get Similarity Scores
```python
results_with_scores = vectorstore.similarity_search_with_score("query", k=5)
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}...")
```

### Filter by Metadata
```python
results = vectorstore.similarity_search("query", k=5, filter={"page": 1})
```

### Maximum Marginal Relevance (diverse results)
```python
results = vectorstore.max_marginal_relevance_search("query", k=5, lambda_mult=0.5)
```

---

## Tips & Best Practices

**Chunk Size:**
- Too small (<500): Loses context
- Too large (>2000): Less precise
- Sweet spot: 800-1200 characters

**Cost (text-embedding-3-small):**
- ~$0.02 per 1M tokens
- 100-page PDF ≈ $0.007 (less than 1 cent)

**Common Issues:**
- File not found → Check your PDF path
- API key error → Set `OPENAI_API_KEY` environment variable  
- Poor results → Try more specific queries or adjust chunk size

---

## Next Steps

- Build a RAG application (use retrieved chunks with an LLM)
- Add multiple PDFs to the same database
- Try other document loaders (`TextLoader`, `UnstructuredMarkdownLoader`)
- Implement metadata filtering

---

## Resources

- [LangChain Docs](https://docs.langchain.com/oss/python/langchain)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

