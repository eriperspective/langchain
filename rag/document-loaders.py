"""
RAG: Document Loaders
Load documents from local files, PDFs, or URLs.
"""
from langchain.document_loaders import TextLoader

loader = TextLoader("example_docs/sample.txt")
docs = loader.load()
print("Loaded Documents:", docs)
