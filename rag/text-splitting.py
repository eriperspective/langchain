"""
RAG: Text Splitting
Chunk large documents into smaller pieces for retrieval.
"""
from langchain.text_splitter import CharacterTextSplitter

text = "Long text to split into smaller chunks for RAG processing."
splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=5)
chunks = splitter.split_text(text)
print("Chunks:", chunks)
