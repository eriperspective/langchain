"""
RAG: Vector Search
Search for relevant documents using embeddings.
"""
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

texts = ["LangChain tutorial", "Context augmented generation"]
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(texts, embeddings)

query = "What is CAG?"
results = vector_store.similarity_search(query)
print("Search Results:", results)
