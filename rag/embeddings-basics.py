"""
RAG: Embeddings
Convert text into vector embeddings for similarity search.
"""
from langchain.embeddings import OpenAIEmbeddings

texts = ["Hello world", "LangChain is powerful"]
embeddings_model = OpenAIEmbeddings()
vectors = [embeddings_model.embed(text) for text in texts]
print("Vector embeddings:", vectors)
