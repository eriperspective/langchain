"""
RAG: Full RAG Pipeline
Combine document loading, embeddings, and search.
"""
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

# Load documents
loader = TextLoader("example_docs/sample.txt")
docs = loader.load()

# Create vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)

# Build RAG QA pipeline
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

query = "Summarize the document."
print("Answer:", qa.run(query))

