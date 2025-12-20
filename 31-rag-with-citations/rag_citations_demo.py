from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

docs = [
    ("LangChain supports RAG systems.", {"source": "LangChain Docs"}),
    ("RAG improves factual accuracy.", {"source": "AI Research"})
]

texts, metadatas = zip(*docs)

vectorstore = Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)

retrieved = vectorstore.similarity_search("Why use RAG?", k=1)[0]

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = f"""
Answer the question using the source below.

Source ({retrieved.metadata['source']}):
{retrieved.page_content}
"""

response = llm.invoke(prompt)

print(response.content)
print("Source:", retrieved.metadata["source"])
