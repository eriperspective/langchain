from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

documents = [
    ("LangChain enables agents.", {"source": "docs"}),
    ("Python is widely used.", {"source": "wiki"})
]

texts, metadatas = zip(*documents)

vectorstore = Chroma.from_texts(
    texts=texts,
    metadatas=metadatas,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)

results = vectorstore.similarity_search(
    "What enables agents?",
    k=1
)

print(results[0].page_content)
print(results[0].metadata)
