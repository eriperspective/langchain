from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

texts = [
    "LangChain helps build LLM applications.",
    "Python is a programming language.",
    "Agents can use tools to take actions."
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings
)

results = vectorstore.similarity_search("What are agents?", k=1)

print(results[0].page_content)
