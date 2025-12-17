from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector = embeddings.embed_query("What is LangChain?")

print(f"Vector length: {len(vector)}")
print(vector[:5])
