from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
LangChain is a framework for building applications powered by language models.
It supports agents, tools, memory, and retrieval augmented generation.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:\n{chunk}\n")
