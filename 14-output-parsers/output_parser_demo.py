from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

response = llm.invoke("Give me one benefit of LangChain.")

parsed_output = parser.invoke(response)

print(parsed_output)
