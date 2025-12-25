from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

proposal = llm.invoke("Propose a feature for an AI assistant.")

print("Proposal:")
print(proposal.content)

approval = input("Approve? (yes/no): ")

if approval == "yes":
    final = llm.invoke("Proceed with implementation details.")
    print(final.content)
else:
    print("Execution stopped.")
