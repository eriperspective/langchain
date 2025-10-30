"""
LangChain Models Demo
=====================
Learn how to work with language models directly and understand their key methods.

What we'll cover:
1. Initializing models
2. Invoke, stream, and batch methods
3. Tool calling
4. Structured outputs

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/oss/python/langchain/models
Last Updated: October 30, 2025
"""

import os
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from pydantic import BaseModel, Field

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set your OPENAI_API_KEY environment variable")
    exit(1)


print("="*70)
print("PART 1: INITIALIZING MODELS")
print("="*70)

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
# Models are the LLM engines that power your applications.
# You can initialize them in two ways:

# Method 1: Simple string (LangChain figures out the provider)
print("\n--- Method 1: Using string ---")
model_from_string = init_chat_model("openai:gpt-4o-mini")
print(f"Model initialized: {type(model_from_string).__name__}")

# Method 2: Using a class instance for more control
print("\n--- Method 2: Using class instance ---")
model_from_class = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,    # Low temperature = more predictable/factual
    timeout=60,         # Max time to wait for response
    max_tokens=2000,    # Limit response length
)
print(f"Model initialized: {type(model_from_class).__name__}")


print("\n" + "="*70)
print("PART 2: INVOCATION METHODS")
print("="*70)

# ============================================================================
# THREE WAYS TO CALL A MODEL
# ============================================================================

# For financial analyst theme as in the original guide
model = init_chat_model(
    "openai:gpt-4o-mini",
    temperature=0.1,  # Low temperature for factual financial analysis
    timeout=60,
    max_tokens=2000,
)

print("\n--- Method 1: INVOKE (Complete Response) ---")
# invoke() sends a request and waits for the complete response

# You can pass a simple string
simple_response = model.invoke("What is the capital of France?")
print(f"\nSimple query response: {simple_response.content}")

# Or pass a list of messages for conversation context
conversation = [
    SystemMessage("You are a financial analyst assistant. Your responses should be formal and data-driven."),
    HumanMessage("What were the main drivers of revenue growth for tech companies in Q3 2024?")
]

detailed_response = model.invoke(conversation)
print(f"\nDetailed query response: {detailed_response.content[:150]}...")


print("\n--- Method 2: STREAM (Real-time Response) ---")
# stream() returns the response piece by piece as it's generated
# Great for showing progress to users

print("\nStreaming response: ", end="", flush=True)
for chunk in model.stream("Provide a brief analysis of current macroeconomic trends affecting the tech sector."):
    print(chunk.content, end="", flush=True)
print("\n")


print("\n--- Method 3: BATCH (Multiple Requests in Parallel) ---")
# batch() processes multiple inputs efficiently at once
# Perfect for analyzing multiple items (stocks, reports, etc.)

prompts = [
    "Summarize the key points of Apple's latest earnings call in one sentence.",
    "What are the main risks for Google in the next fiscal year?",
    "Provide a competitive analysis for Microsoft in one sentence."
]

print("\nProcessing 3 prompts in batch...")
responses = model.batch(prompts)

for i, response in enumerate(responses, 1):
    print(f"\n{i}. {response.content}")


print("\n" + "="*70)
print("PART 3: TOOL CALLING")
print("="*70)

# ============================================================================
# TOOL CALLING
# ============================================================================
# Tool calling lets the model use external functions to get information

# Define tools
@tool
def get_stock_price(ticker: str) -> float:
    """Gets the current stock price for a given ticker symbol."""
    # Mock data for demo
    prices = {
        "ACME": 150.75,
        "AAPL": 180.25,
        "GOOGL": 142.50,
    }
    return prices.get(ticker.upper(), 0.0)


@tool
def get_financial_news(company_name: str) -> str:
    """Searches for the latest financial news about a company."""
    # Mock news
    news = {
        "ACME": "ACME Corp announces record quarterly profits, beating analyst expectations.",
        "Apple": "Apple unveils new product line, investors respond positively.",
    }
    return news.get(company_name, f"No recent news found for {company_name}.")


# Bind tools to the model
# The model can now choose to call these tools when needed
model_with_tools = model.bind_tools([get_stock_price, get_financial_news])

print("\n--- Example: Model decides to call a tool ---")
response = model_with_tools.invoke("What is the current stock price for ACME?")

# Check if the model wants to call a tool
if response.tool_calls:
    print(f"\nModel requested tool call:")
    for tool_call in response.tool_calls:
        print(f"  Tool: {tool_call['name']}")
        print(f"  Arguments: {tool_call['args']}")
    
    # Note: When using models standalone, YOU must execute the tool and pass results back.
    # In an agent, this loop is handled automatically.
else:
    print(f"\nModel response: {response.content}")


print("\n" + "="*70)
print("PART 4: STRUCTURED OUTPUTS")
print("="*70)

# ============================================================================
# STRUCTURED OUTPUTS
# ============================================================================
# Force the model to return data in a specific format
# Perfect for getting reliable, parsable data

# Define the output schema using Pydantic
class FinancialSummary(BaseModel):
    """A structured summary of a company's financial health."""
    company_name: str = Field(description="The name of the company")
    ticker_symbol: str = Field(description="The stock ticker symbol")
    market_sentiment: str = Field(
        description="Current market sentiment, e.g., 'Bullish', 'Bearish', or 'Neutral'"
    )
    key_takeaway: str = Field(description="A one-sentence summary of the financial outlook")


# Attach the structure to the model
structured_model = model.with_structured_output(FinancialSummary)

print("\n--- Example: Get structured financial summary ---")

# Give the model information and ask for structured output
response = structured_model.invoke(
    "Analyze the following report and provide a summary: "
    "ACME Corp (ticker: ACME) just released strong earnings, "
    "beating analyst expectations by 15%. The market outlook is very positive."
)

print(f"\nStructured output:")
print(f"  Company: {response.company_name}")
print(f"  Ticker: {response.ticker_symbol}")
print(f"  Sentiment: {response.market_sentiment}")
print(f"  Key Takeaway: {response.key_takeaway}")


print("\n" + "="*70)
print("DEMO COMPLETE!")
print("="*70)
print("\nKey takeaways:")
print("1. Models can be initialized with strings or class instances")
print("2. invoke() for complete responses, stream() for real-time, batch() for multiple")
print("3. bind_tools() lets models call external functions")
print("4. with_structured_output() forces models to return specific formats")
print("5. When using models standalone, you handle tool execution manually")
print("6. In agents, tool execution is automatic")
print("="*70 + "\n")

