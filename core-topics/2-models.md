# Understanding Models in LangChain

[LangChain Docs - Models](https://docs.langchain.com/oss/python/langchain/models)

Large Language Models (LLMs) are the core reasoning engine for intelligent applications. They can understand and generate human-like text to perform a wide variety of tasks, from summarizing articles and answering questions to writing code, all without needing to be retrained for each specific job.

Modern models have several key capabilities beyond simple text generation:

  * **Tool Calling**: They can interact with external systems, like calling an API to get the latest stock price or querying a database for financial records.
  * **Structured Output**: Their responses can be forced to follow a specific format, such as JSON or a Python class, which is crucial for reliable data processing.
  * **Multimodality**: Some models can understand and process information from images, audio, and video, not just text.
  * **Reasoning**: Advanced models can perform complex, multi-step reasoning to solve problems and arrive at a conclusion.

Within a LangChain **agent**, the model acts as the decision-maker. It analyzes a request, decides which tools to use, interprets the results from those tools, and formulates a final answer. The quality of your model directly determines how well your agent performs.

-----

### Basic Usage

You can use models in two primary ways:

1.  **Inside an Agent**: The model is the core component that drives the agent's logic and tool use.
2.  **As a Standalone Tool**: You can call the model directly for specific tasks like generating a report summary, classifying news sentiment, or extracting data from a document, without needing the full agent framework.

-----

### Initializing a Model

The simplest way to start is with the `init_chat_model` function. LangChain's integrations make it easy to switch between different providers like OpenAI, Anthropic, Google, and others.

Here's how to initialize a model from OpenAI. First, ensure you have the library installed and your API key is set.

```shell
pip install -U "langchain-openai"
```

You can initialize the model using a simple string or by creating an instance of the provider-specific class for more control.

```python
import os
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

# Option 1: Initialize with a string
model_from_string = init_chat_model("openai:gpt-4o")

# Option 2: Initialize with a class instance for more configuration
model_from_class = ChatOpenAI(model="gpt-4o")

# You can now use the model
response = model_from_class.invoke("Summarize the key points of a Q3 earnings report.")
print(response.content)
```

-----

### Key Invocation Methods

There are three primary ways to interact with a model:

  * **invoke()**: Sends a single request and gets a complete response back.
  * **stream()**: Gets the response back in real-time as it's being generated, piece by piece.
  * **batch()**: Sends multiple requests at once for efficient parallel processing.

-----

### Common Parameters

You can configure a model's behavior using several parameters. The exact options vary by provider, but these are the most common:

  * **model**: The specific model identifier you want to use (e.g., `"gpt-4o"`).
  * **temperature**: Controls randomness. A lower value (like `0.1`) makes the output more focused and deterministic, which is good for factual financial summaries. A higher value makes it more creative.
  * **timeout**: The maximum time in seconds to wait for a response.
  * **max\_tokens**: Limits the length of the model's response.
  * **max\_retries**: How many times to retry a request if it fails.

You can pass these directly when initializing the model:

```python
# Initialize a model configured for factual, concise financial analysis
model = init_chat_model(
    "openai:gpt-4o",
    temperature=0.1,
    timeout=60,
    max_tokens=2000,
)
```

-----

### Invocation in Detail

Here’s how to use the three main invocation methods with our financial assistant theme.

#### Invoke

Use `invoke()` for single, complete requests. You can pass a simple string or a list of messages to provide conversational history.

```python
from langchain_core.messages import SystemMessage, HumanMessage

# A list of messages gives the model context for the conversation
conversation = [
    SystemMessage("You are a financial analyst assistant. Your responses should be formal and data-driven."),
    HumanMessage("What were the main drivers of revenue growth for ACME Corp in the last quarter?")
]

response = model.invoke(conversation)
print(response.content)
```

#### Stream

Use `stream()` when you want to display a response as it's generated, which is ideal for long reports or analyses to improve user experience.

```python
# Stream the response chunk by chunk
print("Generating analysis: ", end="")
for chunk in model.stream("Provide a detailed analysis of the current macroeconomic trends affecting the tech sector."):
    print(chunk.content, end="", flush=True)
```

#### Batch

Use `batch()` to process multiple independent inputs in parallel, which is highly efficient for tasks like analyzing a list of stocks.

```python
# A list of financial analysis prompts
prompts = [
    "Summarize the latest earnings call for AAPL.",
    "What are the key risks for GOOG in the next fiscal year?",
    "Provide a competitive analysis for MSFT."
]

# Process all prompts in a single batch call
responses = model.batch(prompts)

for response in responses:
    print(response.content)
    print("-" * 20)
```

-----

### Tool Calling

Tool calling allows the model to use external functions you define. To enable this, you **bind** a list of tools to the model.

Let's define two tools for our financial assistant: one to get a stock price and another to search for news.

```python
from langchain.tools import tool

@tool
def get_stock_price(ticker: str) -> float:
    """Gets the current stock price for a given ticker symbol."""
    # This is a mock function; a real implementation would call a financial API
    if ticker == "ACME":
        return 150.75
    return 0.0

@tool
def get_financial_news(company_name: str) -> str:
    """Searches for the latest financial news about a company."""
    return f"Breaking News: {company_name} announces record profits."

# Bind the tools to the model
model_with_tools = model.bind_tools([get_stock_price, get_financial_news])

# Now, when we invoke the model, it can choose to call a tool
response = model_with_tools.invoke("What is the current stock price for ACME?")

# The response will contain a request to call the tool
print(response.tool_calls)
# Expected output:
# [{'name': 'get_stock_price', 'args': {'ticker': 'ACME'}, 'id': '...'}]
```

When using a model standalone, you are responsible for executing the tool and passing the result back to the model. In an agent, this loop is handled for you automatically.

-----

### Structured Outputs

You can force a model's output to conform to a specific schema, such as a Pydantic model. This is extremely useful for getting reliable, parsable data.

Let's define a schema for a company's financial summary.

```python
from pydantic import BaseModel, Field

class FinancialSummary(BaseModel):
    """A structured summary of a company's financial health."""
    company_name: str = Field(description="The name of the company")
    ticker_symbol: str = Field(description="The stock ticker symbol")
    market_sentiment: str = Field(description="Current market sentiment, e.g., 'Bullish', 'Bearish', or 'Neutral'")
    key_takeaway: str = Field(description="A one-sentence summary of the financial outlook")

# Attach the desired output structure to the model
structured_model = model.with_structured_output(FinancialSummary)

# Invoke the model with a prompt that contains the necessary information
response = structured_model.invoke(
    "Analyze the following report and provide a summary: ACME Corp (ticker: ACME) just released strong earnings, beating analyst expectations. Market outlook is positive."
)

print(response)
# Expected output:
# FinancialSummary(company_name='ACME Corp', ticker_symbol='ACME', market_sentiment='Bullish', key_takeaway='ACME Corp shows a positive financial outlook due to strong earnings that surpassed expectations.')
```

-----

### Supported Models

LangChain integrates with all major model providers, giving you the flexibility to choose the best model for your specific use case. For a complete list, you can refer to the [official LangChain integrations documentation.](https://docs.langchain.com/oss/python/integrations/providers/overview)