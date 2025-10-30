"""
LangChain Messages Demo
=======================
Learn how messages structure conversations with language models.

What we'll cover:
1. Message types (System, Human, AI, Tool)
2. Text vs Message prompts
3. Multimodal content (text + images)

LangChain Version: v1.0+
Documentation: https://docs.langchain.com/oss/python/langchain/messages
Last Updated: October 30, 2025
"""

import os
from langchain.chat_models import init_chat_model
from langchain.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage
)
from langchain.tools import tool

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: Please set your OPENAI_API_KEY environment variable")
    exit(1)


print("="*70)
print("PART 1: TEXT VS MESSAGE PROMPTS")
print("="*70)

# ============================================================================
# TEXT VS MESSAGE PROMPTS
# ============================================================================
# You can interact with models using simple strings or message objects

model = init_chat_model("openai:gpt-4o-mini")

# Method 1: Simple text prompt (easiest for one-off tasks)
print("\n--- Method 1: Simple Text Prompt ---")
response = model.invoke("What are your return policies?")
print(f"Response: {response.content[:100]}...")


# Method 2: Message objects (essential for conversations with history)
print("\n--- Method 2: Message Objects ---")
messages = [
    SystemMessage("You are a customer support agent for the company 'Gadgetron'. Be helpful and professional."),
    HumanMessage("Hello, I have a question about my recent order.")
]

response = model.invoke(messages)
print(f"Response: {response.content}")


# Method 3: Dictionary format (compatible with APIs)
print("\n--- Method 3: Dictionary Format ---")
messages_as_dicts = [
    {"role": "system", "content": "You are a helpful customer support agent."},
    {"role": "user", "content": "I'd like to check my order status."}
]

response = model.invoke(messages_as_dicts)
print(f"Response: {response.content}")


print("\n" + "="*70)
print("PART 2: MESSAGE TYPES")
print("="*70)

# ============================================================================
# SYSTEM MESSAGE
# ============================================================================
# Sets the AI's context, persona, and rules for the entire conversation
# Always comes first in the message list

print("\n--- System Message ---")
system_instructions = SystemMessage("""
You are a support agent for Gadgetron.
Your tone should be helpful, professional, and patient.
Do not offer discounts unless the customer's product is damaged.
If you cannot answer a question, offer to escalate to a human agent.
""")

print(f"System message set: {system_instructions.content[:80]}...")


# ============================================================================
# HUMAN MESSAGE
# ============================================================================
# Represents input from the end-user

print("\n--- Human Message ---")

# Simple text query
user_query = HumanMessage("Where can I find the tracking number for my shipment?")
print(f"User query: {user_query.content}")

# With metadata (optional but useful for logging)
user_query_with_metadata = HumanMessage(
    content="My headphones arrived damaged.",
    name="customer_jane_doe",  # Optional: identify the user
    id="chat_session_987"       # Optional: unique ID for logging
)
print(f"User query with metadata: {user_query_with_metadata.content}")
print(f"  User name: {user_query_with_metadata.name}")


# ============================================================================
# AI MESSAGE
# ============================================================================
# Response from the model

print("\n--- AI Message ---")

# When you call model.invoke(), it returns an AIMessage
response = model.invoke([
    SystemMessage("You are a helpful customer support agent."),
    HumanMessage("Hello, I need help with my order.")
])

print(f"AI response type: {type(response)}")
print(f"AI response content: {response.content}")

# You can also create AIMessage manually (useful for building conversation history)
manual_ai_response = AIMessage("I can certainly help with that. What is your order number?")
print(f"Manual AI message: {manual_ai_response.content}")


# ============================================================================
# BUILDING A CONVERSATION
# ============================================================================
print("\n--- Building a Multi-Turn Conversation ---")

# Create a conversation history
conversation_history = [
    SystemMessage("You are a helpful customer support agent for Gadgetron."),
    HumanMessage("I'd like to check my order status."),
    AIMessage("Certainly, what is your order number?"),
    HumanMessage("My order number is 12345."),
    AIMessage("Let me look that up for you. Order 12345 has been shipped and is scheduled for delivery tomorrow."),
    HumanMessage("Great! What's the tracking number?")
]

# The model sees all previous messages and can reference them
response = model.invoke(conversation_history)
print(f"AI response: {response.content}")


print("\n" + "="*70)
print("PART 3: TOOL MESSAGES")
print("="*70)

# ============================================================================
# TOOL MESSAGES
# ============================================================================
# When a model calls a tool, you pass results back using ToolMessage

# Define a tool
@tool
def lookup_order(order_id: str) -> str:
    """Looks up order status in the database."""
    # Mock database
    orders = {
        "12345": "Order 12345 has been shipped and is scheduled for delivery tomorrow.",
        "67890": "Order 67890 is being processed and will ship within 2 business days.",
    }
    return orders.get(order_id, "Order not found.")


# Bind tool to model
model_with_tools = model.bind_tools([lookup_order])

print("\n--- Tool Call Flow ---")

# Step 1: User asks a question
messages = [
    SystemMessage("You are a customer support agent. Use tools to look up information."),
    HumanMessage("What's the status of order 12345?")
]

# Step 2: Model decides to call a tool
response = model_with_tools.invoke(messages)

if response.tool_calls:
    print("Model requested a tool call:")
    tool_call = response.tool_calls[0]
    print(f"  Tool: {tool_call['name']}")
    print(f"  Arguments: {tool_call['args']}")
    print(f"  ID: {tool_call['id']}")
    
    # Step 3: Execute the tool (in this standalone example, we do it manually)
    tool_result = "Order 12345 has been shipped and is scheduled for delivery tomorrow."
    
    # Step 4: Create a ToolMessage with the result
    tool_result_message = ToolMessage(
        content=tool_result,
        tool_call_id=tool_call['id']  # Must match the ID from the AIMessage
    )
    
    print(f"\nTool result: {tool_result_message.content}")
    
    # Step 5: Continue the conversation with the tool result
    messages.extend([response, tool_result_message])
    final_response = model_with_tools.invoke(messages)
    print(f"\nFinal AI response: {final_response.content}")


print("\n" + "="*70)
print("PART 4: MULTIMODAL CONTENT")
print("="*70)

# ============================================================================
# MULTIMODAL CONTENT
# ============================================================================
# Messages can contain both text and images (or other media)
# This uses the "content blocks" format

print("\n--- Multimodal Message Example ---")

# Create a message with both text and an image URL
multimodal_message = HumanMessage(content=[
    {
        "type": "text",
        "text": "My new speaker has a crack in the casing. Can I get a replacement?"
    },
    {
        "type": "image_url",
        "image_url": {
            "url": "https://example.com/image.jpg"  # In real use, this would be an actual image URL or base64
        }
    }
])

print("Created multimodal message with:")
print("  - Text: My new speaker has a crack...")
print("  - Image URL: https://example.com/image.jpg")
print("\nNote: The model can process both the text and image together!")
print("(Actual image processing requires GPT-4 Vision or similar multimodal model)")


print("\n" + "="*70)
print("DEMO COMPLETE!")
print("="*70)
print("\nKey takeaways:")
print("1. Messages are the fundamental data structure for LLM conversations")
print("2. Four message types: System, Human, AI, Tool")
print("3. SystemMessage sets the AI's behavior and rules")
print("4. HumanMessage represents user input")
print("5. AIMessage is the model's response")
print("6. ToolMessage passes tool results back to the model")
print("7. Conversation history is just a list of messages")
print("8. Messages can contain multimodal content (text + images)")
print("9. ToolMessage must include the tool_call_id from the AIMessage")
print("="*70 + "\n")

