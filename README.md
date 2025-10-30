# LangChain Learning Repository

A comprehensive, hands-on tutorial series for learning LangChain v1.0 and LangGraph from the ground up. This repository contains 12 complete, working examples that progress from basic concepts to advanced multi-agent systems.

## About This Repository

This project provides practical, executable code examples for developers learning to build AI applications with LangChain v1.0. Each tutorial is self-contained in its own folder with a single Python file, complete with detailed comments and explanations.

**Key Features:**
- Built for LangChain v1.0 (latest patterns, no deprecated code)
- Progressive learning path from basics to advanced topics
- Complete working examples, not just snippets
- Detailed inline documentation
- Production-ready patterns and best practices

## What You'll Learn

### Core Concepts (Tutorials 0-5)
- **0-quickstart**: Build your first agent with tools and memory
- **1-agents**: Dynamic models, middleware, and error handling
- **2-models**: Invocation methods, tool calling, and structured outputs
- **3-messages**: Conversation management and message types
- **4-tools**: Creating tools with runtime context access
- **5-memory**: Short-term memory, state management, and summarization

### Production Tools (Tutorial 6)
- **6-langsmith**: Tracing, debugging, and monitoring with LangSmith

### RAG Systems (Tutorials 7-10)
- **7-document-loaders**: PDF processing, text splitting, and vector stores
- **8-retrieval**: Advanced querying techniques (MMR, filtering, thresholds)
- **9-two-step-rag**: Fixed retrieval-then-generation pattern
- **10-agentic-rag**: Agent-controlled retrieval decisions

### Advanced Systems (Tutorial 11)
- **11-multi-agent**: Supervisor pattern with specialized sub-agents

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Basic Python knowledge

## Installation

1. Clone this repository:
```bash
git clone https://github.com/eriperspective/langchain.git
cd langchain
```

2. Install dependencies:
```bash
pip install langchain langchain-openai langchain-community langgraph pypdf langsmith
```

3. Set your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

4. Optional - Enable LangSmith tracing:
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGSMITH_PROJECT="langchain-learning"
```

## Usage

Each tutorial is independent and can be run directly:

```bash
# Start with the basics
python 0-quickstart/quickstart_demo.py

# Or jump to a specific topic
python 5-memory/memory_demo.py
```

**Recommended Learning Path:**
1. Start with tutorials 0-5 in order to build foundational knowledge
2. Complete tutorial 6 to set up debugging tools
3. Work through tutorials 7-10 for RAG systems
4. Finish with tutorial 11 for multi-agent architectures

## Repository Structure

```
langchain/
├── 0-quickstart/          # Simple and advanced agent examples
├── 1-agents/              # Agent configuration and middleware
├── 2-models/              # Working with language models
├── 3-messages/            # Message types and conversations
├── 4-tools/               # Tool creation and context
├── 5-memory/              # State management and memory
├── 6-langsmith/           # Tracing and debugging
├── 7-document-loaders/    # Document processing and embeddings
├── 8-retrieval/           # Vector database querying
├── 9-two-step-rag/        # Basic RAG implementation
├── 10-agentic-rag/        # Advanced RAG with autonomy
├── 11-multi-agent/        # Multi-agent coordination
└── README.md
```

## Key Concepts Covered

- **Agent Creation**: Using `create_agent()` for standard and custom agents
- **Tool Development**: Building tools with `@tool` decorator and runtime context
- **Memory Management**: Checkpointers, state schemas, and conversation history
- **RAG Patterns**: 2-step vs agentic retrieval strategies
- **Multi-Agent Systems**: Supervisor pattern with tool calling
- **Production Practices**: LangSmith integration, error handling, middleware

## Technologies

- **LangChain v1.0+** - Core framework
- **LangGraph** - Agent workflow orchestration
- **OpenAI GPT-4o-mini** - Language model
- **ChromaDB** - Vector database (tutorials 7-10)
- **LangSmith** - Tracing and monitoring (tutorial 6)

## Notes

- All code uses LangChain v1.0 patterns (no deprecated LCEL or legacy chains)
- ChromaDB requires C++ build tools - RAG tutorials may need additional setup
- Each tutorial includes detailed comments explaining concepts and decisions
- Examples use GPT-4o-mini to minimize API costs during learning

## Documentation References

- [LangChain Documentation](https://docs.langchain.com/oss/python/langchain)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph)
- [LangSmith Platform](https://smith.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)

## Contributing

This is a learning resource repository. If you find issues or have suggestions for improvements, please open an issue.

## License

This project is provided for educational purposes.

## Author

Created as a comprehensive learning resource for developers entering the LangChain ecosystem.
