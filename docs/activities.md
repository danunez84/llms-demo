# Activities

This section contains hands-on activities for practicing prompting techniques and LLM application development.

> **Note:** Activity files are available in the [GitHub repository](https://github.com/gperdrizet/llms-demo/tree/main/activities) and include detailed instructions and starter code.

## Activity 1: LLM word problems

Practice basic prompting techniques and chain-of-thought reasoning by solving word problems with an LLM using the Gradio web interface.

**Duration:** 30-45 minutes

**Skills practiced:**
- Basic prompting strategies
- Chain-of-thought reasoning
- System prompt experimentation
- Comparing different prompting approaches

**Prerequisites:**
- Completed [Quickstart](quickstart.md) setup
- Familiarity with Lesson 46 (Prompting fundamentals)

**What you'll use:**
- Gradio chatbot web interface (`demos/chatbots/gradio_chatbot.py`)
- System prompt customization

**Location:** `activities/activity_1_word_problems.md`

## Activity 2: Text summarization

Build a practical text summarization script applying various prompting techniques.

**Duration:** 45-60 minutes

**Skills practiced:**
- Text preprocessing and chunking
- Prompt engineering for summarization
- Handling long documents
- Iterative refinement

**Prerequisites:**
- Completed Activity 1
- Python programming experience
- Understanding of basic file I/O

**Location:** `activities/activity_2_text_summarization.md`

## Activity 3: Building LangChain chains

Build practical LangChain applications using prompt templates, output parsers, and chains.

**Duration:** 60-75 minutes

**Skills practiced:**
- Creating reusable prompt templates with variables
- Structured data extraction with Pydantic schemas
- JSON parsing with JsonOutputParser
- Composing multi-step chains
- Error handling and debugging chains

**Prerequisites:**
- Completed Activities 1 and 2
- Python programming experience
- Understanding of Lesson 48 (LangChain basics)
- Familiarity with Demo 5 (LangChain demo)

**What you'll do:**
- Build a template-based translator chain
- Create a structured book information extractor
- Compose a multi-step review analysis pipeline
- Learn debugging and best practices

**Location:** `activities/activity_3_langchain_chains.md`

## Activity 4: Extending the ReAct agent

Enhance the ReAct agent chatbot by adding custom tools and testing multi-step reasoning.

**Duration:** 45-60 minutes

**Skills practiced:**
- Creating LangChain tools with the `@tool` decorator
- Understanding agent decision-making and tool selection
- Debugging tool execution and agent behavior
- Multi-step problem solving with tool chaining

**Prerequisites:**
- Completed Activities 1-3
- Python programming experience
- Understanding of Lesson 49 (LangChain advanced - agents)
- Familiarity with the ReAct agent demo

**What you'll do:**
- Study existing tool implementations
- Create a new tool (temperature converter, text analyzer, or custom)
- Register your tool with the agent
- Test single-tool and multi-tool reasoning
- Debug and improve your implementation

**Location:** `activities/activity_4_react_agent_tools.md`

## Activity 5: Extending the RAG knowledge system

Add a new document source to the RAG demo by implementing the `BaseIngestor` interface.

**Duration:** 45-60 minutes

**Skills practiced:**
- Implementing a Python interface (`BaseIngestor`)
- Using LangChain document loaders (`WebBaseLoader`)
- Chunking documents with `RecursiveCharacterTextSplitter`
- Registering new components in a running Gradio application

**Prerequisites:**
- Completed Activities 1-4
- Python programming experience
- Understanding of Lesson 49 (LangChain advanced - RAG pipeline)
- Familiarity with the RAG system demo

**What you'll do:**
- Explore the existing `WikipediaIngestor` and `BaseIngestor` interface
- Implement `URLIngestor` using `WebBaseLoader`
- Register it in the demo and verify it appears in the UI

**Location:** `activities/activity_5_rag_sources.md`


## Activity 6: LoRA fine-tuning

Use Low Rank Adaptation to fine tune a small qwen model

**Duration:** 45-60 minutes

**Skills practiced:**
- Direct loading/prompting models (`transformers`)
- Creating SFT datasets (ChatML format, `Dataset`)
- LoRA fine-tuning (`SFTTrainer`)
- Model evaluation

**Prerequisites:**
- Completed Activities 1-5
- Python programming experience
- Understanding of Lesson 50 (Fine-tuning)
- Familiarity with model prompting and training in general

**What you'll do:**
- Prompt a base model and observe it's behavior
- Create a small supervised fine-tuning dataset
- Fine-tune the base model using LoRA
- Compare the base and fine-tuned models

**Location:** `activities/activity_6_finetuning.md`

