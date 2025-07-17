# Awesome LLM Tools 🤖

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![GitHub stars](https://img.shields.io/github/stars/dr-saad-la/awesome-llm-tools.svg?style=social&label=Star)](https://github.com/your-username/awesome-llm-tools)
[![GitHub forks](https://img.shields.io/github/forks/dr-saad-la/awesome-llm-tools.svg?style=social&label=Fork)](https://github.com/your-username/awesome-llm-tools/fork)
[![Last Updated](https://img.shields.io/github/last-commit/dr-saad-la/awesome-llm-tools?label=Last%20Updated)](https://github.com/your-username/awesome-llm-tools)

A curated list of Large Language Model (LLM) tools, frameworks, and platforms for developers, researchers, and AI enthusiasts.

From commercial APIs to local deployment, from RAG systems to fine-tuning platforms - everything you need to build with LLMs.

## 📋 Table of Contents

- [Commercial LLM APIs](#-commercial-llm-apis)
- [Open Source Models](#-open-source-models)
- [Development Frameworks](#-development-frameworks)
- [Local Deployment Tools](#-local-deployment-tools)
- [Vector Databases & RAG](#-vector-databases--rag)
- [Fine-tuning Platforms](#-fine-tuning-platforms)
- [Evaluation & Monitoring](#-evaluation--monitoring)
- [Prompt Engineering](#-prompt-engineering)
- [Supporting Tools](#-supporting-tools)
- [Cost Comparison](#-cost-comparison)
- [Recommended Stacks](#-recommended-stacks)
- [Learning Resources](#-learning-resources)
- [Contributing](#-contributing)

## 🗂️ Legend

- ⭐⭐⭐⭐⭐ Must-have, industry standard
- ⭐⭐⭐⭐ Excellent choice, highly recommended
- ⭐⭐⭐ Good option, worth considering
- ⭐⭐ Decent, has specific use cases
- 💰 Free/Open Source
- 💵 Paid/Commercial
- 🔄 Freemium
- 🚀 Trending/Recently Updated
- 🏢 Enterprise-focused

---

## 🏢 Commercial LLM APIs

### **Tier 1: Leading Providers**

**[OpenAI](https://openai.com/api/)** ⭐⭐⭐⭐⭐ 💵

    - **Models**: GPT-4o, GPT-4o-mini, o1-preview, o1-mini
    - **Best for**: Production apps, complex reasoning, code generation
    - **Pricing**: $0.15-$60/1M tokens
    - **Unique**: Function calling, vision, DALL-E integration, fine-tuning

**[Anthropic Claude](https://www.anthropic.com/)** ⭐⭐⭐⭐⭐ 💵

    - **Models**: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus
    - **Best for**: Analysis, reasoning, safety-critical applications
    - **Pricing**: $0.25-$75/1M tokens
    - **Unique**: 200K context, Constitutional AI, excellent for research

**[Google Gemini](https://ai.google.dev/)** ⭐⭐⭐⭐ 💵

    - **Models**: Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini Ultra
    - **Best for**: Multimodal apps, Google ecosystem integration
    - **Pricing**: $0.075-$35/1M tokens
    - **Unique**: Video understanding, 2M token context, Google Workspace

**[Microsoft Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)** ⭐⭐⭐⭐ 💵 🏢

    - **Models**: GPT-4, GPT-3.5, DALL-E, Whisper
    - **Best for**: Enterprise applications, regulated industries
    - **Pricing**: Similar to OpenAI + Azure costs
    - **Unique**: Enterprise SLA, data residency, compliance features

### **Tier 2: Specialized & Emerging**

**[Cohere](https://cohere.ai/)** ⭐⭐⭐⭐ 💵 🏢

    - **Models**: Command R+, Embed v3, Rerank 3
    - **Best for**: Enterprise NLP, multilingual, RAG applications
    - **Pricing**: $0.15-$15/1M tokens
    - **Unique**: Excellent embeddings, enterprise focus, citation support

**[Together AI](https://www.together.ai/)** ⭐⭐⭐⭐ 💵
    - **Models**: Llama 3.1, Mixtral, Qwen, Code Llama (hosted)
    - **Best for**: Open source models as API, cost-effective scaling
    - **Pricing**: $0.2-$8/1M tokens
    - **Unique**: Multiple open source models, competitive pricing

**[Groq](https://groq.com/)** ⭐⭐⭐⭐ 💵 🚀
    - **Models**: Llama 3.1, Mixtral, Gemma
    - **Best for**: Real-time applications, speed-critical use cases
    - **Pricing**: $0.27-$2.8/1M tokens
    - **Unique**: 500+ tokens/sec inference speed, hardware optimization

**[Perplexity AI](https://www.perplexity.ai/)** ⭐⭐⭐ 💵
    - **Models**: Custom models with web search
    - **Best for**: Research applications, current information
    - **Pricing**: $20/month Pro subscription + API
    - **Unique**: Real-time web search, source citations

**[Replicate](https://replicate.com/)** ⭐⭐⭐ 💵

    - **Models**: Wide variety of open source and specialized models
    - **Best for**: Experimentation, specialized models, prototyping
    - **Pricing**: Pay-per-second usage
    - **Unique**: Model marketplace, easy deployment, diverse catalog

---

## 🔓 Open Source Models

### **Meta's Llama Family** ⭐⭐⭐⭐⭐ 💰

**[Llama 3.1 Series](https://llama.meta.com/)**
- **Sizes**: 8B, 70B, 405B parameters
- **License**: Llama 3.1 Community License (commercial friendly)
- **Best for**: General purpose, fine-tuning base, production deployments
- **Hardware**: 8B: 16GB VRAM | 70B: 80GB VRAM | 405B: 800GB+ VRAM

**[Code Llama](https://github.com/facebookresearch/codellama)**
- **Sizes**: 7B, 13B, 34B parameters
- **Best for**: Code generation, completion, explanation
- **Languages**: Python, JavaScript, Java, C++, TypeScript, Bash

### **European Excellence**

**[Mistral AI Models](https://mistral.ai/)** ⭐⭐⭐⭐ 💰
- **Models**: Mistral 7B, Mixtral 8x7B, Mixtral 8x22B
- **License**: Apache 2.0 (fully open)
- **Best for**: Production deployments, efficient inference
- **Unique**: Excellent performance per parameter, sliding window attention

### **Google's Offerings**

**[Gemma](https://ai.google.dev/gemma)** ⭐⭐⭐⭐ 💰
- **Sizes**: Gemma 2B, 7B, 9B, 27B
- **Best for**: Research, edge deployment, resource-limited environments
- **License**: Custom Gemma Terms of Use
- **Unique**: High quality training, good performance per parameter

### **Microsoft's Small Giants**

**[Phi-3](https://azure.microsoft.com/en-us/products/phi-3)** ⭐⭐⭐⭐ 💰 🚀
- **Models**: Phi-3-mini (3.8B), Phi-3-small (7B), Phi-3-medium (14B)
- **Best for**: Edge deployment, mobile applications, resource constraints
- **Unique**: Exceptional performance for size, high-quality training data

### **Specialized Models**

**[DeepSeek Coder](https://github.com/deepseek-ai/DeepSeek-Coder)** ⭐⭐⭐⭐ 💰
- **Sizes**: 1.3B, 6.7B, 33B parameters
- **Best for**: Code-focused applications, developer tools
- **Languages**: 80+ programming languages
- **License**: DeepSeek License (research friendly)

**[Yi Series](https://github.com/01-ai/Yi)** ⭐⭐⭐ 💰
- **Sizes**: 6B, 9B, 34B parameters
- **Best for**: Multilingual applications, long context tasks
- **Unique**: Strong multilingual capabilities, 200K context length

**[Qwen (Alibaba)](https://github.com/QwenLM/Qwen)** ⭐⭐⭐ 💰
- **Sizes**: 0.5B to 72B parameters
- **Best for**: Chinese language, multimodal applications
- **Unique**: Excellent Chinese support, vision capabilities

---

## 🛠️ Development Frameworks

### **High-Level Application Frameworks**

**[LangChain](https://python.langchain.com/)** ⭐⭐⭐⭐⭐ 💰
- **What it does**: Comprehensive framework for LLM applications
- **Best for**: RAG applications, complex workflows, rapid prototyping
- **Features**: 500+ integrations, memory management, agents
- **Languages**: Python, JavaScript, Go, Java

```python
# LangChain Quick Example
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = OpenAI(temperature=0.7)
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a brief explanation of {topic}"
)
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("quantum computing")
```

**[LlamaIndex](https://www.llamaindex.ai/)** ⭐⭐⭐⭐ 💰
- **What it does**: Data framework for LLM applications
- **Best for**: RAG systems, knowledge bases, document Q&A
- **Features**: 160+ data connectors, advanced indexing, query engines
- **Unique**: Data-focused, excellent for enterprise knowledge systems

**[Haystack](https://haystack.deepset.ai/)** ⭐⭐⭐⭐ 💰
- **What it does**: End-to-end NLP framework with LLM support
- **Best for**: Production applications, search systems, enterprise
- **Features**: Pipeline-based architecture, evaluation tools
- **Unique**: Production-ready, modular design, strong documentation

**[Semantic Kernel](https://github.com/microsoft/semantic-kernel)** ⭐⭐⭐ 💰 🏢
- **What it does**: Microsoft's LLM application framework
- **Best for**: Enterprise applications, Microsoft ecosystem
- **Languages**: C#, Python, Java
- **Unique**: Enterprise integration, plugin architecture

### **Low-Level Libraries**

**[Transformers (Hugging Face)](https://huggingface.co/transformers/)** ⭐⭐⭐⭐⭐ 💰
- **What it does**: Core library for transformer models
- **Best for**: Model loading, fine-tuning, research, custom implementations
- **Features**: 100K+ models, tokenizers, training utilities
- **Essential**: Foundation for most other tools

**[Guidance](https://github.com/guidance-ai/guidance)** ⭐⭐⭐ 💰 🚀
- **What it does**: Programming language for controlling language models
- **Best for**: Structured generation, constrained outputs
- **Unique**: Template-based generation, output constraints

**[Instructor](https://github.com/jxnl/instructor)** ⭐⭐⭐⭐ 💰 🚀
- **What it does**: Structured outputs from language models using Pydantic
- **Best for**: Type-safe LLM responses, data extraction
- **Languages**: Python, JavaScript
- **Unique**: Automatic retries, validation, type safety

---

## 💻 Local Deployment Tools

### **Desktop Applications**

**[Ollama](https://ollama.ai/)** ⭐⭐⭐⭐⭐ 💰
- **What it does**: Simplest way to run LLMs locally
- **Best for**: Development, offline usage, privacy-focused applications
- **Models**: Llama, Mistral, Code Llama, Phi, Gemma, and more
- **Platforms**: macOS, Linux, Windows

```bash
# Ollama Usage
curl -fsSL https://ollama.ai/install.sh | sh
ollama run llama3.1:8b
ollama run codellama:13b-code
```

**[LM Studio](https://lmstudio.ai/)** ⭐⭐⭐⭐ 💰
- **What it does**: Beautiful GUI for running LLMs locally
- **Best for**: Non-technical users, model experimentation
- **Features**: Model discovery, chat interface, local API server
- **Platforms**: macOS, Windows, Linux

**[GPT4All](https://gpt4all.io/)** ⭐⭐⭐ 💰
- **What it does**: Privacy-focused local LLM runner
- **Best for**: Privacy-conscious users, offline environments
- **Features**: No data collection, CPU and GPU support
- **Models**: Optimized models for local use

**[Jan](https://jan.ai/)** ⭐⭐⭐ 💰 🚀
- **What it does**: Open source local AI assistant
- **Best for**: Developers wanting customizable local AI
- **Features**: Plugin system, extensible architecture
- **Platforms**: Cross-platform desktop application

### **Server Deployment & High-Performance Inference**

**[vLLM](https://github.com/vllm-project/vllm)** ⭐⭐⭐⭐⭐ 💰
- **What it does**: High-performance LLM inference server
- **Best for**: Production inference, high-throughput applications
- **Features**: PagedAttention, continuous batching, streaming
- **Performance**: 24x higher throughput than HuggingFace Transformers

```python
# vLLM Example
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct")
sampling_params = SamplingParams(temperature=0.7, top_p=0.9)

prompts = ["Hello, my name is", "The capital of France is"]
outputs = llm.generate(prompts, sampling_params)
```

**[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)** ⭐⭐⭐⭐ 💰
- **What it does**: Hugging Face's production inference server
- **Best for**: Hugging Face model deployment, production serving
- **Features**: Tensor parallelism, streaming, token authentication
- **Integration**: Native HuggingFace Hub integration

**[LocalAI](https://localai.io/)** ⭐⭐⭐⭐ 💰
- **What it does**: OpenAI API compatible local server
- **Best for**: Drop-in OpenAI replacement, multi-model serving
- **Features**: OpenAI API compatibility, multiple backends
- **Models**: Support for various model formats (GGUF, ONNX, etc.)

**[FastChat](https://github.com/lm-sys/FastChat)** ⭐⭐⭐ 💰
- **What it does**: Training, serving, and evaluating LLMs
- **Best for**: Research, model comparison, conversation datasets
- **Features**: Web UI, OpenAI-compatible API, model evaluation
- **Community**: Large model leaderboard (Chatbot Arena)

**[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** ⭐⭐⭐⭐ 💰 🏢
- **What it does**: NVIDIA's optimized LLM inference library
- **Best for**: NVIDIA GPU deployments, maximum performance
- **Features**: FP16/INT8/INT4 quantization, in-flight batching
- **Performance**: Up to 8x speedup on NVIDIA GPUs

---

## 🗂️ Vector Databases & RAG

### **Vector Databases**

**[Pinecone](https://www.pinecone.io/)** ⭐⭐⭐⭐ 💵
- **What it does**: Fully managed vector database service
- **Best for**: Production RAG applications, scaling to millions of vectors
- **Pricing**: $70/month starter + usage-based scaling
- **Features**: Real-time updates, metadata filtering, hybrid search

**[Chroma](https://www.trychroma.com/)** ⭐⭐⭐⭐⭐ 💰
- **What it does**: AI-native open source embedding database
- **Best for**: Development, self-hosted applications, getting started
- **Features**: Simple Python API, local and cloud deployment
- **Integration**: Excellent LangChain and LlamaIndex support

```python
# Chroma Example
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

collection.add(
    documents=["This is document 1", "This is document 2"],
    metadatas=[{"source": "web"}, {"source": "book"}],
    ids=["doc1", "doc2"]
)

results = collection.query(
    query_texts=["document about AI"],
    n_results=2
)
```

**[Weaviate](https://weaviate.io/)** ⭐⭐⭐⭐ 💰
- **What it does**: Open source vector database with built-in ML
- **Best for**: Complex search applications, hybrid search
- **Features**: GraphQL API, vectorization modules, multi-tenancy
- **Unique**: Built-in vectorization, knowledge graphs

**[Qdrant](https://qdrant.tech/)** ⭐⭐⭐⭐ 💰
- **What it does**: High-performance vector similarity search engine
- **Best for**: High-performance applications, complex filtering
- **Features**: Payload-based filtering, distributed deployment, clustering
- **Performance**: Written in Rust, excellent speed and memory efficiency

**[Milvus](https://milvus.io/)** ⭐⭐⭐ 💰 🏢
- **What it does**: Cloud-native vector database
- **Best for**: Large-scale deployments, enterprise environments
- **Features**: Horizontal scaling, cloud-native architecture
- **Deployment**: Kubernetes-native, cloud and on-premises

**[FAISS](https://github.com/facebookresearch/faiss)** ⭐⭐⭐⭐ 💰
- **What it does**: Facebook's library for efficient similarity search
- **Best for**: Research applications, custom implementations
- **Features**: GPU acceleration, billion-scale search, multiple algorithms
- **Note**: Lower-level, requires more setup than databases above

### **RAG Enhancement Tools**

**[LlamaHub](https://llamahub.ai/)** ⭐⭐⭐⭐ 💰
- **What it does**: Data connectors and tools for LlamaIndex
- **Features**: 100+ data connectors (Notion, Slack, Google Drive, etc.)
- **Best for**: Connecting diver
