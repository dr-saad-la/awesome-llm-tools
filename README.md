<!-- # Awesome LLM Tools 🤖 -->

![Awesome LLM Tools](https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=200&section=header&text=Awesome%20LLM%20Tools%20🤖&fontSize=60&fontColor=ffffff&animation=fadeIn&fontAlign=50&fontAlignY=35&desc=A%20curated%20list%20of%20LLM%20tools%20and%20platforms&descAlign=50&descAlignY=60&descSize=18)

---

<!--suppress HtmlDeprecatedAttribute -->
<div align="center">

[![Awesome](https://img.shields.io/badge/Awesome-List-ff69b4?style=for-the-badge&logo=awesomelists&logoColor=white)](https://awesome.re)
[![GitHub stars](https://img.shields.io/github/stars/dr-saad-la/awesome-llm-tools?style=for-the-badge&logo=github&color=yellow)](https://github.com/dr-saad-la/awesome-llm-tools)
[![GitHub forks](https://img.shields.io/github/forks/dr-saad-la/awesome-llm-tools?style=for-the-badge&logo=github&color=blue)](https://github.com/dr-saad-la/awesome-llm-tools/fork)
![GitHub contributors](https://img.shields.io/github/contributors/dr-saad-la/awesome-llm-tools?style=for-the-badge&logo=github&color=green)
[![Last Updated](https://img.shields.io/github/last-commit/dr-saad-la/awesome-llm-tools?style=for-the-badge&logo=github&label=Last%20Updated&color=green)](https://github.com/dr-saad-la/awesome-llm-tools)

</div>

<!-- ![Awesome LLM Tools](https://img.shields.io/badge/🤖%20Awesome%20LLM%20Tools-A%20curated%20list%20of%20LLM%20platforms%20and%20frameworks-blue?style=for-the-badge&logo=openai&logoColor=white&color=gradient) -->

A curated list of Large Language Model (LLM) tools, frameworks, and platforms
for developers, researchers, and AI enthusiasts.

From commercial APIs to local deployment, from RAG systems to fine-tuning
platforms - everything you need to build with LLMs.

## 📋 Table of Contents

- [Commercial LLM APIs](#-commercial-llm-apis)
- [Open Source Models](#-open-source-models)
- [Development Frameworks](#-development-frameworks)
- [Agent Frameworks & Multi-Agent Systems](#-agent-frameworks--multi-agent-systems)
- [Structured Generation & Control](#-structured-generation--control)
- [Local Deployment Tools](#-local-deployment-tools)
- [Self-hosted & Local Platforms](#-self-hosted--local-platforms)
- [Visual Workflow Builders](#-visual-workflow-builders)
- [Vector Databases & RAG](#-vector-databases--rag)
- [Memory & Persistence](#-memory--persistence)
- [Voice & Audio Tools](#-voice--audio-tools)
- [Multimodal & Specialized Models](#-multimodal--specialized-models)
- [Code-Focused LLM Tools](#-code-focused-llm-tools)
- [Mobile & Edge Deployment](#-mobile--edge-deployment)
- [Integration & Utility Tools](#-integration--utility-tools)
- [Fine-tuning Platforms](#-fine-tuning-platforms)
- [LLM Gateway & Operations](#-llm-gateway--operations)
- [Evaluation & Monitoring](#-evaluation--monitoring)
- [Research & Evaluation Tools](#-research--evaluation-tools)
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

### Tier 1: Leading Providers

**[OpenAI](https://openai.com/api/)** ⭐⭐⭐⭐⭐ 💵

- **Models**: GPT-5.4, GPT-5.4 Mini, GPT-5.4 Nano, o3, o4-mini
- **Best for**: Production apps, complex reasoning, code generation,
  computer-use agents
- **Pricing**: $0.05-$180/1M tokens
- **Unique**: Unified reasoning and coding in GPT-5.4, 1M context, Responses
  API, open-weight GPT-OSS

**[Anthropic Claude](https://www.anthropic.com/)** ⭐⭐⭐⭐⭐ 💵

- **Models**: Claude Opus 4.6, Claude Sonnet 4.6, Claude Haiku 4.5
- **Best for**: Agentic workflows, long-context analysis, safety-critical
  applications
- **Pricing**: $1.00-$25/1M tokens
- **Unique**: 91.3% on GPQA Diamond, Adaptive Thinking, 1M context, 67% price
  drop from previous generation

**[Google Gemini](https://ai.google.dev/)** ⭐⭐⭐⭐ 💵

- **Models**: Gemini 3.1 Pro, Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.5
  Flash-Lite
- **Best for**: Multimodal apps, cost-sensitive workloads, Google ecosystem
  integration
- **Pricing**: $0.10-$18/1M tokens
- **Unique**: Ranked #1 on Intelligence Index, aggressive Flash-Lite pricing, 1M
  context, free tier available

**[Microsoft Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)**
⭐⭐⭐⭐ 💵 🏢

- **Models**: GPT-5.4, GPT-4.1, o3, o4-mini, DeepSeek, Llama 4, Mistral, Claude
  (11,000+ total)
- **Best for**: Enterprise applications, regulated industries, high-volume
  production workloads
- **Pricing**: Same as OpenAI direct, typically 15-40% higher at scale with
  infrastructure overhead
- **Unique**: Provisioned throughput units (PTUs), Microsoft Foundry Agent
  Service, MCP support, enterprise compliance

**[xAI Grok](https://x.ai/api)** ⭐⭐⭐⭐⭐ 💵 🚀

- **Models**: Grok 4, Grok 4.1 Fast, Grok 3
- **Best for**: Long-context reasoning, real-time information, cost-sensitive
  production workloads
- **Pricing**: $0.20-$3/1M tokens
- **Unique**: Industry-leading 2M token context window, built-in real-time web
  search, top intelligence benchmark scores

**[Amazon Bedrock](https://aws.amazon.com/bedrock/)** ⭐⭐⭐⭐ 💵 🏢

- **Models**: Claude, Llama 4, Mistral, Cohere Command, Amazon Nova (unified
  catalog)
- **Best for**: Enterprise AWS workloads, multi-model strategies, regulated
  industries
- **Pricing**: Per-token rates vary by model, batch mode at 50% discount
- **Unique**: Single API across all major model families, AWS security and
  compliance, Bedrock Agents for RAG and multi-agent workflows

### Tier 2: Specialized & Emerging

**[Cohere](https://cohere.ai/)** ⭐⭐⭐⭐ 💵 🏢

- **Models**: Command A, Command R+, Command R7B, Embed 4, Rerank 4
- **Best for**: Enterprise RAG pipelines, data sovereignty deployments,
  multilingual NLP
- **Pricing**: $0.04-$10/1M tokens
- **Unique**: Full RAG stack in one provider, private VPC and on-premises
  deployment, EU and government compliance

**[Together AI](https://www.together.ai/)** ⭐⭐⭐⭐ 💵 🚀

- **Models**: Llama 4, DeepSeek R1, Qwen 3, Gemma 3, and 200+ open-source models
- **Best for**: Open-source model hosting, cost-sensitive scaling, fine-tuning
  and custom deployment
- **Pricing**: $0.02-$7/1M tokens
- **Unique**: Largest hosted open-source model catalog, dedicated GPU clusters,
  fine-tuning included at no extra cost

**[Groq](https://groq.com/)** ⭐⭐⭐⭐ 💵 🚀

- **Models**: Llama 3.3 70B, GPT-OSS 120B, Llama 3.1 8B, Gemma 3
- **Best for**: Ultra-low latency applications, real-time agents, speed-critical
  inference
- **Pricing**: $0.05-$0.79/1M tokens
- **Unique**: Custom LPU hardware delivering 840-1000 tokens/sec, now under
  Nvidia following 2025 acquisition

**[Mistral API AI](https://console.mistral.ai/)** ⭐⭐⭐⭐ 💵 🚀

- **Models**: Mistral Large 3, Mistral Small 4, Devstral 2, Magistral, Mistral
  Nemo
- **Best for**: European data sovereignty, multilingual applications,
  cost-effective frontier-quality inference
- **Pricing**: $0.02-$6/1M tokens
- **Unique**: Apache 2.0 open-weight models for self-hosting, 40% cheaper output
  than comparable tiers, GDPR-native EU infrastructure

**[DeepSeek](https://api-docs.deepseek.com/)** ⭐⭐⭐⭐ 💵 🚀

- **Models**: DeepSeek V3.2, DeepSeek R1
- **Best for**: Cost-sensitive production workloads, high-volume batch
  processing, reasoning tasks
- **Pricing**: $0.03-$0.42/1M tokens (90% discount with automatic context
  caching)
- **Unique**: Lowest-cost frontier-class inference available, automatic caching
  with no developer effort required

**[Cerebras](https://cerebras.ai/)** ⭐⭐⭐⭐ 💵 🚀

- **Models**: Llama 3.3 70B, Llama 3.1 8B, GPT-OSS 120B
- **Best for**: Latency-critical agents, real-time voice applications, streaming
  use cases
- **Pricing**: $0.10-$0.60/1M tokens
- **Unique**: Wafer Scale Engine delivers 2300+ tokens/sec, the fastest
  inference hardware available, AWS Bedrock integration coming 2026

**[Perplexity AI](https://www.perplexity.ai/)** ⭐⭐⭐ 💵 🚀

- **Models**: Sonar, Sonar Pro, Sonar Deep Research
- **Best for**: Research applications, fact-grounded responses, real-time
  information retrieval
- **Pricing**: $1-$15/1M tokens, Search API at $5/1K requests
- **Unique**: Built-in 200B+ page web index with citations baked into every
  response, Agentic Research API for multistep investigation

**[Fireworks AI](https://fireworks.ai/)** ⭐⭐⭐⭐ 💵

- **Models**: DeepSeek V3, Llama 4, Qwen 3, and 200+ models via FireAttention
  engine
- **Best for**: Production inference with custom fine-tuned models,
  latency-sensitive applications
- **Pricing**: $0.10-$1.68/1M tokens
- **Unique**: Fine-tuned models served at the same price as base models, 4x
  throughput improvement via FireAttention, SOC 2 and HIPAA-compliant

**[Replicate](https://replicate.com/)** ⭐⭐⭐ 💵

- **Models**: 9000+ community and official models spanning LLMs, image, video,
  and audio
- **Best for**: Rapid prototyping, multi-modal experimentation, one-line access
  to any model type
- **Pricing**: Per-second GPU billing from $0.000100/sec (CPU) to $0.001400/sec
  (A100)
- **Unique**: Broadest multi-modal model marketplace available, now integrated
  into Cloudflare Workers for edge deployment

---

## 🔓 Open Source Models

### Meta Llama

**[Llama 4](https://llama.meta.com/)** ⭐⭐⭐⭐⭐ 💰 🚀

- **Models**: Llama 4 Scout (109B total / 17B active, 16 experts, 10M context),
  Llama 4 Maverick (400B total / 17B active, 128 experts, 1M context)
- **Best for**: Multimodal applications, long-context reasoning, production
  deployments on cloud platforms
- **License**: Llama Community License (commercial friendly with 700M MAU
  threshold; not OSI-approved)
- **Hardware**: Scout: ~109 GB FP8 (1×H100 INT4, or Mac M4 Max 128 GB at Q4);
  Maverick: ~400 GB FP8 (8×H100/H200 node)
- **Unique**: Native MoE architecture with early-fusion multimodality (text +
  image); 12-language support; available on HuggingFace, Ollama, and all
  major cloud platforms

**[Llama 3.x Series](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)** ⭐⭐⭐⭐⭐ 💰

- **Models**: Llama 3.1 (8B/70B/405B, 128K context), Llama 3.2 (1B/3B edge;
  11B/90B vision), Llama 3.3 70B Instruct
- **Best for**: Local deployment, fine-tuning base, production workloads where
  Llama 4 VRAM requirements are prohibitive
- **License**: Llama Community License
- **Hardware**: 8B: ~5 GB quantized (any consumer GPU); 70B: ~35–40 GB
  quantized
- **Unique**: Widest ecosystem of derivative models, tools, and deployment
  guides of any open-weight family

> **Note — Code Llama (legacy)**: Based on Llama 2 architecture, last updated
> January 2024, and effectively superseded by Llama 3.x and specialist coding
> models. Weights remain on HuggingFace but are no longer recommended for new
> projects.

---

### Qwen (Alibaba)

**[Qwen3 / Qwen3.5](https://github.com/QwenLM/Qwen3)** ⭐⭐⭐⭐⭐ 💰 🚀

- **Models**: Qwen3 (0.6B–235B-A22B MoE, 128K context); Qwen3.5 (0.8B–397B-A17B
  MoE, 262K native context extensible to 1M+)
- **Best for**: Multilingual applications (119–201 languages), reasoning with
  hybrid thinking/non-thinking mode, cost-efficient inference
- **License**: Apache 2.0
- **Hardware**: 0.8B–9B: consumer GPU or mobile; 27B: RTX 4090; 35B-A3B MoE:
  Mac 32 GB; 397B flagship: multi-GPU cluster
- **Unique**: 700M+ cumulative HuggingFace downloads; 200,000+ derivative
  models; broadest specialist ecosystem (Qwen3-Coder-480B, QwQ-32B reasoning,
  Qwen2.5-VL vision, Qwen3-Omni audio/video/text)

---

### DeepSeek

**[DeepSeek V3.2 / R1](https://github.com/deepseek-ai/DeepSeek-V3)** ⭐⭐⭐⭐⭐ 💰 🚀

- **Models**: DeepSeek-V3.2 (671B total / 37B active MoE); DeepSeek-R1
  reasoning model; R1 distilled variants (1.5B–70B based on Qwen2.5 and
  Llama 3)
- **Best for**: Frontier reasoning and coding tasks, cost-sensitive
  high-performance inference, agentic workflows
- **License**: MIT License
- **Hardware**: Full 671B: 8×H100/H200 (~1.3 TB BF16); R1-Distill-32B: 1×RTX
  4090 (~20 GB Q4); R1-Distill-70B: ~43 GB Q4
- **Unique**: V3.2 adds "thinking in tool-use" and large-scale agentic RL
  across 1,800+ environments; R1-Distill-32B outperforms OpenAI o1-mini on
  most benchmarks; 82M+ Ollama downloads

> **Note — DeepSeek Coder (legacy)**: The standalone DeepSeek-Coder line was
> merged into the V3 main line as of V2.5 (September 2024). Use DeepSeek-V3.2
> or R1 distills for coding tasks.

---

### Mistral AI

**[Mistral Large 3](https://mistral.ai/)** ⭐⭐⭐⭐⭐ 💰 🚀

- **Models**: Mistral Large 3 (675B total / 41B active MoE, 2.5B vision
  encoder, 256K context)
- **Best for**: European data sovereignty, frontier-class vision-language tasks,
  long-context analysis
- **License**: Apache 2.0
- **Hardware**: 8×H200 FP8 or 4–8×H100 NVFP4
- **Unique**: Largest fully open-weight Apache 2.0 MoE available; vision
  encoder included; GDPR-native EU infrastructure

**[Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4)** ⭐⭐⭐⭐ 💰 🚀

- **Models**: Mistral Small 4 (119B total / 6.5B active, 128 experts, 256K
  context)
- **Best for**: Efficient inference combining reasoning, vision, and coding in
  a single model
- **License**: Apache 2.0
- **Hardware**: 4×H100 minimum; quantized options available
- **Unique**: Unifies reasoning (Magistral), vision (Pixtral), and coding
  (Devstral) with adjustable `reasoning_effort` parameter

**[Ministral 3](https://huggingface.co/collections/mistralai/ministral-3b-671ac9b1c979d08d50f83240)** ⭐⭐⭐⭐ 💰

- **Models**: 3B, 8B, 14B dense variants; each with Base/Instruct/Reasoning
  flavors and vision support
- **Best for**: Edge and on-device deployment, resource-constrained environments
- **License**: Apache 2.0
- **Hardware**: 3B: 6–8 GB; 14B: fits quantized on a single 24 GB GPU
- **Unique**: Full reasoning and vision capabilities down to 3B scale; includes
  Devstral Small 2 (24B) for agentic coding at 68% SWE-bench Verified

---

### Google Gemma

**[Gemma 4](https://ai.google.dev/gemma)** ⭐⭐⭐⭐⭐ 💰 🚀

- **Models**: Gemma 4 E2B, E4B (edge, 128K); Gemma 4 26B MoE (3.8B active,
  256K); Gemma 4 31B Dense (256K, #3 Arena AI)
- **Best for**: Edge deployment, research, multimodal applications requiring
  text, image, video, and audio input
- **License**: Apache 2.0 (upgraded from prior restrictive custom license)
- **Hardware**: E2B/E4B: ~2 GB RAM (phones); 31B Dense: 1×80 GB H100 BF16;
  RTX 3090+ with quantization
- **Unique**: AIME math score jumped from 20.8% to 89.2% vs Gemma 3; broadest
  specialized variant ecosystem (PaliGemma 2, MedGemma, ShieldGemma 2,
  TranslateGemma, TxGemma); 400M+ total downloads

> **Gemma 3** (1B/4B/12B/27B, 128K context, 140+ languages) remains widely
> deployed and is a solid choice for teams not yet on Gemma 4.

---

### Microsoft Phi

**[Phi-4 Family](https://huggingface.co/collections/microsoft/phi-4)** ⭐⭐⭐⭐⭐ 💰 🚀

- **Models**: Phi-4 (14B), Phi-4-mini (3.8B, 128K), Phi-4-multimodal (5.6B,
  text + vision + audio), Phi-4-reasoning / reasoning-plus (14B chain-of-thought),
  Phi-4-reasoning-vision (15B)
- **Best for**: Mobile and edge deployment, resource-constrained environments,
  reasoning at small scale
- **License**: MIT License
- **Hardware**: Phi-4-mini: runs on iPhones, any GPU with 6+ GB; Phi-4 14B:
  ~6–9 GB Q4 on RTX 3090/4090
- **Unique**: Phi-4-reasoning-plus outperforms DeepSeek-R1-Distill-Llama-70B
  at 5× smaller size; Phi-4-multimodal uses Mixture-of-LoRAs for simultaneous
  text, vision, and audio; gold standard for efficient small models

> **Note — Phi-3 (superseded)**: Phi-3 has been fully superseded by the Phi-4
> family. Weights remain available but new projects should use Phi-4.

---

### New Entrants Worth Tracking

**[NVIDIA Nemotron 3](https://huggingface.co/nvidia)** ⭐⭐⭐⭐ 💰 🚀

- **Models**: Nemotron 3 Nano (31.6B total / 3.6B active, 1M context); Nemotron
  3 Super (120B total / 12B active)
- **Best for**: Long-context tasks, deep research workflows, efficient inference
  on NVIDIA hardware
- **License**: NVIDIA Open Model License (permissive)
- **Hardware**: Nano: single consumer GPU; Super: multi-GPU setup
- **Unique**: Hybrid Mamba-Transformer MoE architecture; Super ranks #1 on
  DeepResearch Bench; available on HuggingFace and major inference platforms

**[OpenAI gpt-oss](https://huggingface.co/openai)** ⭐⭐⭐⭐ 💰 🚀

- **Models**: gpt-oss 120B, gpt-oss 20B
- **Best for**: Reasoning tasks, tool use, deployments requiring near o4-mini
  capability with full weight access
- **License**: Apache 2.0
- **Hardware**: 20B fits in 16 GB memory; 120B requires multi-GPU setup
- **Unique**: OpenAI's first open-weight release since GPT-2; near-parity with
  o4-mini on reasoning tasks; strong tool-use performance

**[AI2 OLMo 3](https://allenai.org/olmo)** ⭐⭐⭐⭐ 💰

- **Models**: OLMo 3 (1B–32B), OLMo 3 32B Thinking (chain-of-thought)
- **Best for**: Reproducible research, academic benchmarking, fully auditable
  training pipelines
- **License**: Apache 2.0 (weights, training data, code, and recipes all open)
- **Hardware**: 1B–7B: consumer GPU; 32B: ~20 GB Q4
- **Unique**: Gold standard for fully open models — the only family where every
  training artifact is publicly released; 2.5× more training-efficient than
  Llama 3.1; first fully open model to beat GPT-3.5 Turbo

**[Zhipu GLM-5](https://github.com/THUDM/GLM)** ⭐⭐⭐⭐ 💰 🚀

- **Models**: GLM-5 (744B total / 40B active MoE); GLM-5.1 (April 2026)
- **Best for**: Frontier coding and reasoning tasks, research into non-NVIDIA
  training infrastructure
- **License**: MIT License
- **Hardware**: Multi-GPU cluster required for full model; distilled variants
  available for local use
- **Unique**: Trained entirely on Huawei Ascend chips (no NVIDIA GPUs);
  GLM-5.1 reached #1 on SWE-Bench Pro; geopolitically significant demonstration
  of NVIDIA-independent frontier training

**[Falcon 3](https://huggingface.co/tiiuae)** ⭐⭐⭐⭐ 💰

- **Models**: 1B, 3B, 7B, 10B dense variants; includes a Mamba variant
- **Best for**: Efficient on-device and on-premises deployment, sub-13B
  performance
- **License**: Apache 2.0-based (TII Falcon License)
- **Hardware**: 1B–3B: mobile and edge; 10B: single consumer GPU
- **Unique**: Falcon3-10B outperformed Llama 3.1 8B and Qwen 2.5 7B at launch;
  Mamba variant available for non-attention architectures

---

### Discontinued / Legacy Models

> The following models are no longer actively developed. Weights remain
> available but new projects should migrate to current alternatives.

- **Yi Series** (01.AI) — 01.AI stopped pre-training LLMs in March 2025 and
  pivoted to business solutions built on DeepSeek. Last release: Yi-1.5
  (May 2024). Community has migrated to Qwen and DeepSeek.
- **Code Llama** — Superseded by Llama 3.x and specialist coding models. See
  note above.
- **DeepSeek Coder** — Merged into DeepSeek V3 main line as of V2.5. See note
  above.
- **Phi-3** — Fully superseded by Phi-4. See note above.
- **DBRX** (Databricks) — Deprecated April 2025.

---

## 🛠️ Development Frameworks

### High-Level Application Frameworks

**[LangChain](https://python.langchain.com/)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: Comprehensive framework for LLM applications
- **Best for**: RAG applications, complex workflows, rapid prototyping
- **Features**: 500+ integrations, memory management, agents
- **Languages**: Python, JavaScript, Go, Java

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

### Low-Level Libraries

**[Transformers (Hugging Face)](https://huggingface.co/transformers/)**
⭐⭐⭐⭐⭐ 💰

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

### Desktop Applications

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

### Server Deployment & High-Performance Inference

**[vLLM](https://github.com/vllm-project/vllm)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: High-performance LLM inference server
- **Best for**: Production inference, high-throughput applications
- **Features**: PagedAttention, continuous batching, streaming
- **Performance**: 24x higher throughput than HuggingFace Transformers

**[Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)**
⭐⭐⭐⭐ 💰

- **What it does**: Hugging Face's production inference server
- **Best for**: Hugging Face model deployment, production serving
- **Features**: Tensor parallelism, streaming, token authentication
- **Integration**: Native HuggingFace Hub integration

**[LocalAI](https://localai.io/)** ⭐⭐⭐⭐ 💰

- **What it does**: OpenAI API compatible local server
- **Best for**: Drop-in OpenAI replacement, multimodel serving
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

### Vector Databases

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

### RAG Enhancement Tools

**[LlamaHub](https://llamahub.ai/)** ⭐⭐⭐⭐ 💰

- **What it does**: Data connectors and tools for LlamaIndex
- **Best for**: Connecting diverse data sources to LlamaIndex applications.
- **Features**: 100+ data connectors (Notion, Slack, Google Drive, databases, and more)

---

## 🤖 Agent Frameworks & Multi-Agent Systems

### Advanced Agent Orchestration

**[LangGraph](https://github.com/langchain-ai/langgraph)** ⭐⭐⭐⭐⭐ 💰 🚀

- **What it does**: Build stateful, multi-actor applications with LLMs using
  graph-based workflows
- **Best for**: Complex agent workflows, multistep reasoning, human-in-the-loop
  systems
- **Features**: State management, branching logic, human approval nodes,
  persistence
- **Integration**: LangChain native, part of LangChain ecosystem

**[CrewAI](https://github.com/joaomdmoura/crewAI)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: Framework for orchestrating role-playing, autonomous AI
  agents
- **Best for**: Multi-agent collaboration, complex task delegation, team-based
  AI workflows
- **Features**: Role-based agents, task delegation, collaborative problem-solving
- **Use cases**: Content creation teams, research groups, business process
  automation

**[AutoGen](https://github.com/microsoft/autogen)** ⭐⭐⭐⭐ 💰

- **What it does**: Microsoft's framework for multi-agent conversation and
  collaboration
- **Best for**: Multi-agent conversations, code generation, complex problem-solving
- **Features**: Conversable agents, human-in-the-loop, code execution, group
  chat
- **Developer**: Microsoft Research

**[TaskWeaver](https://github.com/microsoft/TaskWeaver)** ⭐⭐⭐ 💰 🚀

- **What it does**: Code-first agent framework for data analytics and processing
- **Best for**: Data analysis, code generation, analytical workflows
- **Features**: Code interpreter, plugin system, stateful conversations
- **Unique**: Code-first approach, data analysis focused

**[AgentGPT](https://github.com/reworkd/AgentGPT)** ⭐⭐⭐ 💰

- **What it does**: Autonomous AI agent platform that runs in browser
- **Best for**: Goal-oriented tasks, autonomous execution, research
- **Features**: Web-based interface, autonomous goal pursuit, task breakdown
- **Deployment**: Web app, self-hostable

**[SuperAGI](https://github.com/TransformerOptimus/SuperAGI)** ⭐⭐⭐ 💰

- **What it does**: Open source autonomous AI agent framework
- **Best for**: Building and deploying autonomous agents, agent management
- **Features**: Agent management, tool integration, performance monitoring
- **UI**: Graphical interface for agent management

---

## 🧠 Structured Generation & Control

### Output Structure & Validation

**[DSPy](https://github.com/stanfordnlp/dspy)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: Programming language model pipelines declaratively
- **Best for**: Complex reasoning chains, prompt optimization, systematic LM
  programming
- **Features**: Automatic prompt optimization, modular components, few-shot
  learning
- **Developer**: Stanford NLP Group

**[Outlines](https://github.com/outlines-dev/outlines)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: Structured text generation for LLMs (JSON, regex, grammars)
- **Best for**: Guaranteed valid outputs, structured data extraction, API
  responses
- **Features**: JSON schema validation, regex constraints, custom grammars
- **Performance**: Fast, reliable structured generation

**[Guardrails AI](https://github.com/guardrails-ai/guardrails)** ⭐⭐⭐⭐ 💰

- **What it does**: Framework for validating and correcting LLM outputs
- **Best for**: Output validation, safety checks, quality assurance
- **Features**: Custom validators, automatic correction, quality metrics
- **Use cases**: Production safety, content moderation, data validation

**[LMQL](https://lmql.ai/)** ⭐⭐⭐ 💰

- **What it does**: Query language for programming large language models
- **Best for**: Complex prompting, structured queries, research
- **Features**: SQL-like syntax, constraints, multistep reasoning
- **Unique**: Declarative approach to LLM programming

**[jsonformer](https://github.com/1rgs/jsonformer)** ⭐⭐⭐ 💰

- **What it does**: Generate valid JSON from language models
- **Best for**: Structured data extraction, API integrations, data validation
- **Features**: Guaranteed valid JSON, fast generation, simple API
- **Performance**: Efficient, reliable JSON generation

---

## 🧐 Memory & Persistence

### Long-term Memory Systems

**[Mem0](https://github.com/mem0ai/mem0)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: Memory layer for AI applications with personalization
- **Best for**: Personalized AI, long-term context, user memory
- **Features**: User preferences, adaptive learning, memory graphs
- **Use cases**: Personal assistants, customer support, educational apps

**[Zep](https://github.com/getzep/zep)** ⭐⭐⭐⭐ 💰

- **What it does**: Long-term memory store for conversational AI applications
- **Best for**: Chat applications, conversation history, context persistence
- **Features**: Conversation summarization, semantic search, memory extraction
- **Integration**: LangChain, custom applications

---

## 🎙️ Voice & Audio Tools

### Speech-to-Text & Text-to-Speech

**[OpenAI Whisper](https://github.com/openai/whisper)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: Automatic speech recognition system
- **Best for**: Transcription, multilingual speech recognition, audio processing
- **Features**: 99 languages, robust performance, multiple model sizes
- **Performance**: State-of-the-art accuracy

**[Coqui TTS](https://github.com/coqui-ai/TTS)** ⭐⭐⭐⭐ 💰

- **What it does**: Deep learning toolkit for text-to-speech synthesis
- **Best for**: Voice synthesis, voice cloning, multilingual TTS
- **Features**: 1000+ languages, voice cloning, real-time synthesis
- **Community**: Large open source community

**[ElevenLabs](https://elevenlabs.io/)** ⭐⭐⭐⭐ 💵

- **What it does**: AI voice generation and cloning platform
- **Best for**: High-quality voice synthesis, voice cloning, content creation
- **Features**: Realistic voices, voice cloning, API access
- **Quality**: Premium voice quality

**[Tortoise TTS](https://github.com/neonbjb/tortoise-tts)** ⭐⭐⭐ 💰

- **What it does**: Multi-voice text-to-speech system
- **Best for**: Voice cloning, custom voices, research
- **Features**: Voice cloning from samples, emotional control
- **Note**: Slower but high quality

---

## 💻 Code-Focused LLM Tools

### IDE Extensions & Code Assistants

**[Continue](https://github.com/continuedev/continue)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: Open source autopilot for VS Code and JetBrains
- **Best for**: Code completion, refactoring, explanation
- **Features**: Multiple LLM support, customizable, self-hosted
- **Integration**: VS Code, JetBrains IDEs

**[Cursor](https://cursor.sh/)** ⭐⭐⭐⭐ 💵 🚀

- **What it does**: AI-first code editor built on VS Code
- **Best for**: AI-assisted coding, pair programming, code generation
- **Features**: GPT-4 integration, codebase chat, AI commands
- **Performance**: Fast, smooth AI integration

**[Codeium](https://codeium.com/)** ⭐⭐⭐⭐ 🔄

- **What it does**: Free AI code completion and chat
- **Best for**: Code completion, chat assistance, multiple IDEs
- **Features**: 70+ programming languages, IDE extensions, free tier
- **Pricing**: Generous free tier, enterprise options

**[Tabnine](https://www.tabnine.com/)** ⭐⭐⭐ 💵

- **What it does**: AI code completion assistant
- **Best for**: Code completion, team collaboration, enterprise
- **Features**: Local models, team training, compliance features
- **Focus**: Privacy-first, on-premises options

---

## 🌐 Self-hosted & Local Platforms

### Complete LLM Platforms

**[AnythingLLM](https://github.com/Mintplex-Labs/anything-llm)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: Full-stack application for turning documents into chatbots
- **Best for**: Document Q&A, knowledge bases, private deployments
- **Features**: Multi-user, document processing, vector databases, local LLMs
- **Deployment**: Docker, desktop app, cloud

**[LibreChat](https://github.com/danny-avila/LibreChat)** ⭐⭐⭐⭐ 💰

- **What it does**: Enhanced ChatGPT clone with multiple AI providers
- **Best for**: Self-hosted ChatGPT alternative, multiple providers
- **Features**: Multiple AI providers, plugins, conversation branching
- **Providers**: OpenAI, Anthropic, Google, local models

**[Chatbot UI](https://github.com/mckaywrigley/chatbot-ui)** ⭐⭐⭐ 💰

- **What it does**: Open source ChatGPT UI clone
- **Best for**: Custom ChatGPT interface, self-hosting
- **Features**: Clean UI, multiple providers, conversation management
- **Deployment**: Vercel, self-hosted

**[Big-AGI](https://github.com/enricoros/big-AGI)** ⭐⭐⭐ 💰 🚀

- **What it does**: Advanced web UI for AI models with productivity features
- **Best for**: Power users, productivity, advanced AI interactions
- **Features**: Multiple models, personas, diagrams, voice input
- **UI**: Modern, feature-rich interface

---

## 🔄 Visual Workflow Builders

### No-Code/Low-Code LLM Apps

**[Flowise](https://github.com/FlowiseAI/Flowise)** ⭐⭐⭐⭐ 💰

- **What it does**: Drag & drop UI to build LLM flows using LangChain
- **Best for**: Visual workflow building, no-code LLM apps, prototyping
- **Features**: Visual flow builder, LangChain integration, API endpoints
- **Deployment**: Self-hosted, cloud options

**[Langflow](https://github.com/logspace-ai/langflow)** ⭐⭐⭐⭐ 💰

- **What it does**: Visual framework for building LangChain applications
- **Best for**: Visual LLM app development, rapid prototyping
- **Features**: Drag-and-drop interface, component library, export to Python
- **Integration**: LangChain ecosystem

**[Dify](https://github.com/langgenius/dify)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: LLM application development platform
- **Best for**: Building and operating generative AI applications
- **Features**: Workflow orchestration, model management, API development
- **Enterprise**: Production-ready, enterprise features

**[FastGPT](https://github.com/labring/FastGPT)** ⭐⭐⭐ 💰

- **What it does**: Knowledge-based platform built on LLMs
- **Best for**: Knowledge base applications, Q&A systems
- **Features**: Visual workflow, knowledge base management, API
- **Language**: Chinese-focused but supports English

---

## 📊 LLM Gateway & Operations

### LLM Infrastructure & Management

**[Portkey](https://github.com/Portkey-AI/gateway)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: AI gateway for observability, reliability, and security
- **Best for**: Production LLM deployments, cost optimization, observability
- **Features**: Load balancing, caching, rate limiting, cost tracking
- **Enterprise**: Production-grade reliability features

**[HeliconeAI](https://github.com/Helicone/helicone)** ⭐⭐⭐ 💰

- **What it does**: Open source observability platform for generative AI
- **Best for**: Request monitoring, cost tracking, performance analysis
- **Features**: Request logging, cost analysis, custom properties
- **Integration**: Simple proxy setup, multiple providers

**[Braintrust](https://www.braintrust.dev/)** ⭐⭐⭐ 🔄

- **What it does**: Enterprise-grade evaluation and observability for AI
- **Best for**: Enterprise AI evaluation, performance monitoring
- **Features**: Evaluation frameworks, dataset management, A/B testing
- **Focus**: Enterprise and research use cases

---

## 🔬 Research & Evaluation Tools

### Academic & Research Platforms

**[EleutherAI Eval Harness](https://github.com/EleutherAI/lm-evaluation-harness)**
⭐⭐⭐⭐ 💰

- **What it does**: Framework for evaluating language models on various tasks
- **Best for**: Research evaluation, model comparison, academic benchmarks
- **Features**: 200+ tasks, standardized evaluation, research-grade metrics
- **Community**: EleutherAI research community

**[Alpaca Eval](https://github.com/tatsu-lab/alpaca_eval)** ⭐⭐⭐ 💰

- **What it does**: Automatic evaluator for instruction-following models
- **Best for**: Instruction tuning evaluation, model comparison
- **Features**: Automated evaluation, leaderboards, research benchmarks
- **Developer**: Stanford Alpaca team

**[BIG-bench](https://github.com/google/BIG-bench)** ⭐⭐⭐⭐ 💰

- **What it does**: Beyond the Imitation Game collaborative benchmark
- **Best for**: Comprehensive model evaluation, research
- **Features**: 200+ tasks, diverse evaluation, research collaboration
- **Developer**: Google Research

**[MMLU](https://github.com/hendrycks/test)** ⭐⭐⭐⭐ 💰

- **What it does**: Massive Multitask Language Understanding benchmark
- **Best for**: Academic evaluation, knowledge assessment
- **Features**: 57 academic subjects, standardized testing
- **Usage**: Standard benchmark in research papers

---

## 🎨 Multimodal & Specialized Models

### Vision-Language Models

**[LLaVA](https://github.com/haotian-liu/LLaVA)** ⭐⭐⭐⭐ 💰

- **What it does**: Large Language and Vision Assistant
- **Best for**: Vision-language tasks, image understanding, research
- **Features**: Visual instruction tuning, conversation about images
- **Performance**: Strong vision-language capabilities

**[InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)**
⭐⭐⭐ 💰

- **What it does**: Vision-language instruction tuning
- **Best for**: Visual question answering, image captioning
- **Features**: Instruction following for vision tasks
- **Developer**: Salesforce Research

**[OpenAI CLIP](https://github.com/openai/CLIP)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: Connecting text and images
- **Best for**: Image-text understanding, zero-shot classification
- **Features**: Multi-modal embeddings, zero-shot capabilities
- **Impact**: Foundation for many vision-language applications

---

## 🔗 Integration & Utility Tools

### Connectors & Utilities

**[Pinecone Datasets](https://github.com/pinecone-io/datasets)** ⭐⭐⭐ 💰

- **What it does**: Ready-to-use datasets for vector databases
- **Best for**: Quick RAG setup, testing, benchmarking
- **Features**: Pre-processed datasets, embeddings included
- **Integration**: Pinecone ecosystem

**[Weaviate Recipes](https://github.com/weaviate/recipes)** ⭐⭐⭐ 💰

- **What it does**: Code examples and tutorials for Weaviate
- **Best for**: Learning Weaviate, implementation examples
- **Features**: Production-ready examples, best practices
- **Community**: Community-contributed recipes

**[ChromaDB Recipes](https://github.com/chroma-core/chroma/tree/main/examples)**
⭐⭐⭐ 💰

- **What it does**: Example applications using ChromaDB
- **Best for**: Learning ChromaDB, quick starts
- **Features**: Various use case examples, integration patterns
- **Documentation**: Well-documented examples

---

## 📱 Mobile & Edge Deployment

### Mobile LLM Solutions

**[MLX](https://github.com/ml-explore/mlx)** ⭐⭐⭐ 💰 🚀

- **What it does**: Apple's ML framework optimized for Apple Silicon
- **Best for**: Apple silicon deployment, iOS/macOS apps
- **Features**: Apple Silicon optimization, Swift/Python APIs
- **Performance**: Optimized for M-series chips

**[MLC LLM](https://github.com/mlc-ai/mlc-llm)** ⭐⭐⭐ 💰

- **What it does**: Enable LLMs natively on mobile devices and edge
- **Best for**: Mobile deployment, edge computing, resource constraints
- **Features**: Mobile optimization, multiple platforms
- **Platforms**: iOS, Android, WebGPU, native

---

## 🎛️ Fine-tuning Platforms

### Training Frameworks

**[Unsloth](https://github.com/unslothai/unsloth)** ⭐⭐⭐⭐⭐ 💰 🚀

- **What it does**: Accelerated fine-tuning library for LLMs with hand-written
  CUDA kernels
- **Best for**: Fast LoRA and QLoRA fine-tuning on consumer and cloud GPUs,
  reducing VRAM usage
- **Features**: 2–5× faster training, 70% less VRAM, supports Llama 4, Qwen3,
  Mistral, Phi-4, Gemma 4
- **Pricing**: Free and open source; Unsloth Pro for managed runs
- **Unique**: No approximations — exact same outputs as standard training but
  significantly faster; Colab notebooks for every major model family

**[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)** ⭐⭐⭐⭐ 💰

- **What it does**: Flexible fine-tuning framework driven by YAML configuration
- **Best for**: Reproducible fine-tuning pipelines, teams wanting
  config-as-code training workflows
- **Features**: LoRA, QLoRA, full fine-tune, FSDP, DeepSpeed, multi-GPU,
  Flash Attention 2
- **Pricing**: Free and open source
- **Unique**: Single YAML file defines the entire training run; first-class
  support for custom chat templates and dataset formats

**[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: Unified fine-tuning framework with web UI and CLI
- **Best for**: Rapid experimentation, teams wanting a visual interface for
  training and evaluation
- **Features**: 100+ model support, LoRA/QLoRA/full FT, built-in evaluation,
  web UI (LlamaBoard), dataset management
- **Pricing**: Free and open source
- **Unique**: LlamaBoard GUI makes fine-tuning accessible without writing code;
  broadest model compatibility of any open-source framework

**[Hugging Face AutoTrain](https://huggingface.co/autotrain)** ⭐⭐⭐⭐ 🔄

- **What it does**: No-code fine-tuning platform for LLMs and other ML models
- **Best for**: Teams without ML infrastructure expertise, quick domain
  adaptation of open-weight models
- **Features**: Text classification, causal LM, chat fine-tuning, automatic
  hyperparameter search
- **Pricing**: Free self-hosted; paid cloud compute via HuggingFace Spaces
- **Unique**: Fully managed training with no infrastructure setup; integrates
  directly with HuggingFace Hub for dataset and model storage

**[OpenAI Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)** ⭐⭐⭐⭐ 💵

- **What it does**: Managed fine-tuning service for GPT-4o and GPT-4o-mini
- **Best for**: Production teams wanting fine-tuned OpenAI models without
  managing GPU infrastructure
- **Features**: JSONL dataset upload, automatic training, model evaluation,
  cost tracking
- **Pricing**: $25/1M training tokens; fine-tuned model inference at standard
  rates
- **Unique**: Simplest path to a production fine-tuned model; no infrastructure
  to manage; integrates directly with the OpenAI API

**[Together AI Fine-tuning](https://www.together.ai/fine-tuning)** ⭐⭐⭐⭐ 💵

- **What it does**: Managed fine-tuning for open-source models at scale
- **Best for**: Teams fine-tuning Llama, Qwen, Mistral, or DeepSeek models
  with production serving included
- **Features**: Full fine-tune and LoRA, custom datasets, automatic evaluation,
  instant deployment after training
- **Pricing**: Per-GPU-hour; fine-tuned models served at same price as base
  models
- **Unique**: Fine-tuned model is immediately available via API after training
  completes, no separate deployment step

**[Modal](https://modal.com/)** ⭐⭐⭐⭐ 🔄

- **What it does**: Serverless GPU platform for running fine-tuning and
  inference workloads in Python
- **Best for**: Engineers who want full code control over training without
  managing cloud infrastructure
- **Features**: On-demand H100/A100 access, Python-native API, persistent
  volumes, fast cold starts
- **Pricing**: Pay-per-second GPU billing; free tier included
- **Unique**: Runs any Python training script on cloud GPUs with one decorator;
  no YAML, no clusters, no DevOps

---

## 📈 Evaluation & Monitoring

### Production Observability

**[LangSmith](https://smith.langchain.com/)** ⭐⭐⭐⭐⭐ 🔄

- **What it does**: Observability and evaluation platform for LLM applications
- **Best for**: Teams using LangChain or LangGraph who need full trace
  visibility and dataset-driven evaluation
- **Features**: Full trace logging, prompt versioning, dataset management,
  automated evaluators, A/B testing
- **Pricing**: Free tier (5K traces/month); paid from $39/month
- **Unique**: Native LangChain integration gives zero-config tracing for any
  LangChain application; human annotation interface for building eval datasets

**[Langfuse](https://langfuse.com/)** ⭐⭐⭐⭐⭐ 💰 🚀

- **What it does**: Open source LLM observability and evaluation platform
- **Best for**: Teams wanting full self-hosted observability with no
  vendor lock-in
- **Features**: Tracing, prompt management, evaluation, cost tracking,
  user feedback, dataset curation
- **Pricing**: Free and open source (self-hosted); cloud from $59/month
- **Unique**: Full-featured open source alternative to LangSmith; works with
  any LLM provider via OpenTelemetry and SDK integrations

**[Weights & Biases Weave](https://wandb.ai/site/weave)** ⭐⭐⭐⭐ 🔄

- **What it does**: LLM tracing and evaluation layer built on top of W&B
- **Best for**: Teams already using W&B for ML training who want unified
  observability across training and inference
- **Features**: Trace logging, evaluations, dataset versioning, cost tracking,
  integration with W&B experiment tracking
- **Pricing**: Free tier; paid via W&B plans from $50/month
- **Unique**: Unified platform for both model training metrics and production
  LLM application observability; strong dataset versioning

**[Arize Phoenix](https://github.com/Arize-ai/phoenix)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: Open source AI observability with built-in evals and
  tracing
- **Best for**: Teams wanting local or self-hosted observability with strong
  evaluation tooling
- **Features**: OpenTelemetry-native tracing, LLM evals, RAG evaluation,
  embedding visualization, dataset management
- **Pricing**: Free and open source
- **Unique**: Runs entirely locally with no data leaving your environment;
  strong RAG-specific evaluation metrics out of the box

**[PromptLayer](https://promptlayer.com/)** ⭐⭐⭐ 🔄

- **What it does**: Prompt management and request logging platform
- **Best for**: Teams wanting simple request logging and prompt version
  control without full observability infrastructure
- **Features**: Request logging, prompt templates, A/B testing, cost tracking,
  visual diff for prompt versions
- **Pricing**: Free tier; paid from $25/month
- **Unique**: Minimal setup — drop-in wrapper around the OpenAI client;
  good for teams starting with observability

---

## 🎨 Prompt Engineering

### Testing & Optimization

**[Promptfoo](https://github.com/promptfoo/promptfoo)** ⭐⭐⭐⭐⭐ 💰 🚀

- **What it does**: Open source tool for testing and evaluating LLM prompts
  and models
- **Best for**: Systematic prompt testing, model comparison, CI/CD integration
  for LLM quality
- **Features**: YAML-based test definitions, automated evals, red teaming,
  multi-model comparison, CI/CD integration
- **Pricing**: Free and open source; cloud version available
- **Unique**: Treats prompt evaluation like unit tests — define expected
  outputs, run assertions, catch regressions before they reach production

**[Humanloop](https://humanloop.com/)** ⭐⭐⭐⭐ 🔄 🏢

- **What it does**: Prompt management and evaluation platform for enterprise
  teams
- **Best for**: Large teams collaborating on prompts, enterprise prompt
  lifecycle management
- **Features**: Prompt versioning, A/B testing, human feedback collection,
  automated evals, fine-tuning integration
- **Pricing**: Free tier; paid from $149/month
- **Unique**: Full prompt lifecycle management from development to production
  with team collaboration and approval workflows

**[Agenta](https://github.com/Agenta-AI/agenta)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: Open source LLM application development platform with
  prompt management and evaluation
- **Best for**: Teams wanting self-hosted prompt engineering with evaluation
  in a single platform
- **Features**: Prompt playground, A/B testing, custom evaluators, tracing,
  human annotation
- **Pricing**: Free and open source; cloud version available
- **Unique**: Combines prompt playground, versioning, and evaluation in one
  open source platform; designed for the full team, not just engineers

**[DSPy](https://github.com/stanfordnlp/dspy)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: Framework for algorithmically optimizing LLM prompts and
  pipelines
- **Best for**: Replacing hand-written prompts with automatically optimized
  ones, complex multi-step reasoning pipelines
- **Features**: Automatic prompt optimization, few-shot example selection,
  modular pipeline composition, multiple optimizers (MIPROv2, BootstrapRS)
- **Pricing**: Free and open source
- **Unique**: Eliminates manual prompt engineering — defines the task
  declaratively and optimizes prompts automatically against a metric

---

## 🔧 Supporting Tools

### Data Processing & Document Parsing

**[Unstructured](https://github.com/Unstructured-IO/unstructured)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: Document parsing library that extracts clean text from
  any file format for LLM ingestion
- **Best for**: RAG pipelines requiring robust document parsing across diverse
  file types
- **Features**: PDF, Word, Excel, HTML, email, images (OCR), 20+ file formats,
  element-level chunking
- **Pricing**: Free and open source; managed API available
- **Unique**: De facto standard for document preprocessing in RAG pipelines;
  maintains document structure (titles, tables, lists) rather than dumping
  raw text

**[Docling](https://github.com/DS4SD/docling)** ⭐⭐⭐⭐ 💰 🚀

- **What it does**: IBM's document parsing library with advanced layout
  understanding
- **Best for**: Complex PDFs with tables, figures, and multi-column layouts
  that simpler parsers struggle with
- **Features**: PDF layout analysis, table extraction, figure detection,
  Markdown and JSON output, DoclingDocument format
- **Pricing**: Free and open source (Apache 2.0)
- **Unique**: Handles complex academic and enterprise PDFs that other parsers
  corrupt; native integration with LlamaIndex and LangChain

**[LlamaParse](https://llamaindex.ai/llamaparse)** ⭐⭐⭐⭐ 🔄

- **What it does**: Managed document parsing API optimized for RAG
- **Best for**: Teams needing high-quality PDF parsing without running
  infrastructure
- **Features**: Advanced PDF parsing, table extraction, image extraction,
  Markdown output, page-level metadata
- **Pricing**: Free tier (1K pages/day); paid from $3/1K pages
- **Unique**: Purpose-built for RAG — outputs are structured for optimal
  chunking and retrieval; handles complex layouts better than open source
  alternatives for most document types

### Unified API & Cost Management

**[LiteLLM](https://github.com/BerriAI/litellm)** ⭐⭐⭐⭐⭐ 💰 🚀

- **What it does**: Unified Python SDK and proxy that calls 100+ LLM APIs
  using the OpenAI format
- **Best for**: Applications needing to switch between providers, load
  balancing across APIs, cost tracking
- **Features**: 100+ providers, load balancing, fallbacks, spend tracking,
  virtual API keys, OpenAI-compatible proxy
- **Pricing**: Free and open source; LiteLLM Pro for enterprise features
- **Unique**: Single interface for every LLM provider — change one string to
  swap from OpenAI to Anthropic to Bedrock; built-in cost tracking across all
  providers

**[tiktoken](https://github.com/openai/tiktoken)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: Fast tokenizer for OpenAI models written in Rust
- **Best for**: Accurately counting tokens before sending requests to avoid
  truncation or overspend
- **Features**: BPE tokenization, cl100k_base and o200k_base encodings,
  Python bindings, very fast
- **Pricing**: Free and open source (MIT)
- **Unique**: The reference tokenizer for all OpenAI models; 3–6× faster than
  pure Python alternatives

**[Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: Fast tokenizer library for any Hugging Face model
- **Best for**: Tokenizing text for open-weight models, building custom
  tokenization pipelines
- **Features**: Rust-backed, BPE/WordPiece/Unigram, padding, truncation,
  offset mapping
- **Pricing**: Free and open source (Apache 2.0)
- **Unique**: Used by every Hugging Face Transformers model; supports
  tokenizer training from scratch

### Context & Memory Utilities

**[LangChain Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/)** ⭐⭐⭐⭐ 💰

- **What it does**: Collection of text chunking strategies for RAG pipelines
- **Best for**: Splitting documents into optimal chunks before embedding
  and retrieval
- **Features**: Recursive, semantic, token-aware, code-aware, and
  markdown-aware splitters
- **Pricing**: Free and open source
- **Unique**: Most comprehensive collection of chunking strategies in a single
  library; code splitter understands language syntax for better splits

---

## 💲 Cost Comparison

> Pricing as of April 2026. All rates are per 1M tokens (input / output)
> unless noted. Verify current pricing directly with each provider before
> making decisions.

### Frontier Models

| Provider | Model | Input | Output | Context |
| --- | --- | --- | --- | --- |
| OpenAI | GPT-5.4 | $5.00 | $20.00 | 1M |
| OpenAI | GPT-5.4 Mini | $0.40 | $1.60 | 1M |
| Anthropic | Claude Opus 4.6 | $15.00 | $75.00 | 1M |
| Anthropic | Claude Sonnet 4.6 | $3.00 | $15.00 | 1M |
| Google | Gemini 3.1 Pro | $3.50 | $10.50 | 1M |
| Google | Gemini 2.5 Flash | $0.15 | $0.60 | 1M |
| xAI | Grok 4 | $3.00 | $15.00 | 2M |
| xAI | Grok 4.1 Fast | $0.20 | $0.50 | 2M |
| Mistral | Mistral Large 3 API | $2.00 | $6.00 | 256K |
| DeepSeek | DeepSeek V3.2 | $0.07 | $0.28 | 64K |

### Budget & Efficient Models

| Provider | Model | Input | Output | Notes |
| --- | --- | --- | --- | --- |
| Anthropic | Claude Haiku 4.5 | $0.25 | $1.25 | Fastest Claude |
| OpenAI | GPT-5.4 Nano | $0.05 | $0.20 | Smallest GPT |
| Google | Gemini 2.5 Flash-Lite | $0.05 | $0.15 | Lowest cost tier |
| Groq | Llama 3.3 70B | $0.05 | $0.10 | 840+ tok/sec |
| Cerebras | Llama 3.3 70B | $0.10 | $0.10 | 2300+ tok/sec |
| DeepSeek | R1 (cached) | $0.04 | $0.16 | 90% cache discount |
| Mistral | Mistral Small 4 API | $0.10 | $0.30 | Unified reasoning+vision |
| Together AI | Llama 4 Scout | $0.10 | $0.30 | 10M context |

### Cost-Saving Strategies

- **Prompt caching**: Anthropic (90% discount on cached prefixes), OpenAI, Google all offer
  caching — structure prompts so the static system prompt is always a prefix
- **Batch processing**: All major providers offer 50% discounts for
  asynchronous batch jobs with 24-hour SLA
- **Model tiering**: Route simple classification or extraction tasks to
  Flash/Haiku/Nano; reserve frontier models for complex reasoning
- **Context management**: Trim conversation history aggressively — costs
  scale linearly with context length

---

## 🏗️ Recommended Stacks

> Opinionated starting points for common use cases. Swap components as your
> requirements evolve.

### Starter RAG Application

Best for: developers building their first document Q&A or knowledge base.

| Layer | Tool | Why |
| --- | --- | --- |
| LLM | Claude Sonnet 4.6 or Gemini 2.5 Flash | Balance of quality and cost |
| Framework | LlamaIndex | Purpose-built for RAG |
| Vector DB | Chroma (dev) → Qdrant (prod) | Easy local start, fast production |
| Parsing | Unstructured | Handles mixed document types |
| Serving | FastAPI | Lightweight, async-ready |

### Production Agentic System

Best for: multi-step autonomous workflows with tool use and human-in-the-loop.

| Layer | Tool | Why |
| --- | --- | --- |
| LLM | Claude Sonnet 4.6 | Best agentic performance |
| Orchestration | LangGraph | Stateful graph-based agent control |
| Memory | Mem0 | Long-term user and session memory |
| Tools | LiteLLM proxy | Provider fallback and cost tracking |
| Observability | Langfuse | Full trace visibility, self-hostable |
| Vector DB | Qdrant | Fast filtering for tool retrieval |

### Local / Privacy-First Stack

Best for: teams that cannot send data to external APIs.

| Layer | Tool | Why |
| --- | --- | --- |
| LLM | Ollama + Llama 4 Scout or Qwen3-32B | Easy local serving |
| Framework | LangChain | Broad local model support |
| Vector DB | Chroma | Runs fully in-process |
| Parsing | Docling | Local, no API calls |
| UI | AnythingLLM | Full-stack local platform |

### High-Performance Inference (Cloud)

Best for: latency-critical production applications at scale.

| Layer | Tool | Why |
| --- | --- | --- |
| LLM | Groq or Cerebras | 840–2300+ tokens/sec |
| Serving | vLLM (self-hosted) | 24× throughput for owned models |
| Gateway | LiteLLM | Load balance across providers |
| Caching | Redis + LiteLLM cache | Avoid redundant inference calls |
| Monitoring | Arize Phoenix | Low-overhead production tracing |

### Research & Evaluation

Best for: researchers benchmarking models or building eval-driven pipelines.

| Layer | Tool | Why |
| --- | --- | --- |
| Models | AI2 OLMo 3 + DeepSeek R1 | Fully open; reproducible results |
| Prompting | DSPy | Automated prompt optimization |
| Evaluation | Promptfoo + EleutherAI Eval Harness | CI testing + standard benchmarks |
| Tracking | Weights & Biases Weave | Unified training and inference metrics |
| Fine-tuning | Unsloth + Axolotl | Fast, reproducible training runs |

---

## 📚 Learning Resources

### Courses & Tutorials

**[fast.ai Practical Deep Learning](https://course.fast.ai/)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: Practical deep learning course with hands-on LLM modules
- **Best for**: Developers wanting a code-first path from basics to LLM
  fine-tuning
- **Unique**: Top-down teaching approach — run models first, understand
  theory after; completely free

**[DeepLearning.AI Short Courses](https://www.deeplearning.ai/short-courses/)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: Focused 1–2 hour courses on specific LLM topics
  (RAG, agents, fine-tuning, evals)
- **Best for**: Practitioners who want targeted skill upgrades without
  committing to a full course
- **Unique**: Built with tool providers (LangChain, Anthropic, OpenAI) —
  uses current APIs and best practices

**[Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: End-to-end course on transformers, fine-tuning, and
  the HuggingFace ecosystem
- **Best for**: Developers wanting to understand transformer internals and
  work confidently with open-weight models
- **Unique**: The definitive resource for the HuggingFace ecosystem; always
  kept up to date; completely free

**[Andrej Karpathy — Neural Networks: Zero to Hero](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: YouTube series building GPT from scratch in pure Python
- **Best for**: Anyone who wants deep intuition for how language models work
  at the mathematical level
- **Unique**: Best intuition-building resource available; builds every
  component from scratch — no black boxes

### Essential Reading

**[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: Original transformer architecture paper (Vaswani et al., 2017)
- **Best for**: Understanding the architecture that underpins every modern LLM
- **Unique**: Required reading; every LLM traces its architecture back to
  this paper

**[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** ⭐⭐⭐⭐⭐ 💰

- **What it does**: Visual, intuitive walkthrough of the transformer
  architecture
- **Best for**: Developers who learn best from diagrams and step-by-step
  visual explanations
- **Unique**: The most-referenced explanation of transformers for
  practitioners; no math required to follow along

**[Chip Huyen — AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)** ⭐⭐⭐⭐⭐ 💵

- **What it does**: Comprehensive book on building production AI applications
  with LLMs
- **Best for**: Engineers and architects designing production LLM systems
- **Unique**: Covers the full stack from model selection to deployment,
  evaluation, and monitoring — the definitive production LLM engineering
  reference

### Communities & News

**[Hugging Face Forums](https://discuss.huggingface.co/)** ⭐⭐⭐⭐ 💰

- **What it does**: Community forum for open-weight models, datasets, and
  the HuggingFace ecosystem
- **Best for**: Getting help with model loading, fine-tuning, and deployment
  issues; following open-source model releases
- **Unique**: Largest open-source AI community; model authors often respond
  directly to questions

**[r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)** ⭐⭐⭐⭐ 💰

- **What it does**: Reddit community focused on running LLMs locally
- **Best for**: Staying current on quantization, hardware recommendations,
  and new open-weight releases
- **Unique**: Fastest community for local deployment news; model benchmarks
  and hardware comparisons appear here before anywhere else

**[The Batch (DeepLearning.AI)](https://www.deeplearning.ai/the-batch/)** ⭐⭐⭐⭐ 💰

- **What it does**: Weekly AI newsletter curated by Andrew Ng
- **Best for**: Staying informed on AI research and industry developments
  without drowning in noise
- **Unique**: High signal-to-noise ratio; covers both research papers and
  industry news with context

## 🤝 Contributing

We welcome contributions from the LLM community. Whether you want to add a
new tool, update outdated information, or fix a broken link, every contribution
helps make this resource better for everyone.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the full contribution guide,
including the required tool entry format, rating system, quality standards, and
review process.

For questions, open a
[Discussion](https://github.com/dr-saad-la/awesome-llm-tools/discussions). For
security concerns, see [SECURITY.md](SECURITY.md).

---

## 📄 License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

This work is licensed under
[Creative Commons Zero v1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
You are free to use, copy, modify, and distribute this list for any purpose
without attribution or permission.

### Attribution (Optional)

Attribution is not required, but appreciated:

```markdown
LLM tools sourced from [Awesome LLM Tools](https://github.com/dr-saad-la/awesome-llm-tools)
```

### Disclaimer

This list is provided "as is" without warranty of any kind. Tool information,
pricing, and availability change frequently. Always verify current details
directly with each provider before making decisions.

---

### Star Growth

[![Star History Chart](https://api.star-history.com/svg?repos=dr-saad-la/awesome-llm-tools&type=Date)](https://star-history.com/#dr-saad-la/awesome-llm-tools&Date)
