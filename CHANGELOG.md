# Changelog

All notable changes to this project will be documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

## [0.4.0] - 2026-04-09

### Added

- xAI Grok and Amazon Bedrock added to Commercial LLM APIs Tier 1
- Fully rewritten Open Source Models section with April 2026 landscape:
  - Llama 4 (Scout and Maverick) and Llama 3.x Series replacing outdated Llama 3.1 entry
  - Qwen3 / Qwen3.5 promoted to five-star entry reflecting 700M+ downloads
  - DeepSeek V3.2 / R1 with MIT license and full distilled model family
  - Mistral Large 3, Mistral Small 4, and Ministral 3 replacing single outdated entry
  - Gemma 4 reflecting Apache 2.0 license upgrade and major benchmark gains
  - Phi-4 Family replacing superseded Phi-3 entry
  - New entrants: NVIDIA Nemotron 3, OpenAI gpt-oss, AI2 OLMo 3, Zhipu GLM-5, Falcon 3
  - Discontinued / Legacy Models section for Yi, Code Llama, DeepSeek Coder, Phi-3, DBRX
- Fine-tuning Platforms section: Unsloth, Axolotl, LLaMA-Factory, HF AutoTrain,
  OpenAI Fine-tuning, Together AI Fine-tuning, Modal
- Evaluation & Monitoring section: LangSmith, Langfuse, W&B Weave, Arize Phoenix,
  PromptLayer
- Prompt Engineering section: Promptfoo, Humanloop, Agenta, DSPy
- Supporting Tools section: Unstructured, Docling, LlamaParse, LiteLLM, tiktoken,
  HF Tokenizers, LangChain Text Splitters
- Cost Comparison section with April 2026 pricing tables and cost-saving strategies
- Recommended Stacks section covering five common deployment patterns
- Learning Resources section: courses, essential reading, and communities

### Changed

- Open Source Models section fully restructured around April 2026 model landscape
- Yi Series marked as discontinued following 01.AI pivot in March 2025
- Code Llama, DeepSeek Coder, Phi-3, and DBRX marked as legacy or deprecated

## [0.3.0] - 2026-04-07

### Added

- Pre-commit hooks for automated code quality checks
- Markdownlint configuration (.markdownlint.yaml)
- Justfile task runner with validate, lint, and CI recipes
- GitHub Actions CI workflow
- GitHub Funding configuration
- uv lockfile for reproducible environments

### Changed

- Applied markdownlint auto-fixes across all documentation files
- Updated lint rules to better suit awesome-list format
- Improved duplicate detection script

---

## [0.2.0] - 2025-07-17

### Added

- Contribution guidelines (CONTRIBUTING.md)
- Code of Conduct (CODE_OF_CONDUCT.md)
- GitHub issue templates for bug reports, tool suggestions, and conduct reports
- Automated format validation script (scripts/validate_format.py)
- Automated duplicate detection script (scripts/check_duplicates.py)
- Tool categories documentation (docs/tool-categories.md)
- GitHub Funding configuration

### Changed

- License changed from MIT to CC0 1.0 Universal
- Major expansion of tool listings across all categories
- Improved README structure and formatting

---

## [0.1.0] - 2024-07-30

### Added

- Initial curated list of LLM tools
- Commercial LLM APIs section
- Open Source Models section
- Development Frameworks section
