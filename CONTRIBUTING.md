# Contributing to Awesome LLM Tools

Thank you for your interest in contributing. This resource grows stronger
with community input, and we appreciate every contribution — from adding
new tools to fixing typos.

---

## Table of Contents

- [Quick Start](#quick-start)
- [How to Add a Tool](#how-to-add-a-tool)
- [Field Reference](#field-reference)
- [Rating System](#rating-system)
- [Category Guidelines](#category-guidelines)
- [Quality Standards](#quality-standards)
- [Review Process](#review-process)
- [Getting Help](#getting-help)
- [Recognition](#recognition)
- [Code of Conduct](#code-of-conduct)

---

## Quick Start

1. Fork this repository to your GitHub account
2. Create a new branch: `git switch -c add-tool-name`
3. Make your changes following the format described below
4. Verify all links are reachable
5. Submit a pull request using the provided template
6. Wait for review — we aim to respond within 48 hours

For simple fixes such as typos or broken links, you can use the GitHub
Edit button (pencil icon) directly on README.md without forking.

---

## How to Add a Tool

### Standard format

Use this format for tools in most sections:

```markdown
**[Tool Name](https://tool-website.com/)** ⭐⭐⭐⭐ 💰

- **What it does**: Brief description of primary function (1-2 sentences)
- **Best for**: Main use cases and target audience
- **Features**: Key features that set it apart
- **Pricing**: Cost model or "Free and open source"
- **Unique**: What makes this tool different from alternatives
```

### Format for commercial APIs

For tools in the Commercial LLM APIs section, use `Models` instead of
`What it does`:

```markdown
**[Tool Name](https://tool-website.com/)** ⭐⭐⭐⭐ 💵

- **Models**: List of available models
- **Best for**: Main use cases and target audience
- **Pricing**: Cost per million tokens or subscription cost
- **Unique**: What makes this provider different
```

### Format for open source models

For tools in the Open Source Models section, use `Sizes` or `Models`:

```markdown
**[Model Name](https://model-repo.com/)** ⭐⭐⭐⭐ 💰

- **Sizes**: Available parameter counts (e.g. 7B, 13B, 70B)
- **Best for**: Main use cases and target audience
- **License**: License type (e.g. Apache 2.0, MIT, custom)
- **Unique**: What makes this model stand out
```

### Real example

```markdown
**[LangGraph](https://github.com/langchain-ai/langgraph)** ⭐⭐⭐⭐⭐ 💰 🚀

- **What it does**: Build stateful, multi-actor applications with LLMs
  using graph-based workflows
- **Best for**: Complex agent workflows, multi-step reasoning,
  human-in-the-loop systems
- **Features**: State management, branching logic, human approval nodes,
  persistence
- **Pricing**: Free and open source
- **Unique**: LangChain native with advanced state management for complex
  agent workflows
```

---

## Field Reference

### Required fields

Every tool entry must include:

| Field      | Description                           |
| ---------- | ------------------------------------- |
| `Best for` | Target audience and primary use cases |

Every tool entry must also include at least one description field:

| Field          | Use when                                         |
| -------------- | ------------------------------------------------ |
| `What it does` | Standard tools and frameworks                    |
| `Models`       | Commercial APIs and hosted model services        |
| `Sizes`        | Open source models with multiple parameter sizes |

### Optional fields

Include any of these when they add meaningful information:

`Features`, `Pricing`, `Unique`, `Languages`, `License`, `Hardware`,
`Platforms`, `Performance`, `Integration`, `Deployment`, `Developer`,
`Community`, `Note`

### Field length

- Minimum: 5 characters
- Maximum: 300 characters
- Keep descriptions factual and concise

---

## Rating System

### Star ratings

| Rating     | Meaning   | Criteria                                    |
| ---------- | --------- | ------------------------------------------- |
| ⭐⭐⭐⭐⭐ | Must-have | Industry standard, dominant in its category |
| ⭐⭐⭐⭐   | Excellent | High quality, highly recommended            |
| ⭐⭐⭐     | Good      | Solid choice with specific strengths        |
| ⭐⭐       | Decent    | Functional, useful for specific cases       |

Minimum rating is 2 stars. Do not submit tools that would score 1 star.

### Rating guidelines

Rate based on objective factors:

- Quality — performance, reliability, accuracy
- Documentation — setup guides, API docs, examples
- Community — active development, responsiveness, user base
- Maintenance — regular updates, bug fixes
- Uniqueness — what it does that alternatives do not

Do not over-rate tools you have built. Do not under-rate competitors.
Rate conservatively when unsure.

### Badge icons

| Icon | Meaning                                 |
| ---- | --------------------------------------- |
| 💰   | Free / open source                      |
| 💵   | Paid / commercial                       |
| 🔄   | Freemium (free tier + paid options)     |
| 🚀   | Trending / recently launched or updated |
| 🏢   | Enterprise-focused                      |

---

## Category Guidelines

### Choosing the right category

| Category               | What belongs here                             |
| ---------------------- | --------------------------------------------- |
| Commercial LLM APIs    | Hosted AI services accessed via API           |
| Open Source Models     | Downloadable model weights                    |
| Development Frameworks | Libraries for building LLM applications       |
| Agent Frameworks       | Multi-agent and autonomous systems            |
| Local Deployment       | Tools for running models on your own hardware |
| Vector Databases & RAG | Embedding storage and retrieval systems       |
| Fine-tuning Platforms  | Tools for customizing pre-trained models      |

When unsure, look at where similar tools are placed. Open an issue to
discuss categorization before submitting if you are still uncertain.

### Creating new categories

A new category requires:

- At least 5 quality tools that belong in it
- A clear gap that existing categories do not fill
- Discussion in an issue before the PR is submitted

### Ordering within categories

Tools should be ordered roughly by star rating, highest first. Within
the same rating tier, order by popularity or ecosystem importance.

---

## Quality Standards

### What we accept

- Actively maintained — recent commits or releases within 6 months
- Well documented — clear setup instructions and usage examples
- Functional — tool works as described
- Unique value — not a direct duplicate of an existing listing
- LLM-relevant — directly supports LLM development or deployment

### What we do not accept

- Abandoned projects — no updates in 6+ months
- Broken tools — non-functional, broken links, major unresolved bugs
- Exact duplicates — same tool already listed without meaningful difference
- Off-topic tools — not directly related to LLM development
- Alpha or placeholder projects — not ready for general use

### Link requirements

- URL must be reachable at time of submission
- Link must point to the official website or repository
- Use HTTPS where available
- Do not use URL shorteners or redirects

---

## Review Process

### Automated checks

Every pull request runs the following checks automatically:

- Markdownlint — formatting and style consistency
- Format validation — required fields, star ratings, badge icons
- Duplicate detection — exact name and URL matches against existing entries

Fix any errors reported by these checks before requesting review.
Warnings are informational and will not block merging.

### Manual review stages

1. Automated checks pass (within minutes of submission)
2. Maintainer reviews content quality and category fit (1-3 days)
3. Final decision — merged, changes requested, or closed with explanation

### Response times

| Contribution type     | Expected response |
| --------------------- | ----------------- |
| New tool addition     | 1-3 days          |
| Information update    | 1-2 days          |
| Typo or link fix      | Same day to 1 day |
| New category proposal | 3-7 days          |

---

## Getting Help

- General questions: open a [Discussion](https://github.com/dr-saad-la/awesome-llm-tools/discussions)
- Bug reports: use the [Bug Report issue template](https://github.com/dr-saad-la/awesome-llm-tools/issues/new?template=bug_report.md)
- Tool suggestions: use the [Tool Suggestion issue template](https://github.com/dr-saad-la/awesome-llm-tools/issues/new?template=tool_suggestion.md)
- Security concerns: see [SECURITY.md](SECURITY.md)

### Common questions

**My PR was closed. Why?**
The most common reasons are inactive maintenance, broken links, duplicate
functionality, or the tool not being directly relevant to LLM development.
The closing comment will explain the specific reason.

**What rating should I give my tool?**
Compare it honestly to similar tools already listed. When in doubt, rate
conservatively. Maintainers may adjust ratings during review.

**Can I add a tool I built?**
Yes, but apply the same quality standards as any other tool. Self-promotion
without genuine value will be declined.

**Can I suggest reorganizing sections?**
Yes. Open an issue with your proposal and rationale before making changes.

---

## Recognition

All contributors are credited through:

- Commit history attribution
- Mention in release notes for significant contributions

Outstanding contributors may be invited to become maintainers.

---

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).

All participants are expected to be respectful, constructive, and
professional. Harassment, trolling, and spam will result in removal.
Report concerns to the maintainers.

---

_Last updated: April 2026_
