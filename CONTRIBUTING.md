# Contributing to Awesome LLM Tools 🤝

Thank you for your interest in contributing to **Awesome LLM Tools**! This resource grows stronger with community input, and we appreciate every contribution - from new tool suggestions to fixing typos.

## 📋 Table of Contents

- [Quick Start Guide](#-quick-start-guide)
- [How to Add a Tool](#-how-to-add-a-tool)
- [Rating System](#-rating-system)
- [Category Guidelines](#-category-guidelines)
- [Quality Standards](#-quality-standards)
- [Review Process](#-review-process)
- [Common Scenarios](#-common-scenarios)
- [Getting Help](#-getting-help)

---

## 🚀 Quick Start Guide

### **For New Contributors:**

1. **Fork** this repository to your GitHub account
2. **Create a new branch** for your changes: `git checkout -b add-new-tool`
3. **Make your changes** following the format below
4. **Test all links** work correctly
5. **Submit a Pull Request** with a clear description
6. **Wait for review** - we typically respond within 48 hours

### **For Simple Updates:**
- Use GitHub's **Edit button** (pencil icon) directly on README.md
- Perfect for fixing typos, updating links, or correcting information

---

## 🛠️ How to Add a Tool

### **Required Format:**

Use this **exact format** for consistency:

```markdown
**[Tool Name](https://tool-website.com/)** ⭐⭐⭐⭐ 💰 🚀
- **What it does**: Brief description of primary function (1-2 sentences)
- **Best for**: Main use cases and target audience
- **Features**: Key features that set it apart from competitors
- **Pricing**: Cost model (if applicable)
- **Unique**: What makes this tool special or different
```

### **Real Example:**

```markdown
**[LangGraph](https://github.com/langchain-ai/langgraph)** ⭐⭐⭐⭐⭐ 💰 🚀
- **What it does**: Build stateful, multi-actor applications with LLMs using graph-based workflows
- **Best for**: Complex agent workflows, multi-step reasoning, human-in-the-loop systems
- **Features**: State management, branching logic, human approval nodes, persistence
- **Pricing**: Free and open source
- **Unique**: LangChain native with advanced state management for complex agent workflows
```

### **Field Guidelines:**

| Field | Requirements | Example |
|-------|--------------|---------|
| **Tool Name** | Official name with working link | `[OpenAI](https://openai.com/api/)` |
| **Stars** | 2-5 stars based on quality | `⭐⭐⭐⭐⭐` |
| **Cost Icons** | One of: 💰 🔄 💵 🏢 🚀 | `💰 🚀` |
| **What it does** | 1-2 sentences max | Clear, concise function |
| **Best for** | Target audience/use cases | `Production apps, complex reasoning` |
| **Features** | 3-5 key differentiators | Comma-separated list |
| **Pricing** | Cost model or "Free" | `$2.50-10/1M tokens` or `Free` |
| **Unique** | What sets it apart | Why choose over alternatives |

---

## ⭐ Rating System

### **Star Ratings:**

- **⭐⭐⭐⭐⭐ Must-have** - Industry standard, essential tool
  - Dominant in category, exceptional quality
  - Examples: OpenAI GPT-4, LangChain, Transformers

- **⭐⭐⭐⭐ Excellent** - High quality, highly recommended
  - Strong performance, good documentation
  - Examples: Anthropic Claude, MLflow, vLLM

- **⭐⭐⭐ Good** - Solid choice, worth considering
  - Reliable with specific strengths
  - Examples: Cohere, Haystack, FastChat

- **⭐⭐ Decent** - Has specific use cases
  - Limited but functional
  - Examples: Specialized or niche tools

### **Category Icons:**

| Icon | Meaning | Use When |
|------|---------|----------|
| 💰 | **Free/Open Source** | No cost, open source license |
| 💵 | **Paid/Commercial** | Requires payment, commercial license |
| 🔄 | **Freemium** | Free tier + paid options |
| 🚀 | **Trending/New** | Recently launched or major update |
| 🏢 | **Enterprise-focused** | Designed for enterprise use |

### **Rating Guidelines:**

**Consider these factors:**
- ✅ **Quality** - Performance, reliability, accuracy
- ✅ **Documentation** - Clear guides, examples, API docs
- ✅ **Community** - Active development, user base, support
- ✅ **Uniqueness** - What it does that others don't
- ✅ **Maintenance** - Regular updates, bug fixes, active repo

**Be objective:**
- ❌ Don't rate based on personal preference
- ❌ Don't over-rate tools you've built
- ❌ Don't under-rate competitors
- ✅ Focus on objective quality and utility

---

## 📂 Category Guidelines

### **Choosing the Right Category:**

| Category | What Belongs Here | Examples |
|----------|-------------------|----------|
| **Commercial LLM APIs** | Hosted AI services | OpenAI, Anthropic, Google |
| **Open Source Models** | Downloadable model weights | Llama, Mistral, Gemma |
| **Development Frameworks** | Libraries for building apps | LangChain, LlamaIndex |
| **Agent Frameworks** | Multi-agent, autonomous systems | LangGraph, CrewAI, AutoGen |
| **Local Deployment** | Running models locally | Ollama, vLLM, LocalAI |
| **Vector Databases** | Embedding storage & search | Pinecone, Chroma, Weaviate |
| **Fine-tuning Platforms** | Model customization | Unsloth, Together AI, Axolotl |

### **When Creating New Categories:**
- Must have **5+ quality tools** to justify new section
- Should fill a **clear gap** in existing categories
- **Discuss first** in an issue before adding

### **Category Ordering:**
- Tools within categories should be **roughly ordered by importance/popularity**
- **5-star tools first**, then 4-star, etc.
- **Most popular/essential tools** at the top of each rating tier

---

## ✅ Quality Standards

### **We Accept Tools That:**

- ✅ **Actively maintained** - Recent commits, responsive to issues
- ✅ **Well documented** - Clear setup instructions, examples
- ✅ **Actually work** - Tool functions as described
- ✅ **Add unique value** - Solves problems not addressed by existing tools
- ✅ **Relevant to LLMs** - Directly supports LLM development or deployment
- ✅ **Stable enough** - Not alpha/experimental unless exceptionally promising

### **We Don't Accept:**

- ❌ **Abandoned projects** - No updates in 6+ months, unresponsive maintainers
- ❌ **Broken tools** - Doesn't work, broken links, major bugs
- ❌ **Duplicate functionality** - Same as existing tool without clear advantage
- ❌ **Self-promotion** - Tools that exist just to promote a service/company
- ❌ **Off-topic tools** - Not directly related to LLM development
- ❌ **Placeholder/beta** - Tools that aren't ready for public use

### **Link Requirements:**
- ✅ **Working links** - All URLs must be accessible
- ✅ **Official sources** - Link to official website/repo, not tutorials
- ✅ **HTTPS preferred** - Use secure links when available
- ✅ **Permanent links** - Avoid temporary or redirect URLs

---

## 🔍 Review Process

### **What Happens to Your PR:**

1. **Automated Checks** (< 5 minutes)
   - Link validation
   - Format checking
   - Duplicate detection

2. **Manual Review** (1-3 days)
   - Content quality assessment
   - Category fit evaluation
   - Rating appropriateness

3. **Community Feedback** (2-7 days)
   - Public review period for community input
   - Chance for others to provide insights

4. **Final Decision**
   - **Approved**: Merged with attribution
   - **Needs Changes**: Feedback provided for improvements
   - **Rejected**: Clear explanation of why (rare)

### **Speeding Up Review:**

- ✅ **Follow format exactly** - Reduces back-and-forth
- ✅ **Test all links** - Prevents common rejections
- ✅ **Write clear descriptions** - Makes review easier
- ✅ **Use appropriate ratings** - Shows you understand the system
- ✅ **Add to correct category** - Demonstrates understanding

### **Response Times:**
- **Simple additions**: 1-2 days
- **New categories**: 3-7 days
- **Major updates**: 2-5 days
- **Corrections/fixes**: Same day to 2 days

---

## 🎯 Common Scenarios

### **Adding a New Tool:**

```markdown
## 🆕 Adding [Tool Name]

**Category**: [Specify category]
**Why this tool**: [Brief explanation of value]
**Replaces/Complements**: [How it fits with existing tools]

[Include tool in proper format]
```

### **Updating Existing Tool Info:**

```markdown
## 📝 Updating [Tool Name]

**What changed**: [Pricing update/new features/etc.]
**Source**: [Link to announcement/documentation]

[Include updated tool entry]
```

### **Reporting Issues:**

```markdown
## 🐛 Issue with [Tool Name]

**Problem**: [Broken link/wrong info/etc.]
**Expected**: [What it should be]
**Current**: [What it shows now]

[Additional context]
```

### **Suggesting Categories:**

```markdown
## 📂 New Category Suggestion: [Category Name]

**Why needed**: [Gap in current categorization]
**Tools to include**: [List 5+ relevant tools]
**How it differs**: [From existing categories]

[Rationale for why this improves organization]
```

---

## 🆘 Getting Help

### **Common Questions:**

**Q: My tool was rejected, why?**
A: Check our quality standards. Common reasons: inactive maintenance, broken links, duplicate functionality, or poor documentation.

**Q: What rating should I give my tool?**
A: Be honest and objective. Compare to similar tools in the same category. When in doubt, rate conservatively.

**Q: Can I add a tool I built?**
A: Yes, but be objective about its quality and utility. Self-promotional tools that don't add value will be rejected.

**Q: How do I know which category to use?**
A: Look at similar tools and see where they're placed. When in doubt, ask in an issue before submitting.

**Q: Can I suggest reorganizing sections?**
A: Yes! Open an issue with your proposal and rationale. We're always looking to improve organization.

### **Getting Support:**

- 💬 **General questions**: Open a [Discussion](https://github.com/dr-saad-la/awesome-llm-tools/discussions)
- 🐛 **Bug reports**: Use [Bug Report template](https://github.com/dr-saad-la/awesome-llm-tools/issues/new?template=bug_report.md)
- 💡 **Tool suggestions**: Use [Tool Suggestion template](https://github.com/dr-saad-la/awesome-llm-tools/issues/new?template=tool_suggestion.md)
- 📧 **Direct contact**: Only for sensitive issues

### **Response Expectations:**
- **Issues**: Response within 2-3 days
- **PRs**: Initial review within 1-3 days
- **Discussions**: Community-driven, may vary

---

## 🙏 Recognition

### **How We Thank Contributors:**

- ✅ **Commit attribution** - Your name in git history
- ✅ **Release notes** - Major contributions highlighted
- ✅ **Community recognition** - Top contributors featured
- ✅ **Maintainer invites** - Outstanding contributors may become maintainers

### **Contribution Types We Value:**

- 🔍 **Tool discoveries** - Finding and adding valuable new tools
- 📝 **Information updates** - Keeping existing entries current
- 🐛 **Issue reporting** - Helping maintain quality
- 📚 **Documentation** - Improving guides and explanations
- 🤖 **Automation** - Building tools to help maintain the list
- 💬 **Community support** - Helping other contributors

---

## 📜 Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By participating, you agree to:

- ✅ **Be respectful** of different viewpoints and experiences
- ✅ **Focus on constructive feedback** and collaboration
- ✅ **Show empathy** towards other community members
- ✅ **Accept responsibility** for mistakes and learn from them

**Unacceptable behavior includes**: harassment, trolling, spam, or disrespectful conduct. Report issues to the maintainers.

---

## 🚀 Ready to Contribute?

**Start here:**
1. **Browse existing tools** to understand our standards
2. **Check open issues** for requested tools or improvements
3. **Fork the repository** and make your changes
4. **Submit a pull request** following our templates

**Thank you for helping make Awesome LLM Tools the most comprehensive and up-to-date resource for the LLM development community!** 🎉

---

*Last updated: January 2025*
*Questions? Open an [issue](https://github.com/dr-saad-la/awesome-llm-tools/issues) or [discussion](https://github.com/dr-saad-la/awesome-llm-tools/discussions)*
