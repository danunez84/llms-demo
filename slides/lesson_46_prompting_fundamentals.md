---
marp: true
theme: default
paginate: true
style: |
  section {
    background-color: #1a1a2e;
    color: #e0e0e0;
  }
  section h1, section h2, section h3 {
    color: #e2b55a;
  }
  section a {
    color: #93c5fd;
  }
  section strong {
    color: #f0f0f0;
  }
  section table th {
    background-color: #2a2a4a;
    color: #e2b55a;
  }
  section table td {
    background-color: #1e1e36;
    color: #e0e0e0;
  }
  section code {
    background-color: #2a2a4a;
    color: #e0e0e0;
  }
  section pre {
    background-color: #12122a;
  }
  section::after {
    color: #888;
  }
---

# Lesson 46: Prompting fundamentals

**How to communicate effectively with LLMs**

---

## Recap: the LLM application stack

From last lesson:

```
┌─────────────────────────────────┐
│  Your application (Python, JS)  │  <- Your prompts live here
├─────────────────────────────────┤
│  Framework (LangChain, etc.)    │  <- Orchestration, prompt templates
├─────────────────────────────────┤
│  Inference server (Ollama, API) │  <- Executes the model
├─────────────────────────────────┤
│  Model (Llama, GPT, etc.)       │  <- The weights that generate text
└─────────────────────────────────┘
```

**Today:** How to write **effective prompts** that get the results you want

---

## Why prompting matters

LLMs are powerful, but they don't read your mind:

- **Same model, different results**: the quality of your prompt determines the quality of the output
- **No code changes needed**: better prompts = better results without retraining
- **Universal skill**: applies to all LLMs (GPT, Claude, Llama, etc.)

**Key insight:** Prompting is the primary interface for controlling LLM behavior

---

## Today's outline

1. **What is a prompt?** - anatomy and components
2. **Basic techniques** - clear instructions, examples, formatting
3. **System prompts** - setting behavior and persona
4. **Few-shot learning** - teaching by example
5. **Common pitfalls** - what to avoid

---

# What is a prompt?

Understanding the anatomy of effective prompts

- Text input that guides the model
- Components: instruction, context, examples, constraints
- Different from traditional programming

---

## Anatomy of a prompt

A well-structured prompt typically includes:

1. **Instruction** - what you want the model to do
2. **Context** - background information the model needs
3. **Input data** - the specific content to process
4. **Output format** - how you want the response structured

**Example:** "Summarize this customer review in one sentence focusing on sentiment and key features. We sell kitchen appliances..."

---

## How prompts differ from code

| Traditional code | Prompting |
|------------------|-----------|
| Precise syntax required | Natural language, more flexible |
| Deterministic (same input = same output) | Stochastic (randomness, temperature) |
| Explicit logic | Implicit pattern matching |
| Breaks immediately on errors | Degrades gracefully (but unpredictably) |

**Key takeaway:** Prompting is more like **communication** than programming - clarity and context matter more than syntax

---

## Prompt complexity spectrum

**Simple:** Direct questions ("What is the capital of France?")

**Structured:** Clear components for complex tasks

**Advanced:** Chain-of-thought, examples, constraints

**Key insight:** Match complexity to the task - start simple and add structure only when needed

---

# Basic techniques

Five fundamental strategies for better prompts

---

## Basic techniques overview

**1. Be specific and clear**
- Replace vague requests with detailed instructions
- Example: "Write about dogs" → "Write a 3-paragraph explanation of why dogs are good pets for families"

**2. Provide context**
- Give the model background information it needs
- Example: Include your goals, constraints, and relevant details

**3. Use formatting**
- Structure prompts with clear sections
- Use headers, bullets, and separators to organize information

---

## Basic techniques overview (continued)

**4. Set constraints**
- Specify length, tone, format, and what to avoid
- Example: "Maximum 100 words, no technical jargon, use analogies"

**5. Iterate and refine**
- Start simple, test, observe, refine, repeat
- Treat prompt engineering like debugging

**Key insight:** Clear, structured prompts with context and constraints produce better results

---

# System prompts

Setting global behavior for conversations

---

## What are system prompts?

**System prompt:** Sets the "rules" for the entire conversation
- Processed first, before any user input
- Defines persona, tone, output format, constraints

**User prompt:** The actual queries or tasks

**Think of it as:** System prompt = job description, User prompt = specific task

---

## System vs user prompts

**System prompt:** Sets global behavior for the entire conversation
- Example: "You are a helpful Python tutor. Explain simply."

**User prompt:** The actual queries or tasks
- Example: "How do I read a CSV file?"

**Key difference:** System prompt persists across all interactions, user prompts are individual messages

---

## Common system prompt patterns

**1. Persona/role:** "You are an expert data scientist..."

**2. Tone and style:** "Respond in a friendly, conversational tone..."

**3. Output format:** "Always respond in valid JSON..."

**4. Constraints:** "Never discuss pricing, escalate to human if asked..."

**Best practices:** Define role clearly, set tone/format expectations, keep concise (100-300 words), test thoroughly

**Key insight:** System prompts guide behavior but aren't foolproof - always validate

---

# Few-shot learning

Teaching by example rather than just instructions

---

## Zero-shot vs few-shot

**Zero-shot:** Instructions only
- "Classify the sentiment of this review"

**Few-shot:** Instructions + examples
- Show 2-5 examples of input → output pairs
- Model learns the pattern from examples

**Why it works:** Examples anchor the model's understanding of format and style

**When to use:** Custom formats, specific output structures, domain-specific tasks

---

## Few-shot best practices

**Use 2-5 examples** - more isn't always better

**Show edge cases** - include tricky inputs

**Be consistent** - examples should follow the same pattern

**Trade-off:** Uses context window space, may be slower/costlier

**Rule of thumb:** Start with zero-shot, add examples only if needed

---



---

# Common pitfalls

What to avoid when prompting

---

## Five common mistakes

**1. Ambiguity and vagueness**
- Problem: "Make this better" → Better how?
- Fix: Be specific about what you want improved

**2. Over-prompting**
- Problem: 3 paragraphs of credentials to reverse a string
- Fix: Keep it simple - match complexity to the task

**3. Assuming knowledge**
- Problem: Referencing internal projects/frameworks the model can't know
- Fix: Provide definitions and context

---

## Five common mistakes (continued)

**4. Ignoring model limitations**
- No real-time data, can't run code, math errors, hallucination
- Fix: Use tools and external APIs to fill gaps

**5. Not testing edge cases**
- Problem: Only testing happy path
- Fix: Test with empty inputs, malformed data, extreme values

**Key insight:** Test thoroughly and treat prompts like code - check edge cases, consistency, and failure modes

---

# Summary

Key takeaways from today

---

## What we covered

**1. Anatomy of a prompt**
- Instruction, context, input, output format

**2. Basic techniques**
- Be specific, provide context, use formatting, set constraints, iterate

**3. System prompts**
- Set global behavior and persona for conversations

**4. Few-shot learning**
- Teach by example when zero-shot isn't enough

**5. Common pitfalls**
- Ambiguity, over-prompting, assuming knowledge, ignoring limits, not testing

---

## Next lesson preview

**Lesson 47: Advanced prompting strategies**

We'll cover:
- Chain-of-thought prompting
- ReAct (reasoning + acting)
- Tree of thoughts
- Self-consistency
- Prompt chaining and decomposition

These techniques unlock more complex LLM capabilities and improve reliability.

---
