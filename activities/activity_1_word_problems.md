# Activity 1: LLM word problems

**Objective:** Practice basic prompting techniques and chain-of-thought reasoning by solving word problems with an LLM.

**Duration:** 30-45 minutes

---

## Overview

In this activity, you'll work through a series of word problems using an LLM. You'll start with basic prompts and progressively apply techniques from Lesson 46 to improve the results. This hands-on practice will help you understand when and how to use different prompting strategies.

---

## Setup

### Use the Gradio chatbot (recommended)

The Gradio chatbot provides an interactive web interface where you can experiment with different system prompts directly in the UI. This is perfect for learning prompting strategies without editing code.

**Start the Gradio chatbot:**

```bash
# Make sure your backend is running:
# For Ollama:
ollama serve

# For llama.cpp, see the systemd-deployment docs

# Then start the chatbot:
python src/gradio_chatbot.py
```

**Access the interface:**

Open your browser to `http://localhost:7860` (or the URL shown in the terminal).

### How to use the Gradio chatbot for word problems

The interface has three main sections:

1. **Backend selector** - Choose between Ollama or llama.cpp
2. **System Prompt field** - This is where you'll experiment with different prompting strategies
3. **Chat interface** - Where you send messages and see responses

**Basic workflow:**

1. **Start with a default system prompt:**
   ```
   You are a helpful assistant that solves math word problems.
   ```

2. **Enter your word problem** in the chat and observe the response

3. **Modify the system prompt** to test different techniques:
   ```
   You are a helpful math tutor. When solving problems, always show your 
   work step by step. Break down complex problems into smaller steps and 
   explain your reasoning clearly.
   ```

4. **Try the same problem again** by clearing the chat (use the Clear button) and re-entering it

5. **Compare results** - which system prompt produced better reasoning?

### Prompting strategies to experiment with

Try modifying the system prompt with these techniques:

**Chain-of-thought prompting:**
```
You are a helpful math tutor. Always solve problems step by step. 
Show your reasoning for each step before moving to the next one.
```

**Structured output:**
```
You are a helpful math tutor. For each problem:
1. Identify what you know
2. Identify what you need to find
3. Plan your approach
4. Solve step by step
5. State your final answer clearly
```

**Self-verification:**
```
You are a careful math tutor. Solve problems step by step, then 
double-check your work by working backwards or using a different method.
```

**Role-specific:**
```
You are an elementary school math teacher. Explain your reasoning 
simply and clearly, as if teaching a 10-year-old student.
```

### Tips for success

- **Test one strategy at a time:** Change only the system prompt between attempts
- **Clear the chat between tests:** Use the Clear button to start fresh conversations
- **Document your experiments:** Note which prompting strategy worked best for each problem
- **Compare side-by-side:** Try the same problem with different system prompts and compare
- **Backend matters:** You might get different results with Ollama vs llama.cpp

---

## Word problems

For each problem below:
1. Start with a **simple system prompt** (e.g., "You are a helpful assistant")
2. Enter the problem and **observe the result** - did it work? What went wrong?
3. **Modify the system prompt** to apply a prompting technique
4. Clear the chat and **try the problem again**
5. **Compare the results** - what changed?

### Problem 1: The cookie problem

**Question:** A baker made 60 cookies. She sold 2/3 of them in the morning and gave away half of the remaining ones in the afternoon. How many cookies does she have left?

**Your task:**
1. Ask the LLM with just the question
2. If it gets the wrong answer or makes mistakes, try adding:
   - **Chain-of-thought:** "Let's think step by step."
   - **Explicit structure:** Ask it to show its work for each step

**Hints:**
- Look for arithmetic errors (especially with fractions)
- The answer should be 10 cookies
- Chain-of-thought often helps with multi-step math

---

### Problem 2: The meeting scheduler

**Question:** Alice needs to schedule a meeting with Bob, Carol, and Dave. Alice is free on Monday, Wednesday, and Friday. Bob is free on Tuesday, Wednesday, and Thursday. Carol is free on Monday and Wednesday. Dave is free on Wednesday and Friday. When can they all meet?

**Your task:**
1. Try with just the question
2. If the LLM struggles or gives unclear reasoning, try:
   - **Structured prompt:** Ask it to list each person's availability first
   - **Chain-of-thought:** Request step-by-step reasoning
   - **Output format:** Ask for the answer in a specific format

**Hints:**
- The answer is Wednesday
- Try asking it to create a table or matrix first
- Structured prompts help with complex logical problems

---

### Problem 3: The age puzzle

**Question:** Sarah is twice as old as Tom was when Sarah was as old as Tom is now. The sum of their current ages is 42. How old are they?

**Your task:**
1. Try basic prompt
2. This one is tricky! Try different approaches:
   - **Chain-of-thought:** "Let's work through this step by step"
   - **Context:** "This is an algebra problem. Let's define variables..."
   - **Constraints:** "Show your algebraic equations clearly"

**Hints:**
- This is a classic algebra problem
- Sarah is 24, Tom is 18
- The LLM may need multiple attempts or guidance
- Consider breaking it into smaller steps

---

### Problem 4: The train problem (with a trick!)

**Question:** A train leaves Station A heading to Station B at 60 mph. Another train leaves Station B heading to Station A at 40 mph. The stations are 300 miles apart. How far apart are the trains after 2 hours?

**Your task:**
1. Try basic prompt - does it answer correctly?
2. Try these techniques:
   - **Chain-of-thought:** Break down the problem
   - **Constraints:** "Calculate each train's distance traveled separately first"

**Hints:**
- Each train travels for 2 hours
- Train A: 60 mph × 2 = 120 miles
- Train B: 40 mph × 2 = 80 miles
- Distance remaining: 300 - 120 - 80 = 100 miles
- Watch for the LLM trying to solve "when will they meet" instead

---

### Problem 5: The tricky one

**Question:** A farmer has 17 sheep. All but 9 die. How many sheep are left?

**Your task:**
1. Try with just the question - does it get it right?
2. This is a language comprehension problem, not math!
3. If it gets it wrong, try:
   - **Chain-of-thought** to see its reasoning
   - **Rephrasing** the question for clarity

**Hints:**
- "All but 9" means 9 survive
- The answer is 9
- LLMs often want to do math (17 - 9) and get 8
- This illustrates the importance of careful problem reading

---

## Bonus challenge: Create your own

Think of a word problem that might be tricky for an LLM:
- Multi-step reasoning
- Tricky wording
- Common misconceptions
- Requires careful reading

Test it and see what prompting techniques help!

---

## Reflection questions

After completing the problems, consider:

1. **Which prompting techniques helped most?**
   - Chain-of-thought?
   - Structured formatting?
   - Explicit constraints?

2. **What types of problems were hardest for the LLM?**
   - Pure math?
   - Logic puzzles?
   - Language ambiguity?

3. **How did you adapt your prompting strategy?**
   - What worked?
   - What didn't?

4. **When would you use chain-of-thought in real applications?**

---

## Key takeaways

- **Chain-of-thought ("Let's think step by step")** significantly improves multi-step reasoning
- **Structure and formatting** help the LLM organize complex information
- **Language ambiguity** can trick LLMs just like humans
- **Iteration is key** - your first prompt rarely works perfectly
- **Different problems need different techniques** - there's no one-size-fits-all approach

---

## Next steps

Apply these techniques in Activity 2: Text summarization, where you'll build a practical application using prompting strategies.
