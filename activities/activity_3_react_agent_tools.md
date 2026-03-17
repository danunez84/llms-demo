# Activity 3: Extending the ReAct agent

**Objective:** Enhance the ReAct agent chatbot by adding custom tools and testing multi-step reasoning with real-world problems.

**Duration:** 45-60 minutes

---

## Overview

In this activity, you'll extend the ReAct agent chatbot by adding a new tool that provides additional functionality. You'll learn how LangChain tools work, how agents decide which tools to use, and how to debug tool execution. This hands-on practice will deepen your understanding of the ReAct (Reasoning + Acting) pattern.

---

## Part 1: Understanding the existing tools

Before adding a new tool, let's examine how the current tools work.

### Step 1: Review the tools module

Open `src/tools.py` and examine the three existing tools:

1. **calculator** - Performs arithmetic calculations
2. **get_current_date** - Returns today's date
3. **days_between** - Calculates days between two dates

### Step 2: Understand the tool decorator

Notice how each tool is defined:

```python
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Performs arithmetic calculations on mathematical expressions.
    
    Use this tool when you need to calculate numbers, percentages, or solve
    math problems. Supports addition (+), subtraction (-), multiplication (*),
    division (/), modulo (%), and exponentiation (**).
    
    Args:
        expression: A mathematical expression as a string
    
    Returns:
        The calculated result as a string
    """
    # Implementation here
```

**Key observations:**
- The `@tool` decorator converts a function into a LangChain tool
- The docstring is critical - the agent reads it to understand when and how to use the tool
- Input parameters must be simple types (strings, numbers, booleans)
- Return values should be strings (agents work best with text)

### Step 3: Test the existing agent

Run the ReAct agent chatbot:

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, run the agent
python src/react_agent_chatbot.py
```

Try these questions and observe the reasoning process:
1. "How many days until Christmas from today?"
2. "Calculate 15% tip on a $47.50 bill"
3. "What's the date 100 days from today?"

**Questions to consider:**
- How does the agent decide which tool to use?
- What happens in the reasoning panel?
- How does the agent chain multiple tool calls together?

---

## Part 2: Add a new tool

Now you'll add a new tool to expand the agent's capabilities. You'll need to study the existing tools in `src/tools.py` to understand the pattern, then implement your own.

### Learning from existing tools

Before writing your tool, examine how the existing tools are structured:

1. **Look at the `@tool` decorator usage**
2. **Study the docstring format** - notice how it describes when to use the tool
3. **Examine parameter types and return types**
4. **See how error handling is implemented**

Pay special attention to:
- How `calculator` handles string input and performs operations
- How `get_current_date` works with no parameters
- How `days_between` parses date strings and calculates differences

### Choose a tool to implement

Pick one of these options:

### Option A: Temperature converter (Recommended for beginners)

**Tool name:** `temperature_converter`

**What it should do:**
- Convert temperatures between Fahrenheit and Celsius
- Take three parameters: `value` (the temperature), `from_unit` ('F' or 'C'), and `to_unit` ('F' or 'C')
- Return the converted temperature as a string

**Conversion formulas:**
- Fahrenheit to Celsius: (F - 32) × 5/9
- Celsius to Fahrenheit: (C × 9/5) + 32

**Test cases to handle:**
- temperature_converter(32, "F", "C") should return "0.0"
- temperature_converter(100, "C", "F") should return "212.0"
- temperature_converter(25, "C", "C") should return "25" (same unit)

**Hints:**
- Look at how `calculator` handles string input and converts it
- Consider using `.upper()` to handle 'f'/'F' and 'c'/'C'
- Wrap logic in a try/except block like the other tools

### Option B: Text analyzer

**Tool name:** `text_analyzer`

**What it should do:**
- Count words, characters, or sentences in text
- Take two parameters: `text` (the string to analyze) and `count_type` ('words', 'characters', or 'sentences')
- Return the count as a string

**Counting logic:**
- Words: split by spaces
- Characters: count all characters including spaces
- Sentences: count punctuation marks (., !, ?)

**Test cases to handle:**
- text_analyzer("Hello world", "words") should return "2"
- text_analyzer("Hello!", "characters") should return "6"
- text_analyzer("Hi! How are you?", "sentences") should return "2"

**Hints:**
- Use Python string methods like `.split()`, `len()`, `.count()`
- Handle case-insensitive input for count_type
- Return helpful error messages for invalid count_type values

### Option C: Design your own

Create a custom tool based on your interests:

**Ideas:**
- **unit_converter** - Convert between miles/km, pounds/kg, inches/cm
- **factorial** - Calculate factorial of a number (useful for statistics questions)
- **is_prime** - Check if a number is prime (returns "True" or "False")
- **string_reverser** - Reverse text or check if it's a palindrome
- **percentage_calculator** - Calculate "X% of Y" or "X is what % of Y"

**Requirements for any custom tool:**
- Clear, detailed docstring explaining when to use it
- Include examples in the docstring
- Handle errors gracefully with try/except
- Return strings (not integers, booleans, etc.)
- Use simple parameter types (str, int, float)

---

## Part 3: Register your tool with the agent

After creating your tool, you need to add it to the agent's tool list.

### Step 1: Import your tool

In `src/react_agent_chatbot.py`, find the tool imports section:

```python
# Import our tools
from tools import calculator, get_current_date, days_between
```

Add your new tool to the import:

```python
# Import our tools
from tools import calculator, get_current_date, days_between, temperature_converter
```

### Step 2: Add to the tools list

Find the TOOLS list:

```python
# --- Tools list ---
TOOLS = [calculator, get_current_date, days_between]
```

Add your tool:

```python
# --- Tools list ---
TOOLS = [calculator, get_current_date, days_between, temperature_converter]
```

### Step 3: Update the UI description (optional)

Find the description in the Gradio interface:

```python
**Available tools:** Calculator, Current Date, Days Between Dates
```

Update it to include your tool:

```python
**Available tools:** Calculator, Current Date, Days Between Dates, Temperature Converter
```

---

## Part 4: Test your enhanced agent

### Step 1: Restart the agent

Stop the running agent (Ctrl+C) and restart it:

```bash
python src/react_agent_chatbot.py
```

### Step 2: Test individual tool usage

Try questions that use only your new tool:

**For temperature_converter:**
- "What is 98.6 degrees Fahrenheit in Celsius?"
- "Convert 0 Celsius to Fahrenheit"
- "Is 25 Celsius hot or cold in Fahrenheit?"

**For text_analyzer:**
- "How many words are in the sentence: 'The quick brown fox jumps over the lazy dog'?"
- "Count the characters in 'Hello, World!'"

**For your custom tool:**
- *Create appropriate test questions*

### Step 3: Test multi-tool reasoning

Now try questions that require using multiple tools together:

**For temperature_converter:**
1. "If it's 30 degrees Celsius today, and it will be 5 degrees warmer tomorrow, what will the temperature be in Fahrenheit?"
2. "Water boils at 100°C. If I heat water from room temperature (20°C) to boiling, how many degrees Fahrenheit did it increase?"

**For text_analyzer:**
1. "Count the words in this sentence and multiply the result by 5: 'AI is transforming technology'"
2. "How many characters are in the current date?"

**General multi-step challenges:**
1. "Today's date has how many characters in it? Double that number."
2. "If there are 90 days until Christmas and each day has 24 hours, how many hours is that?"
3. "Calculate 20% of 350, then tell me what that would be in a different unit" (adapt based on your tool)

---

## Part 5: Debugging and improvement

### Common issues and solutions

**Issue: Agent doesn't use your tool**

*Possible causes:*
- Docstring unclear or missing
- Tool not added to TOOLS list
- Tool name too similar to existing tool

*Solution:*
- Improve the docstring description
- Be explicit about when to use the tool
- Add examples in the docstring

**Issue: Tool returns errors**

*Debugging steps:*
1. Check the reasoning panel to see what arguments the agent passed
2. Add print statements in your tool function:
   ```python
   print(f"DEBUG: Received {value=}, {from_unit=}, {to_unit=}")
   ```
3. Test the tool function directly in Python:
   ```python
   from src.tools import temperature_converter
   print(temperature_converter.run({"value": 32, "from_unit": "F", "to_unit": "C"}))
   ```

**Issue: Agent chains tools incorrectly**

*This is actually interesting behavior to observe:*
- The agent is making decisions based on its understanding
- Try rephrasing your question to be clearer
- Check if the system prompt needs adjustment

### Improvement exercise

Once your tool works, try improving it:

1. **Add validation:** Check for invalid inputs before processing
2. **Expand functionality:** Add more conversion types or analysis options
3. **Better error messages:** Make errors more helpful for the agent
4. **Handle edge cases:** What if the input is empty? Very large? Negative?

---

## Resources

- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/tools/)
- [Building Custom Tools](https://python.langchain.com/docs/modules/tools/custom_tools)
- [ReAct Paper](https://arxiv.org/abs/2210.03629) - Original research on ReAct pattern
- ReAct demo code: `src/react_agent_chatbot.py`
- Existing tools: `src/tools.py`