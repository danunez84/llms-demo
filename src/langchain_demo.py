"""LangChain basics demo

This demo demonstrates core LangChain concepts:
1. Chat models and LLM wrappers
2. Chat prompt templates
3. Output parsers
4. Basic chains

The demo uses Gradio to provide an interactive interface where you can:
- Try different prompt templates
- See structured output parsing
- Experiment with chained operations

Usage:
    python src/langchain_demo.py
"""

import os
from typing import List
import gradio as gr
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field


# Load environment variables
load_dotenv()

# --- Configuration ---

temperature = 0.1

# --- Initialize backends ---

ollama_model = 'qwen2.5:3b'
ollama_client = ChatOllama(model=ollama_model, temperature=temperature)

llamacpp_server = os.environ.get('PERDRIZET_URL', 'localhost:8502')

if llamacpp_server.startswith('localhost') or llamacpp_server.startswith('127.'):
    llamacpp_api_key = os.environ.get('LLAMA_API_KEY', 'dummy')
    llamacpp_base_url = f'http://{llamacpp_server}/v1'
else:
    llamacpp_api_key = os.environ.get('PERDRIZET_API_KEY')
    llamacpp_base_url = f'https://{llamacpp_server}/v1'

llamacpp_client = ChatOpenAI(
    base_url=llamacpp_base_url,
    api_key=llamacpp_api_key,
    timeout=120.0,
    model='gpt-oss-20b',
    temperature=temperature
)

llamacpp_model = 'gpt-oss-20b'


# --- Pydantic models for output parsing ---

class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="Overall sentiment: positive, negative, or mixed")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    key_phrases: List[str] = Field(description="Important phrases that support the sentiment")


class RecipeInfo(BaseModel):
    name: str = Field(description="Name of the dish")
    cuisine: str = Field(description="Type of cuisine")
    ingredients: List[str] = Field(description="List of main ingredients")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard")


class PersonInfo(BaseModel):
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job or profession")
    location: str = Field(description="City or country where person lives")


# --- Demo functions ---

def demo_simple_chain(text: str, backend: str) -> tuple[str, str]:
    """Demo 1: Simple chain with prompt template and string output."""

    llm = ollama_client if backend == 'Ollama' else llamacpp_client
    
    # Create a simple prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains concepts concisely."),
        ("human", "Explain {topic} in 2-3 sentences."),
    ])
    
    # Create chain: prompt -> model -> string parser
    chain = prompt | llm | StrOutputParser()
    
    # Execute
    result = chain.invoke({"topic": text})
    
    explanation = f"""**Chain components:**
    1. Prompt template with system message and variable placeholder
    2. {backend} chat model
    3. StrOutputParser (extracts text from AIMessage)

    **Input:** topic = "{text}"
    """
    
    return result, explanation


def demo_sentiment_analysis(text: str, backend: str) -> tuple[str, str]:
    """Demo 2: Chain with structured output (JSON)."""

    llm = ollama_client if backend == 'Ollama' else llamacpp_client
    
    # Create output parser
    parser = JsonOutputParser(pydantic_object=SentimentAnalysis)
    
    # Create prompt with format instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a sentiment analysis expert. Analyze the sentiment of the given text.
        {format_instructions}"""),
        ("human", "{text}"),
    ])
    
    # Create chain
    chain = prompt | llm | parser
    
    try:
        # Execute
        result = chain.invoke({
            "text": text,
            "format_instructions": parser.get_format_instructions()
        })
        
        # Format output
        output = f"""**Sentiment:** {result['sentiment']}
        **Confidence:** {result['confidence']:.2%}
        **Key phrases:**
        {chr(10).join(f"- {phrase}" for phrase in result['key_phrases'])}"""
                
                explanation = f"""**Chain components:**
        1. Prompt template with format instructions
        2. {backend} chat model
        3. JsonOutputParser with Pydantic schema

        **Schema fields:**
        - sentiment (str): positive/negative/mixed
        - confidence (float): 0.0 to 1.0
        - key_phrases (list[str]): Supporting evidence
        """
        
        return output, explanation
    
    except Exception as e:
        return f"Error: {str(e)}", f"An error occurred during parsing. Try a different input or backend."


def demo_entity_extraction(text: str, backend: str, entity_type: str) -> tuple[str, str]:
    """Demo 3: Entity extraction with different schemas."""

    llm = ollama_client if backend == 'Ollama' else llamacpp_client
    
    # Choose schema based on entity type
    if entity_type == "Person":
        schema = PersonInfo

    elif entity_type == "Recipe":
        schema = RecipeInfo

    else:
        return "Invalid entity type", "Please select a valid entity type"
    
    parser = JsonOutputParser(pydantic_object=schema)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Extract {entity_type} information from the text.
        {format_instructions}"""),
        ("human", "{text}"),
    ])
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "text": text,
            "entity_type": entity_type.lower(),
            "format_instructions": parser.get_format_instructions()
        })
        
        # Format output nicely
        output = "**Extracted information:**\n\n"

        for key, value in result.items():
            if isinstance(value, list):

                output += f"**{key.replace('_', ' ').title()}:**\n"
                output += "\n".join(f"- {item}" for item in value) + "\n\n"

            else:
                output += f"**{key.replace('_', ' ').title()}:** {value}\n"
        
        explanation = f"""**Chain components:**
        1. Prompt template with dynamic entity type
        2. {backend} chat model
        3. JsonOutputParser with {entity_type} schema

        **Selected schema:** {entity_type}
        **Fields:** {', '.join(schema.model_fields.keys())}
        """
        
        return output, explanation
    
    except Exception as e:
        return f"Error: {str(e)}", f"Make sure your text contains {entity_type.lower()} information."


def demo_few_shot(text: str, backend: str) -> tuple[str, str]:
    """Demo 4: Few-shot learning with prompt templates."""

    llm = ollama_client if backend == 'Ollama' else llamacpp_client
    
    # Few-shot prompt with examples
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a text style classifier. Classify the writing style as: technical, casual, formal, or creative."),
        ("human", "The efficacy of the proposed methodology was validated through rigorous experimental procedures."),
        ("ai", "technical"),
        ("human", "Hey! Just wanted to say this app is super cool and easy to use."),
        ("ai", "casual"),
        ("human", "We are pleased to inform you that your application has been approved."),
        ("ai", "formal"),
        ("human", "The moonlight danced across the waves like silver ribbons weaving through the night."),
        ("ai", "creative"),
        ("human", "{text}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"text": text})
    
    explanation = """**Chain components:**
    1. Prompt template with 4 few-shot examples
    2. Chat model (learns from examples)
    3. StrOutputParser

    **Few-shot examples:**
    - Technical: "efficacy", "methodology", "validated"
    - Casual: "Hey!", "super cool", informal language
    - Formal: "We are pleased", official tone
    - Creative: Metaphors, descriptive language

    The model learns the pattern from examples!
    """
    
    return result.strip(), explanation


# --- Build Gradio UI ---

with gr.Blocks(title='LangChain Basics Demo') as demo:
    
    gr.Markdown("""
    # LangChain basics demo
    
    Explore core LangChain concepts with interactive examples:
    - **Prompt templates** with variable substitution
    - **Structured output parsing** with Pydantic schemas
    - **Basic chains** composing multiple steps
    - **Few-shot learning** with example-driven prompts
    """)
    
    # Backend selector (shared across all tabs)
    with gr.Row():
        backend_selector = gr.Radio(
            choices=['Ollama', 'llama.cpp'],
            value='Ollama',
            label='Model backend',
            info=f'Ollama: {ollama_model} | llama.cpp: {llamacpp_model} @ {llamacpp_base_url}'
        )
    
    # Tabs for different demos
    with gr.Tabs():
        
        # Tab 1: Simple chain
        with gr.Tab("1. Simple chain"):
            gr.Markdown("""
            **Concept:** Basic chain with prompt template and string output
            
            The chain: `prompt_template | llm | string_parser`
            """)
            
            with gr.Row():
                with gr.Column():

                    simple_input = gr.Textbox(
                        label="Topic to explain",
                        placeholder="e.g., machine learning, photosynthesis, blockchain",
                        value="neural networks"
                    )

                    simple_btn = gr.Button("Explain", variant="primary")
                
                with gr.Column():
                    simple_output = gr.Textbox(label="Explanation", lines=5)
                    simple_info = gr.Markdown()
            
            simple_btn.click(
                fn=demo_simple_chain,
                inputs=[simple_input, backend_selector],
                outputs=[simple_output, simple_info]
            )
        
        # Tab 2: Sentiment analysis
        with gr.Tab("2. Sentiment analysis"):
            gr.Markdown("""
            **Concept:** Structured output with JSON parser and Pydantic schema
            
            The chain: `prompt_template | llm | json_parser`
            """)
            
            with gr.Row():
                with gr.Column():

                    sentiment_input = gr.Textbox(
                        label="Text to analyze",
                        placeholder="Enter a review, comment, or any text...",
                        lines=4,
                        value="I absolutely love this product! The quality is outstanding and it exceeded my expectations."
                    )
                    sentiment_btn = gr.Button("Analyze sentiment", variant="primary")
                
                with gr.Column():
                    sentiment_output = gr.Textbox(label="Analysis results", lines=8)
                    sentiment_info = gr.Markdown()
            
            sentiment_btn.click(
                fn=demo_sentiment_analysis,
                inputs=[sentiment_input, backend_selector],
                outputs=[sentiment_output, sentiment_info]
            )
        
        # Tab 3: Entity extraction
        with gr.Tab("3. Entity extraction"):
            gr.Markdown("""
            **Concept:** Different Pydantic schemas for different entity types
            
            Try different schemas to see how the same chain extracts different information!
            """)
            
            with gr.Row():
                with gr.Column():

                    entity_type = gr.Radio(
                        choices=["Person", "Recipe"],
                        value="Person",
                        label="Entity type to extract"
                    )
    
                    entity_input = gr.Textbox(
                        label="Text to extract from",
                        placeholder="Enter text containing person or recipe information...",
                        lines=4,
                        value="Sarah Chen is a 34-year-old software engineer living in San Francisco."
                    )
    
                    entity_btn = gr.Button("Extract entities", variant="primary")
                
                with gr.Column():
                    entity_output = gr.Textbox(label="Extracted information", lines=10)
                    entity_info = gr.Markdown()
            
            entity_btn.click(
                fn=demo_entity_extraction,
                inputs=[entity_input, backend_selector, entity_type],
                outputs=[entity_output, entity_info]
            )
            
            # Example text updater
            def update_example(entity_type):
                if entity_type == "Person":
                    return "Sarah Chen is a 34-year-old software engineer living in San Francisco."
                else:
                    return "Pad Thai is a popular Thai stir-fried noodle dish. Main ingredients include rice noodles, eggs, tofu, bean sprouts, and peanuts. It's considered medium difficulty to make."
            
            entity_type.change(
                fn=update_example,
                inputs=[entity_type],
                outputs=[entity_input]
            )
        
        # Tab 4: Few-shot learning
        with gr.Tab("4. Few-shot learning"):
            gr.Markdown("""
            **Concept:** Learning from examples in the prompt
            
            The prompt includes 4 examples showing different writing styles.
            The model learns the pattern and classifies new text!
            """)
            
            with gr.Row():
                with gr.Column():

                    fewshot_input = gr.Textbox(
                        label="Text to classify",
                        placeholder="Enter some text to classify its style...",
                        lines=4,
                        value="The implementation utilizes a recursive algorithm to optimize computational efficiency."
                    )
    
                    fewshot_btn = gr.Button("Classify style", variant="primary")
                
                with gr.Column():
                    fewshot_output = gr.Textbox(label="Classified style", lines=2)
                    fewshot_info = gr.Markdown()
            
            fewshot_btn.click(
                fn=demo_few_shot,
                inputs=[fewshot_input, backend_selector],
                outputs=[fewshot_output, fewshot_info]
            )
            
            gr.Markdown("""
            **Try these examples:**
            - Technical: "The algorithm converges asymptotically to the optimal solution."
            - Casual: "Dude, that concert was totally awesome! Best night ever!"
            - Formal: "Please be advised that the meeting has been rescheduled to next Tuesday."
            - Creative: "The autumn leaves whispered secrets to the wind as they fell."
            """)
    
    gr.Markdown("""
    ---
    
    ## Key takeaways
    
    1. **Prompt templates** make prompts reusable and maintainable
    2. **Output parsers** extract structured data reliably
    3. **Chains** compose multiple steps with the `|` operator
    4. **Pydantic schemas** ensure type-safe structured outputs
    5. **Few-shot examples** help models learn patterns
    
    **Next step:** Try Activity 4 to build your own LangChain chains!
    """)


# Launch the Gradio app
if __name__ == '__main__':
    demo.launch()
