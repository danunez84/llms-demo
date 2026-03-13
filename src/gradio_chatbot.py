'''
Gradio chatbot with selectable backend (Ollama or llama.cpp).

This demo provides a web UI where users can:
- Choose between Ollama and llama.cpp backends
- Customize the system prompt
- Have multi-turn conversations with context

Usage:
    python src/gradio_chatbot.py
'''

import os
import gradio as gr
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---

# Temperature controls randomness (0.0 = deterministic, 1.0+ = creative)
temperature = 0.7

# Default system prompt - sets the assistant's behavior and personality
default_system_prompt = (
    'You are a helpful teaching assistant at an AI/ML boot camp. '
    'Answer questions in simple language with examples when possible. '
    'Answer in the style of a pirate and use nautical themed analogies.'
)

# --- Initialize Ollama backend ---

# Model to use from Ollama (must be pulled first: ollama pull qwen2.5:3b)
ollama_model = 'qwen2.5:3b'

# Create Ollama client using LangChain wrapper
ollama_client = ChatOllama(
    model=ollama_model,
    temperature=temperature
)

# --- Initialize llama.cpp backend (OpenAI-compatible API) ---

# Get server URL from environment, default to localhost
llamacpp_server = os.environ.get('PERDRIZET_URL', 'localhost:8502')

# Configure API key and base URL based on server location
# Localhost uses 'dummy' key, remote servers use PERDRIZET_API_KEY
if llamacpp_server.startswith('localhost') or llamacpp_server.startswith('127.'):
    llamacpp_api_key = os.environ.get('LLAMA_API_KEY', 'dummy')
    llamacpp_base_url = f'http://{llamacpp_server}/v1'
else:
    llamacpp_api_key = os.environ.get('PERDRIZET_API_KEY')
    llamacpp_base_url = f'https://{llamacpp_server}/v1'

# Create OpenAI client pointed at llama.cpp server
llamacpp_client = OpenAI(
    base_url=llamacpp_base_url,
    api_key=llamacpp_api_key,
)

# Try to get the model name from the server (gracefully fail if unavailable)
try:
    models = llamacpp_client.models.list()
    llamacpp_model = models.data[0].id
except:
    llamacpp_model = 'llama.cpp (server not available)'


def respond(message, history, backend, system_prompt):
    '''Sends message to selected model backend, gets response back.
    
    Args:
        message: User's current message
        history: List of [user_msg, assistant_msg] pairs from Gradio
        backend: Either 'Ollama' or 'llama.cpp'
        system_prompt: System prompt to set model behavior
    
    Returns:
        Response string from the model (or error message if backend unavailable)
    '''
    
    # --- Ollama Backend ---
    if backend == 'Ollama':
        try:
            # Build message list in LangChain format (SystemMessage, HumanMessage, AIMessage)
            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history to maintain context
            # Gradio passes history as list of [user, assistant] pairs
            for item in history:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    user_msg, assistant_msg = item[0], item[1]
                    messages.append(HumanMessage(content=user_msg))
                    messages.append(AIMessage(content=assistant_msg))
            
            # Add current user message
            messages.append(HumanMessage(content=message))
            
            # Invoke Ollama model and return response
            response = ollama_client.invoke(messages)
            return response.content
        
        except Exception as e:
            # Return helpful error message if Ollama server is unreachable
            error_msg = (
                f'**Ollama backend is unavailable**\n\n'
                f'Make sure the Ollama server is running:\n'
                f'```bash\n'
                f'ollama serve\n'
                f'```\n\n'
                f'Error details: {str(e)}'
            )
            return error_msg
    
    # --- llama.cpp Backend ---
    else:
        try:
            # Build message list in OpenAI format (dict with 'role' and 'content')
            messages = [{'role': 'system', 'content': system_prompt}]
            
            # Add conversation history to maintain context
            for item in history:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    user_msg, assistant_msg = item[0], item[1]
                    messages.append({'role': 'user', 'content': user_msg})
                    messages.append({'role': 'assistant', 'content': assistant_msg})
            
            # Add current user message
            messages.append({'role': 'user', 'content': message})
            
            # Call llama.cpp server using OpenAI-compatible API
            response = llamacpp_client.chat.completions.create(
                model=llamacpp_model,
                messages=messages,
                temperature=temperature,
            )
            
            # Extract and return response text
            return response.choices[0].message.content
        
        except Exception as e:
            # Return helpful error message if llama.cpp server is unreachable
            error_msg = (
                f'**llama.cpp backend is unavailable**\n\n'
                f'Make sure the llama-server is running at: `{llamacpp_base_url}`\n\n'
                f'To start the server:\n'
                f'```bash\n'
                f'llama.cpp/build/bin/llama-server -m <model.gguf> --host 0.0.0.0 --port 8502\n'
                f'```\n\n'
                f'Or configure remote server in `.env` file.\n\n'
                f'Error details: {str(e)}'
            )
            return error_msg



# --- Build Gradio UI ---

# Use Gradio Blocks for custom layout with multiple input controls
with gr.Blocks(title='Multi-backend chatbot') as demo:
    
    # Page title and description
    gr.Markdown('# Multi-backend chatbot')
    
    # Backend selector - radio buttons for Ollama vs llama.cpp
    with gr.Row():
        backend_selector = gr.Radio(
            choices=['Ollama', 'llama.cpp'],
            value='Ollama',
            label='Model Backend',
            info=f'Ollama: {ollama_model} | llama.cpp: {llamacpp_model}'
        )
    
    # System prompt input - allows customizing model behavior
    system_prompt_input = gr.Textbox(
        label='System Prompt',
        value=default_system_prompt,
        lines=3,
        placeholder='Enter system prompt to set the assistant\'s behavior...'
    )
    
    # Chat interface with backend and system prompt as additional inputs
    chatbot = gr.ChatInterface(
        fn=respond,
        additional_inputs=[backend_selector, system_prompt_input],
    )


# Launch the Gradio app
if __name__ == '__main__':
    demo.launch()