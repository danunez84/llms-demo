'''
Gradio chatbot using llama.cpp server backend.

This demo provides a web UI where users can:
- Customize the system prompt
- Have multi-turn conversations with context
- Connect to local or remote llama.cpp servers

Usage:
    python src/gradio_chatbot.py
'''

import os
import gradio as gr
from dotenv import load_dotenv
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


def respond(message, history, system_prompt):
    '''Sends message to llama.cpp server, gets response back.
    
    Args:
        message: User's current message
        history: List of [user_msg, assistant_msg] pairs from Gradio
        system_prompt: System prompt to set model behavior
    
    Returns:
        Response string from the model (or error message if server unavailable)
    '''
    
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
with gr.Blocks(title='llama.cpp chatbot') as demo:
    
    # Page title and description
    gr.Markdown(f'# llama.cpp chatbot\n\n**Model:** {llamacpp_model} | **Server:** {llamacpp_base_url}')
    
    # System prompt input - allows customizing model behavior
    system_prompt_input = gr.Textbox(
        label='System Prompt',
        value=default_system_prompt,
        lines=3,
        placeholder='Enter system prompt to set the assistant\'s behavior...'
    )
    
    # Chat interface with system prompt as additional input
    chatbot = gr.ChatInterface(
        fn=respond,
        additional_inputs=[system_prompt_input],
    )


# Launch the Gradio app
if __name__ == '__main__':
    demo.launch()