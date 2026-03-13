'''
Gradio chatbot with selectable backend (Ollama or llama.cpp).
'''

import os
import gradio as gr
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from openai import OpenAI

load_dotenv()

# Configuration
temperature = 0.7

default_system_prompt = (
    'You are a helpful teaching assistant at an AI/ML boot camp. '
    'Answer questions in simple language with examples when possible. '
    'Answer in the style of a pirate and use nautical themed analogies.'
)

# Initialize Ollama backend
ollama_model = 'qwen2.5:3b'

ollama_client = ChatOllama(
    model=ollama_model,
    temperature=temperature
)

# Initialize llama.cpp backend (OpenAI-compatible)
llamacpp_server = os.environ.get('PERDRIZET_URL', 'localhost:8502')

# For localhost, default to 'dummy' API key unless explicitly set
# For remote servers, use the API key from the environment
if llamacpp_server.startswith('localhost') or llamacpp_server.startswith('127.'):
    llamacpp_api_key = os.environ.get('LLAMA_API_KEY', 'dummy')
    llamacpp_base_url = f'http://{llamacpp_server}/v1'

else:
    llamacpp_api_key = os.environ.get('PERDRIZET_API_KEY')
    llamacpp_base_url = f'https://{llamacpp_server}/v1'

llamacpp_client = OpenAI(
    base_url=llamacpp_base_url,
    api_key=llamacpp_api_key,
)

# Try to get llama.cpp model name (may fail if server not running)
try:
    models = llamacpp_client.models.list()
    llamacpp_model = models.data[0].id

except:
    llamacpp_model = 'llama.cpp (server not available)'


def respond(message, history, backend, system_prompt):
    '''Sends message to selected model backend, gets response back.
    
    Args:
        message: User's current message
        history: List of [user_msg, assistant_msg] pairs
        backend: Either 'Ollama' or 'llama.cpp'
        system_prompt: System prompt to set model behavior
    '''
    
    if backend == 'Ollama':

        # Use LangChain's ChatOllama
        messages = [SystemMessage(content=system_prompt)]
        
        for user_msg, assistant_msg in history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=assistant_msg))
        
        messages.append(HumanMessage(content=message))
        response = ollama_client.invoke(messages)
        return response.content
    
    else:  # llama.cpp
    
        # Use OpenAI client with llama.cpp server
        messages = [{'role': 'system', 'content': system_prompt}]
        
        for user_msg, assistant_msg in history:
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': assistant_msg})
        
        messages.append({'role': 'user', 'content': message})
        
        response = llamacpp_client.chat.completions.create(
            model=llamacpp_model,
            messages=messages,
            temperature=temperature,
        )

        return response.choices[0].message.content


# Build custom UI with Gradio Blocks for backend selection
with gr.Blocks(title='Multi-Backend Chatbot') as demo:
    gr.Markdown('# Multi-Backend Chatbot')
    gr.Markdown('Choose your model backend and start chatting!')
    
    with gr.Row():
        backend_selector = gr.Radio(
            choices=['Ollama', 'llama.cpp'],
            value='Ollama',
            label='Model Backend',
            info=f'Ollama: {ollama_model} | llama.cpp: {llamacpp_model}'
        )
    
    system_prompt_input = gr.Textbox(
        label='System Prompt',
        value=default_system_prompt,
        lines=3,
        placeholder='Enter system prompt to set the assistant\'s behavior...'
    )
    
    chatbot = gr.ChatInterface(
        fn=respond,
        additional_inputs=[backend_selector, system_prompt_input],
    )


if __name__ == '__main__':
    demo.launch()