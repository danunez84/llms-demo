'''Chat bot demo using a llama.cpp server with the OpenAI-compatible API.

--- Option 1: Connect to the public class server via the API ---

1. Create a .env file in the repo root with the server address and your API key:

   PERDRIZET_URL=<server-address-provided-in-class>
   PERDRIZET_API_KEY=<api-key-provided-in-class>

2. Run the chatbot - it will automatically connect to the remote server:

   $ python demos/chatbots/llamacpp_chatbot.py

--- Option 2: Build and run the server locally ---

See README.md for instructions on how to build llama.cpp and start the server.

Once the server is running, run the chatbot (no .env needed, defaults to localhost:8502 with API key "dummy"):

   $ python demos/chatbots/llamacpp_chatbot.py

---

Environment variables (read from .env file if present):
- PERDRIZET_URL: Remote server address (default: localhost:8502)
- PERDRIZET_API_KEY: API key for remote servers
- LLAMA_API_KEY: API key for localhost (default: "dummy")
'''

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configuration
server = os.environ.get('PERDRIZET_URL', 'localhost:8502')

# For localhost, default to 'dummy' API key unless explicitly set
# For remote servers, use the API key from the environment
if server.startswith('localhost') or server.startswith('127.'):
    api_key = os.environ.get('LLAMA_API_KEY', 'dummy')
    base_url = f'http://{server}/v1'

else:
    api_key = os.environ.get('PERDRIZET_API_KEY', 'dummy')
    base_url = f'https://{server}/v1'

temperature = 0.7

system_prompt = (
    'Reasoning: low\n\n'
    'You are a helpful teaching assistant at an AI/ML boot camp. '
    'Answer questions in simple language with examples when possible.'
)

# Initialize the OpenAI client pointing at the llama.cpp server
client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

# Get the model name from the server
models = client.models.list()
model = models.data[0].id

# Start conversation history with system prompt
history = [{'role': 'system', 'content': system_prompt}]


def main():
    '''Main conversation loop.'''

    print(f'Connected to GPT server at {base_url}')
    print(f'Model: {model}')
    print('Type "exit" to quit.\n')

    # Loop until user types 'exit'
    while True:

        # Get text input from the user
        user_input = input('User: ')

        # Check for exit condition
        if user_input.lower() in ['exit', 'quit']:
            print('Exiting chatbot.')
            break

        # Add the user's message to the conversation history
        history.append({'role': 'user', 'content': user_input})

        # Stream the response so tokens appear as they are generated
        stream = client.chat.completions.create(
            model=model,
            messages=history,
            temperature=temperature,
            stream=True,
        )

        print(f'\n{model}: ', end='', flush=True)

        assistant_message = ''

        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                print(token, end='', flush=True)
                assistant_message += token

        print('\n')

        # Add the model's response to the conversation history
        history.append({'role': 'assistant', 'content': assistant_message})


# Main entry point
if __name__ == '__main__':
    main()
