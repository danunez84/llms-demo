'''Chat bot demo using Ollama and LangChain.

Start the ollama server, and download the model:

$ ollama serve
$ ollama pull qwen2.5:3b
'''

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Configuration for the Ollama chatbot
model = 'qwen2.5:3b'
temperature = 0.9

system_prompt = (
    'You are a helpful teaching assistant at an AI/ML boot camp.',
    'Answer questions in simple language with examples when possible.',
    'Answer in the style of a pirate and use nautical themed analogies.'
)

# Initialize the Ollama chatbot
llm = ChatOllama(
    model=model,
    temperature=temperature
)

# Start conversation history with system prompt
history = [SystemMessage(content=system_prompt)]


def main():
    '''Main conversation loop.'''

    # Loop until user types 'exit'
    while True:

        # Get text input from the user
        user_input = input('User: ')

        # Check for exit condition
        if user_input.lower() in ['exit', 'quit']:
            print('Exiting chatbot.')
            break

        # Add the user's message to the conversation history
        history.append(HumanMessage(content=user_input))

        # Send the conversation history to the Ollama model and get a response
        response = llm.invoke(history)

        # Add the model's response to the conversation history
        history.append(response)

        # Print the model's response
        print(f'\n{model}: {response.content}\n')


# Main entry point
if __name__ == '__main__':

    main()