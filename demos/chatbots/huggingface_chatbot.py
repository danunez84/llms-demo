'''
Simple chatbot using HuggingFace Transformers (direct model loading).

This loads the model directly into memory - no inference server needed.
Compare with chatbot.py which uses Ollama as a separate server process.

Set up:
    pip install transformers torch

Usage:
    python huggingface_chatbot.py
'''

import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Suppress noisy key warnings from checkpoint loading
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# --- Configuration ---
# Model to use from HuggingFace Hub
model_name = 'Qwen/Qwen2.5-3B-Instruct'

# Temperature controls randomness (0.0 = deterministic, 1.0 = creative)
temperature = 0.7

# Maximum number of tokens to generate in each response
max_new_tokens = 512

# System prompt sets the assistant's behavior and personality
system_prompt = (
    'You are a helpful teaching assistant for a data science course. '
    'Keep answers concise and use examples when possible. '
    'If you are not sure about something, say so.'
)

# Load tokenizer (converts text to numbers the model understands)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Select the best available device (GPU > CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the actual model (downloads first time, then cached locally)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)

# Print the model architecture and number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f'\nLoaded model: {model_name} with {num_params/1e9:.2f} billion parameters on {device}')
print(f'\nArchitecture:\n')
print(model)


def generate(messages):
    '''Generate a response from the model given a list of messages.'''

    # Convert conversation history to the format expected by the model
    # This uses the model's chat template (e.g., adds <|im_start|> tags for Qwen)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # Return text, not token IDs yet
        add_generation_prompt=True,  # Add the prompt for the assistant's response
    )

    # Tokenize the text (convert to numbers) and prepare for the model
    # 'pt' means PyTorch tensors, move to model's device (CPU or GPU)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    # Generate new tokens using the model
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # Limit response length
        temperature=temperature,        # Control randomness
        do_sample=True,                 # Use sampling (vs greedy decoding)
    )

    # Decode only the newly generated tokens (skip the input prompt)
    # This extracts just the assistant's response
    new_tokens = output[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    '''Main conversation loop.'''

    # Initialize conversation history with system prompt
    # Format matches OpenAI/HuggingFace chat message structure
    history = [{'role': 'system', 'content': system_prompt}]

    print(f'\nChatbot ready. Model: {model_name}, device: {device}, temperature: {temperature}')

    # Main conversation loop - runs until interrupted (Ctrl+C)
    while True:

        # Get user input
        user_input = input('User: ')

        # Check for exit condition
        if user_input.lower() in ['exit', 'quit']:
            print('Exiting chatbot.')
            break

        user_message = {'role': 'user', 'content': user_input}
        history.append(user_message)

        # Generate response using entire conversation history
        response = generate(history)
        
        # Add assistant's response to history for context in next turn
        model_message = {'role': 'assistant', 'content': response}
        history.append(model_message)

        print(f'\n{model_name}: {response}\n')


if __name__ == '__main__':
    main()
