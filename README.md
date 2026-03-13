# LLM chatbots demo

## Quickstart

### 1. Fork and clone

1. Click **Fork** in the top-right corner of this repo on GitHub to create your own copy.
2. Clone your fork:

   ```bash
   git clone https://github.com/<your-username>/llms-demo.git
   ```

### 2. Open in a dev container

1. Open the cloned folder in VS Code.
2. When prompted **"Reopen in Container"**, click it - or run the command **Dev Containers: Reopen in Container** from the Command Palette (`Ctrl+Shift+P`).
3. VS Code will build and start the container. This takes a few minutes the first time.

### 3. What happens during container startup

The dev container is based on the `gperdrizet/deeplearning-gpu` image (NVIDIA GPU-enabled). On first creation, the `postCreateCommand` runs automatically and does the following:

| Step | What it does |
|------|-------------|
| `mkdir -p models/hugging_face && mkdir -p models/ollama` | Creates local directories for model storage |
| `pip install -r requirements.txt` | Installs Python dependencies: **gradio**, **huggingface-hub**, **langchain-ollama**, **openai**, **python-dotenv**, **torch**, **transformers** |
| `bash .devcontainer/install_ollama.sh` | Downloads and installs the Ollama CLI |

The container also pre-configures the following:

| Setting | Detail |
|---------|--------|
| **GPU access** | All host GPUs are passed through (`--gpus all`) |
| **Python interpreter** | `/usr/bin/python` is set as the default |
| **`HF_HOME`** | Points to `models/hugging_face` so Hugging Face downloads stay in the repo |
| **`OLLAMA_MODELS`** | Points to `models/ollama` so Ollama downloads stay in the repo |
| **Port 7860** | Forwarded automatically for Gradio web UIs |
| **VS Code extensions** | Python, Jupyter, Code Spell Checker, and Marp (slide viewer) are installed |

Once the container is ready you can start running the demos - no extra setup needed.

---

## Demos

| File | What it does | Stack |
|------|-------------|-------|
| `src/ollama_chatbot.py` | Terminal chatbot | Ollama + LangChain |
| `src/llamacpp_chatbot.py` | Terminal chatbot using a large MoE model | llama.cpp + OpenAI client |
| `src/gradio_chatbot.py` | Web-based chatbot with Gradio UI | Ollama + LangChain + Gradio |
| `src/huggingface_chatbot.py` | Terminal chatbot with direct model loading | HuggingFace Transformers |

### Running the demos

**Ollama chatbot**:
```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the chatbot
python src/ollama_chatbot.py
```

**llama.cpp chatbot**:

You have two choices for this one: use the hosted model at `gpt.perdrizet.org`, or run llama.cpp locally.

**Option 1: Using the remote server (recommended for quick start)**

```bash
# 1. Create a .env file with your API credentials
cp .env.example .env

# 2. Edit .env and set:
#    PERDRIZET_URL=gpt.perdrizet.org
#    PERDRIZET_API_KEY=your-api-key-here

# 3. Run the chatbot
python src/llamacpp_chatbot.py
```

**Option 2: Running llama.cpp locally**

```bash
# 1. Download a GGUF model (e.g., GPT-OSS-120B)
python utils/download_gpt_oss_120b.py

# 2. Build llama.cpp with CUDA support
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
cd ..

# 3. Start the llama-server (see model-specific commands below)
llama.cpp/build/bin/llama-server -m <model.gguf> <flags...>

# 4. Run the chatbot (in another terminal)
python src/llamacpp_chatbot.py
```

> **Note**: For localhost, the defaults work automatically (localhost:8502 with "dummy" API key). For remote servers, configure `PERDRIZET_URL` and `PERDRIZET_API_KEY` in your `.env` file.

**Gradio chatbot**:
```bash
# 1. Start the Ollama server in a terminal
ollama serve

# 2. Pull a model (in another terminal)
ollama pull qwen2.5:3b

# 3. Run the Gradio chatbot
python src/gradio_chatbot.py

# 4. Open the URL shown in the terminal (usually http://127.0.0.1:7860)
```

**HuggingFace chatbot**:
```bash
# 1. Run the chatbot (downloads model on first run)
python src/huggingface_chatbot.py

# Note: This loads the model directly into memory - no inference server needed.
# First run will download ~6GB model files to models/hugging_face/
```

---

## Demo cheat sheet

Quick reference for the tools and libraries used in the demos:

1. Ollama
2. HuggingFace Transformers
3. LangChain
4. Gradio
5. llama.cpp

### 1. [Ollama](https://ollama.com/)

Local inference server - runs models on your machine behind a REST API.

- [Model library](https://ollama.com/library)
- [Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md)
- [API reference](https://github.com/ollama/ollama/blob/main/docs/api.md)

```bash
# Install
curl -fsSL https://ollama.com/install.sh | sh

# Start the server (runs on localhost:11434)
ollama serve

# Pull a model
ollama pull qwen2.5:3b

# List downloaded models
ollama list

# Run a model interactively
ollama run qwen2.5:3b

# Remove a model
ollama rm qwen2.5:3b
```

>**Note**: if you are running the demos in this repo via a devcontainer as intended, you do not need to install Ollama. The container environment includes it.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `OLLAMA_MODELS` | Directory where models are stored |
| `OLLAMA_HOST` | Server address (default `127.0.0.1:11434`) |

---

### 2. [HuggingFace Transformers](https://huggingface.co/docs/transformers)

Load and run models directly in Python - no server needed.

- [Model Hub](https://huggingface.co/models)
- [Documentation](https://huggingface.co/docs/transformers)
- [GitHub](https://github.com/huggingface/transformers)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer (downloads on first run)
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')

# Build a prompt using the chat template
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'Hello!'},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Tokenize and generate
inputs = tokenizer(text, return_tensors='pt').to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)

# Decode only the new tokens
new_tokens = output[0][inputs['input_ids'].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

### Key classes

| Class | Purpose |
|-------|---------|
| `AutoTokenizer` | Loads the correct tokenizer for any model |
| `AutoModelForCausalLM` | Loads a causal language model for text generation |
| `apply_chat_template()` | Formats a list of messages into the model's expected prompt format |
| `model.generate()` | Runs inference and returns generated token IDs |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_HOME` | Directory where models/cache are stored |

---

### 3. [LangChain](https://python.langchain.com/)

Framework for building LLM applications. Used here for chat message types and the Ollama integration.

- [Documentation](https://python.langchain.com/docs)
- [langchain-ollama](https://python.langchain.com/docs/integrations/chat/ollama)
- [Other avalible chat integrations](https://docs.langchain.com/oss/python/integrations/chat)

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Create the LLM client (talks to a running Ollama server)
llm = ChatOllama(
    model='qwen2.5:3b',
    temperature=0.7,
)

# Build a conversation as a list of message objects
messages = [
    SystemMessage(content='You are a helpful assistant.'),
    HumanMessage(content='Hello!'),
]

# Get a response
response = llm.invoke(messages)
print(response.content)

# Append the response to history and continue
messages.append(response)  # response is an AIMessage
messages.append(HumanMessage(content='Follow-up question'))
response = llm.invoke(messages)
```

### Key classes

| Class | Purpose |
|-------|---------|
| `ChatOllama` | LLM client that connects to Ollama |
| `SystemMessage` | Sets the model's behavior/persona |
| `HumanMessage` | A user message |
| `AIMessage` | A model response |

---

### 4. [Gradio](https://www.gradio.app)

Build a web UI for a chatbot with a few lines of code.

- [Documentation](https://www.gradio.app/docs)
- [GitHub](https://github.com/gradio-app/gradio)
- [ChatInterface guide](https://www.gradio.app/guides/creating-a-chatbot-fast)

```python
import gradio as gr

def respond(message, history):
    # message: str - the latest user input
    # history: list of [user, assistant] pairs
    return 'Hello!'  # return the assistant's reply as a string

demo = gr.ChatInterface(
    fn=respond,
    title='My Chatbot',
)

demo.launch()
```

### `gr.ChatInterface`

| Parameter | Purpose |
|-----------|---------|
| `fn` | Response function `(message, history) -> str` |
| `title` | Title shown in the web page |

### `demo.launch()`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `server_name` | `127.0.0.1` | Bind address (`0.0.0.0` for external access) |
| `server_port` | `7860` | Port number |
| `share` | `False` | Create a public Gradio link |

---

### 5. [llama.cpp](https://github.com/ggml-org/llama.cpp)

High-performance C/C++ inference engine. Runs GGUF-quantized models and can split MoE layers across CPU and GPU, making it possible to serve large models (100B+) on consumer hardware.

- [GitHub](https://github.com/ggml-org/llama.cpp)
- [GGUF model format](https://huggingface.co/docs/hub/gguf)
- [Server documentation](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md)

```bash
# Build from source with CUDA support (compiles for your GPU automatically)
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j$(nproc)
```

> **Note**: The build compiles CUDA kernels for the GPU(s) detected on your machine.
> This takes several minutes but only needs to be done once. If you change GPUs, rebuild.

```bash
# Start the server with CPU/GPU MoE split
# Replace <model.gguf> with the path to your GGUF file
llama.cpp/build/bin/llama-server \
    -m <model.gguf> \
    --n-gpu-layers 999 \
    --n-cpu-moe <N> \
    -c 0 --flash-attn on \
    --jinja \
    --host 0.0.0.0 --port 8502 --api-key "dummy"
```

See the model sections below for complete, copy-paste run commands.

The server exposes an **OpenAI-compatible API**, so any OpenAI client library can connect to it.

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:8502/v1',
    api_key='your-api-key',
)

response = client.chat.completions.create(
    model='model-name',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello!'},
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

### Key server flags

| Flag | Purpose |
|------|----------|
| `-m` | Path to the GGUF model file |
| `--n-gpu-layers` | Number of layers to offload to GPU (`999` = all non-MoE layers) |
| `--n-cpu-moe` | Number of MoE blocks to keep on CPU (e.g. `36` = all MoE on CPU) |
| `-c` | Context length (`0` = model maximum) |
| `--flash-attn` | Enable flash attention |
| `--host` / `--port` | Server bind address and port |
| `--jinja` | Enable Jinja chat templates (required for harmony and similar formats) |
| `--api-key` | API key for authenticating requests |

### CPU/GPU MoE split explained

Mixture-of-Experts models have two types of layers: **attention layers** (small, benefit from GPU) and **MoE/expert layers** (large, run well on CPU). The `--n-cpu-moe` flag controls how many MoE blocks stay on CPU:

| Config | VRAM usage | Generation speed |
|--------|-----------|------------------|
| `--n-cpu-moe 36` (all MoE on CPU) | ~5–8 GB | ~18–25 tok/s |
| `--n-cpu-moe 28` (8 MoE on GPU) | ~22 GB | ~25–31 tok/s |

This makes it possible to run a 120B parameter model with as little as 8 GB of VRAM and 64 GB of system RAM.

---

### Model: openai/gpt-oss-120b

A 120B parameter Mixture-of-Experts (MoE) model released by OpenAI. We use the mxfp4-quantized [GGUF version](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) published by [ggml-org](https://github.com/ggml-org) - the organization behind [llama.cpp](https://github.com/ggml-org/llama.cpp), the [GGML](https://github.com/ggml-org/ggml) tensor library, and the [GGUF](https://huggingface.co/docs/hub/gguf) model format.

- [Model card (OpenAI)](https://huggingface.co/openai/gpt-oss-120b)
- [GGUF quantization (ggml-org)](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF)

| Detail | Value |
|--------|-------|
| **Parameters** | 120B (Mixture-of-Experts) |
| **Quantization** | mxfp4 (expert layers), BF16 (attention layers) |
| **Format** | GGUF (3 shards) |
| **Download size** | ~60 GB |
| **Min system RAM** | 64 GB (96 GB recommended) |
| **Min VRAM** | ~5 GB (attention layers only, with `--n-cpu-moe 36`) |

```bash
# Download the GGUF model
python utils/download_gpt_oss_120b.py
```

#### Response format: Harmony

GPT-OSS was trained on OpenAI's [harmony response format](https://github.com/openai/harmony). The model uses internal "channels" (e.g. `analysis` for chain-of-thought, `final` for the actual response). llama.cpp auto-detects and parses harmony, separating thinking into `reasoning_content` and the clean response into `content`.

You can control reasoning effort by adding one of these lines at the **top** of the system prompt:

- `Reasoning: low` — fast responses, minimal thinking
- `Reasoning: medium` — balanced speed and detail
- `Reasoning: high` — deep, detailed analysis

#### Run

```bash
llama.cpp/build/bin/llama-server \
    -m models/hugging_face/hub/models--ggml-org--gpt-oss-120b-GGUF/snapshots/*/gpt-oss-120b-mxfp4-00001-of-00003.gguf \
    --n-gpu-layers 999 \
    --n-cpu-moe 36 \
    -c 0 --flash-attn on \
    --jinja \
    --host 0.0.0.0 --port 8502 --api-key "dummy"
```

The model has 36 MoE blocks. `--n-cpu-moe 36` keeps all expert layers on CPU (lowest VRAM, ~5 GB). Reduce the value to move MoE blocks to GPU if you have VRAM to spare.

---

### Model: openai/gpt-oss-20b

The smaller sibling of GPT-OSS-120B, designed for lower latency and local use cases. At ~11 GB it fits entirely in GPU memory on many consumer GPUs no CPU MoE split needed — delivering ~50 tok/s generation. Uses the same [harmony response format](https://github.com/openai/harmony) as the 120B model.

- [Model card (OpenAI)](https://huggingface.co/openai/gpt-oss-20b)
- [GGUF quantization (ggml-org)](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)

| Detail | Value |
|--------|-------|
| **Parameters** | 21B total, 3.6B active (Mixture-of-Experts) |
| **Quantization** | mxfp4 (expert layers), BF16 (attention layers) |
| **Format** | GGUF (single file) |
| **Download size** | ~11 GB |
| **Min VRAM** | ~14 GB (fits entirely on GPU) |

```bash
# Download the GGUF model
python utils/download_gpt_oss_20b.py
```

#### Response format: Harmony

Same as GPT-OSS-120B — see [above](#response-format-harmony).

#### Run

```bash
llama.cpp/build/bin/llama-server \
    -m models/hugging_face/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/*/gpt-oss-20b-mxfp4.gguf \
    --n-gpu-layers 999 \
    -c 8192 --flash-attn on \
    --jinja \
    --host 0.0.0.0 --port 8502 --api-key "dummy"
```

No `--n-cpu-moe` needed — the model fits entirely in GPU memory.

---

### Model: Qwen3.5-35B-A3B

A 35B parameter Mixture-of-Experts vision-language model from Alibaba's Qwen team, with only 3B active parameters per token. Smaller and faster than GPT-OSS-120B, making it a good choice when serving multiple concurrent users. We use the mxfp4-quantized [GGUF version](https://huggingface.co/noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF) by noctrex.

- [Model card (Qwen)](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [GGUF quantization (noctrex)](https://huggingface.co/noctrex/Qwen3.5-35B-A3B-MXFP4_MOE-GGUF)

| Detail | Value |
|--------|-------|
| **Parameters** | 35B total, 3B active (Mixture-of-Experts) |
| **Quantization** | mxfp4 (expert layers), BF16 (attention layers) |
| **Format** | GGUF (single file) |
| **Download size** | ~22 GB |
| **Vision support** | Yes (with mmproj-BF16.gguf projection file) |
| **Min system RAM** | 32 GB |
| **Min VRAM** | ~3 GB (attention layers only, with `--n-cpu-moe`) |

```bash
# Download the GGUF model
python utils/download_qwen35_35b.py
```

#### Response format

Qwen3.5 uses `<think>...</think>` tags for chain-of-thought reasoning (the same convention as DeepSeek). llama.cpp auto-detects this and separates thinking into `reasoning_content`. To disable thinking and get direct responses, add `/no_think` to the end of your user message.

#### Run

```bash
llama.cpp/build/bin/llama-server \
    -m models/hugging_face/hub/models--noctrex--Qwen3.5-35B-A3B-MXFP4_MOE-GGUF/snapshots/*/Qwen3.5-35B-A3B-MXFP4_MOE_BF16.gguf \
    --n-gpu-layers 999 \
    --n-cpu-moe 40 \
    -c 0 --flash-attn on \
    --jinja \
    --host 0.0.0.0 --port 8502 --api-key "dummy"
```

The model has 40 MoE blocks. `--n-cpu-moe 40` keeps all expert layers on CPU. This model is much smaller (~22 GB) and faster than GPT-OSS-120B, making it a good choice for consumer hardware.

