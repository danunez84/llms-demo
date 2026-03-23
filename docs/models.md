# Models

## openai/gpt-oss-120b

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

### Response format: Harmony

GPT-OSS was trained on OpenAI's [harmony response format](https://github.com/openai/harmony). The model uses internal "channels" (e.g. `analysis` for chain-of-thought, `final` for the actual response). llama.cpp auto-detects and parses harmony, separating thinking into `reasoning_content` and the clean response into `content`.

You can control reasoning effort by adding one of these lines at the **top** of the system prompt:

- `Reasoning: low` - fast responses, minimal thinking
- `Reasoning: medium` - balanced speed and detail
- `Reasoning: high` - deep, detailed analysis

### Run

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

## openai/gpt-oss-20b

The smaller sibling of GPT-OSS-120B, designed for lower latency and local use cases. At ~11 GB it fits entirely in GPU memory on many consumer GPUs (no CPU MoE split needed), delivering ~50 tok/s generation. Uses the same [harmony response format](https://github.com/openai/harmony) as the 120B model.

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

### Response format: Harmony

Same as GPT-OSS-120B (see above).

### Run

```bash
llama.cpp/build/bin/llama-server \
    -m models/hugging_face/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/*/gpt-oss-20b-mxfp4.gguf \
    --n-gpu-layers 999 \
    -c 8192 --flash-attn on \
    --jinja \
    --host 0.0.0.0 --port 8502 --api-key "dummy"
```

`-c 8192` sets the context length to 8,192 tokens (~6,000 words). Increase this to `-c 32768` for longer conversations (~24,000 words — enough for an entire technical manual or codebase in a single prompt), at the cost of more VRAM. Use `-c 0` to let llama.cpp use the model's full supported context length automatically.

No `--n-cpu-moe` needed - the model fits entirely in GPU memory.

---

## Qwen3.5-35B-A3B

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

### Response format

Qwen3.5 uses `<think>...</think>` tags for chain-of-thought reasoning (the same convention as DeepSeek). llama.cpp auto-detects this and separates thinking into `reasoning_content`. To disable thinking and get direct responses, add `/no_think` to the end of your user message.

### Run

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
