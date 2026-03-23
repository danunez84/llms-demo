# Systemd deployment

This guide covers deploying llama.cpp server as a systemd service on the host OS for production use.

## Prerequisites

- Linux host with systemd
- NVIDIA GPU with CUDA drivers installed
- Root/sudo access

## Quick start

### 1. Build llama.cpp on host

```bash
cd /opt
sudo git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
sudo cmake -B build -DGGML_CUDA=ON
sudo cmake --build build --config Release -j$(nproc)
```

> **Note:** Building takes several minutes and compiles CUDA kernels for your GPU.

### 2. Set up directories and models

```bash
# Create model directory
sudo mkdir -p /opt/models
```

Download models using one of these methods:

**Option A: Download on host with HF_HOME set**

```bash
# Set HF_HOME to download directly to /opt/models
sudo HF_HOME=/opt/models python3 /path/to/llms-demo/utils/download_gpt_oss_20b.py
```

**Option B: Copy from dev container**

If you already downloaded models in the dev container:

```bash
# From your host OS (outside container)
sudo cp -r /path/to/llms-demo/models/hugging_face /opt/models/
```

**Option C: Manual download**

Download GGUF files directly from HuggingFace:

```bash
# Example for GPT-OSS-20B
cd /opt/models
sudo wget https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf
```

> **Note:** The download scripts respect the `HF_HOME` environment variable. Without setting it, models download to `~/.cache/huggingface/` by default.

### 3. Create service user

```bash
sudo useradd -r -s /bin/false -d /opt/llama.cpp llama
sudo chown -R llama:llama /opt/llama.cpp
sudo chown -R llama:llama /opt/models
```

### 4. Generate API key

```bash
API_KEY=$(openssl rand -base64 32)
echo "Your API key: $API_KEY"
# Save this key securely!
```

### 5. Install and configure service

```bash
# Copy unit file to systemd
sudo cp utils/llamacpp.service /etc/systemd/system/

# Edit the service file with your API key and model path
sudo nano /etc/systemd/system/llamacpp.service
# Update these lines:
#   - Replace YOUR_API_KEY_HERE with your generated key
#   - Replace the * in model path with actual snapshot hash
#   - Modify --n-cpu-moe if using MoE model
```

> **Important:** Systemd doesn't expand shell wildcards (`*`) in ExecStart. You must replace `snapshots/*/` with the actual hash directory.

### 6. Enable and start

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable llamacpp.service

# Start the service
sudo systemctl start llamacpp.service

# Check status
sudo systemctl status llamacpp.service
```

## Monitoring

### View logs

No log file is needed - systemd automatically captures all stdout/stderr output and forwards it to **journald** (the system journal). This is preferable to a log file: journald handles rotation automatically, logs survive if the service crashes before flushing, and you get structured querying by time, boot, and priority.

```bash
# Follow logs in real-time
sudo journalctl -u llamacpp.service -f

# View last 100 lines
sudo journalctl -u llamacpp.service -n 100

# View logs since boot
sudo journalctl -u llamacpp.service -b
```

### Check metrics

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
    http://localhost:8502/metrics
```

### Service management

```bash
# Stop service
sudo systemctl stop llamacpp.service

# Restart service
sudo systemctl restart llamacpp.service

# Enable service (will start on boot)
sudo systemctl enable llamacpp.service

# Disable service (don't start on boot)
sudo systemctl disable llamacpp.service

# View service status
sudo systemctl status llamacpp.service
```

## Configuration

### Model-specific settings

> **Note:** Replace `*` with the actual snapshot hash from your `/opt/models/hub/models--*/snapshots/` directory. Systemd doesn't expand wildcards.

**GPT-OSS-120B** (120B MoE):
```bash
ExecStart=/opt/llama.cpp/build/bin/llama-server \
    -m PATH_TO_MODEL \
    --n-gpu-layers 999 \
    --n-cpu-moe 36 \
    -c 8192 \
    --flash-attn on \
    --jinja \
    --host 0.0.0.0 \
    --port 8502 \
    --api-key YOUR_API_KEY \
    --metrics \
    --log-timestamps
```

**GPT-OSS-20B** (21B):
```bash
ExecStart=/opt/llama.cpp/build/bin/llama-server \
    -m PATH_TO_MODEL \
    --n-gpu-layers 999 \
    -c 8192 \
    --flash-attn on \
    --jinja \
    --host 0.0.0.0 \
    --port 8502 \
    --api-key YOUR_API_KEY \
    --metrics \
    --log-timestamps
```

**Qwen3.5-35B-A3B** (35B MoE):
```bash
ExecStart=/opt/llama.cpp/build/bin/llama-server \
    -m PATH_TO_MODEL \
    --n-gpu-layers 999 \
    --n-cpu-moe 40 \
    -c 8192 \
    --flash-attn on \
    --jinja \
    --host 0.0.0.0 \
    --port 8502 \
    --api-key YOUR_API_KEY \
    --metrics \
    --log-timestamps
```

### Context length (`-c`)

The `-c` flag sets the maximum context length in tokens (combined prompt + response).

- `-c 8192` - 8K tokens (~6,000 words), suitable for most interactive use cases (~2-4 GB KV cache depending on model)
- `-c 32768` - 32K tokens (~24,000 words), enough to fit an entire technical manual or codebase in a single context window (uses significantly more VRAM)
- `-c 0` - use the model's maximum supported context length (often 128K+); **avoid this unless you have substantial free VRAM** - it will allocate a KV cache for the full context window at startup, which can exhaust GPU memory and force inference to fall back to CPU

If the server starts but inference is unexpectedly slow with high CPU usage, an oversized context is the most likely cause. Start conservative (`-c 8192`) and increase only if needed.

## Troubleshooting

### OpenSSL warning during cmake

If cmake prints the following warning, HTTPS support will be disabled:

```
CMake Warning at vendor/cpp-httplib/CMakeLists.txt:150 (message):
  OpenSSL not found, HTTPS support disabled
```

Install the OpenSSL development libraries and re-run cmake:

```bash
# Debian/Ubuntu
sudo apt install -y libssl-dev

# RHEL/Fedora
sudo dnf install -y openssl-devel

# Then re-run cmake
cd /opt/llama.cpp
sudo cmake -B build -DGGML_CUDA=ON
sudo cmake --build build --config Release -j$(nproc)
```

### Service won't start

Check logs for errors:
```bash
sudo journalctl -u llamacpp.service -n 50
```

Common issues:
- **Model file not found**: Systemd doesn't expand wildcards (`*`). Find the actual snapshot hash:
  ```bash
  sudo ls /opt/models/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/
  ```
  Then replace `snapshots/*/` with `snapshots/ACTUAL_HASH/` in the ExecStart line.
- **Permission denied**: Check ownership with `ls -la /opt/llama.cpp`
- **CUDA errors**: Ensure NVIDIA drivers are installed (`nvidia-smi`)
- **Port already in use**: Check if another process is using port 8502

### Performance issues

Check resource usage:
```bash
# CPU/memory
top -p $(pgrep llama-server)

# GPU
nvidia-smi -l 1

# Detailed GPU stats
nvidia-smi dmon -s u
```

### Update llama.cpp

```bash
# Stop service
sudo systemctl stop llamacpp.service

# Update and rebuild
cd /opt/llama.cpp
sudo git pull
sudo cmake --build build --config Release -j$(nproc)

# Start service
sudo systemctl start llamacpp.service
```

## Security considerations

1. **API key**: Use a strong random key (32+ characters)
2. **User isolation**: Run as dedicated `llama` user (not root)
3. **File permissions**: Ensure models are readable only by `llama` user
