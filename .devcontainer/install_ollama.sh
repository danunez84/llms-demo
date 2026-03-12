#!/bin/bash
# Install zstd (required by the Ollama installer for extraction)
sudo apt-get update && sudo apt-get install -y zstd

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
