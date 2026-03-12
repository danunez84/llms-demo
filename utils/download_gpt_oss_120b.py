'''Download the openai/gpt-oss-120b GGUF model from HuggingFace.

Downloads into models/hugging_face (respects $HF_HOME if set).
Only downloads GGUF files to avoid pulling unnecessary formats.

Usage:
    python utils/download_gpt_oss_120b.py
'''

import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

snapshot_download(
    repo_id='openai/gpt-oss-120b',
    allow_patterns=['*.gguf'],
    resume_download=True,
    token=os.environ.get('HF_TOKEN'),
)
