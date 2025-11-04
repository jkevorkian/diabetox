import os
import subprocess
from pathlib import Path

# ==== CONFIG ====
# Path to your GGUF model
gguf_path = Path("models/Qwen2.5-0.5B-Instruct-Q8_0.gguf")

# Name for intermediate HF model
hf_dir = Path("models/qwen2.5-0.5b-hf")

# Output path for WebLLM-usable weights
webllm_out = Path("models/qwen2.5-0.5b-webllm")

# ==== 1️⃣ Convert GGUF -> Hugging Face format (float16) ====
print("Converting GGUF -> HF...")
subprocess.run([
    "python3", "-m", "llama_cpp.convert",  # from llama.cpp tools
    "--from", "gguf",
    "--to", "hf",
    "--input", str(gguf_path),
    "--output", str(hf_dir)
], check=True)

# ==== 2️⃣ Convert HF -> WebLLM shards ====
# This requires webllm’s weight conversion utility
print("Converting HF -> WebLLM...")
subprocess.run([
    "python3", "-m", "webllm.convert_weights",
    "--model-path", str(hf_dir),
    "--output-path", str(webllm_out),
    "--quantization", "q4f16_1"  # you can use q4f16_1 or leave unquantized
], check=True)

print(f"✅ Conversion complete!\nWebLLM model at: {webllm_out}")
