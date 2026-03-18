
import os
import sys
import torch
import time

# Mocking config
HF_HOME = r"D:\hf_models"
os.environ["HF_HOME"] = HF_HOME

# Add Qwen3-TTS and parent to path
sys.path.append(r"d:\antigravity\Qwen3-TTS")
sys.path.append(r"d:\antigravity")

try:
    from qwen_tts import Qwen3TTSModel
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

model_path = os.path.join(HF_HOME, "hub", "models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice")
# Or just use the repo ID if it's in the hub
repo_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

print(f"Attempting to load {repo_id}...")
start_time = time.time()
try:
    model = Qwen3TTSModel.from_pretrained(
        repo_id,
        device_map="cuda:0",
        dtype=torch.bfloat16
    )
    print(f"Load successful in {time.time() - start_time:.2f}s")
except Exception as e:
    print(f"Load failed: {e}")
