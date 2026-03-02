import os
os.environ["HF_HOME"] = r"D:\hf_models"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import sys
sys.path.append(r"d:\antigravity\Qwen3-TTS")

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
import time

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", 
    device_map="cuda", 
    dtype=torch.bfloat16,
    attn_implementation="sdpa"
)
print("Model loaded. Generating...")
start = time.time()
try:
    wavs, sr = model.generate_custom_voice(
        text="Testing one two three.",
        speaker="Vivian",
        language="Auto",
        instruct="",
        max_new_tokens=100
    )
    print(f"Generated successfully in {time.time() - start} seconds. Length: {len(wavs)}")
except Exception as e:
    import traceback
    traceback.print_exc()
