
import os
import torch
import sys
import traceback

# Add ACE-Step to path
sys.path.append(os.path.join(os.getcwd(), "ACE-Step-1.5"))

print("--- GPU Selection Logic Verification ---")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

print("\n--- Detection Results ---")

def check_qwen_tts_logic():
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    target_idx = 1 if gpu_count > 1 else 0
    return f"cuda:{target_idx}" if gpu_count > 0 else "cpu"

print(f"Qwen3 TTS (logic): Target Device = {check_qwen_tts_logic()}")

try:
    from acestep.gpu_config import get_preferred_device_index
    pref_idx = get_preferred_device_index()
    print(f"ACE-Step (gpu_config): Preferred Index = {pref_idx} (cuda:{pref_idx})")
except Exception as e:
    print(f"ACE-Step (gpu_config) Import Error: {e}")
    # traceback.print_exc()

try:
    from acestep.handler import _resolve_preferred_cuda_device as handler_device
    print(f"ACE-Step (handler): Preferred Device = {handler_device()}")
except Exception as e:
    print(f"ACE-Step (handler) Import Error: {e}")

print("\n--- Environmental Overrides ---")
print(f"QWEN_TTS_DEVICE: {os.getenv('QWEN_TTS_DEVICE', 'Not Set')}")
print(f"ACESTEP_LM_DEVICE: {os.getenv('ACESTEP_LM_DEVICE', 'Not Set')}")
print(f"ACESTEP_DEVICE: {os.getenv('ACESTEP_DEVICE', 'Not Set')}")
