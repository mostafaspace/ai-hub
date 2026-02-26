import socket
import os

def get_local_ip():
    """Detect the local LAN IP address."""
    try:
        # connect to an external server (Google DNS) to determine the interface's IP
        # We don't actually send data, just connecting determines the routing.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# --- Network Configuration ---
# Host '0.0.0.0' exposes servers to the local network.
# Use '127.0.0.1' to restrict to localhost only.
HOST = "0.0.0.0"       # Bind address (listen on all interfaces)
PUBLIC_HOST = get_local_ip()  # Announce address (actual LAN IP for clients)

# Ports for each service
TTS_PORT = 8000
MUSIC_PORT = 8001
ASR_PORT = 8002
VISION_PORT = 8003
VIDEO_PORT = 8004

# --- Client / Test Configuration ---
# IP and ports to use when running tests locally.
# Usually 127.0.0.1 is fine for testing on the same machine.
TEST_DEVICE_IP = "127.0.0.1"

# --- Model Paths ---
# Hugging Face Cache Directory
HF_HOME = r"D:\hf_models"

# Qwen3 TTS Models
TTS_MODEL_CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
TTS_MODEL_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
TTS_MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Qwen3 ASR Model
ASR_MODEL_PATH = "Qwen/Qwen2-Audio-7B-Instruct"

# Z-Image Vision Model
VISION_MODEL = "Tongyi-MAI/Z-Image"
VISION_OFFLOAD_CPU = False  # Disabled: 2 GPUs (5090 32GB + 4070Ti 12GB) can hold the full model

# LTX-2 Video Models
VIDEO_MODEL_BASE = os.path.join(HF_HOME, "Lightricks/LTX-2/ltx-2-19b-dev-fp8.safetensors")
VIDEO_GEMMA_ROOT = os.path.join(HF_HOME, "google/gemma-3-12b-it-qat-q4_0-unquantized")
VIDEO_DISTILLED_LORA = os.path.join(HF_HOME, "Lightricks/LTX-2/ltx-2-19b-distilled-lora-384.safetensors")
VIDEO_SPATIAL_UPSAMPLER = os.path.join(HF_HOME, "Lightricks/LTX-2/ltx-2-spatial-upscaler-x2-1.0.safetensors")

# --- Device Configuration ---
# Default strategy: "auto" (Accelerate), "cuda", or "cpu"
DEVICE_MAP_DEFAULT = "auto"

# Helper to ensure HF_HOME is set in environment
os.environ["HF_HOME"] = HF_HOME
VISION_EDIT_MODEL = "Tongyi-MAI/Z-Image-Edit"
VISION_EDIT_MODEL = "Tongyi-MAI/Z-Image-Edit"
