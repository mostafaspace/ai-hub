import os

# --- Network Configuration ---
# Host '0.0.0.0' exposes servers to the local network.
# Use '127.0.0.1' to restrict to localhost only.
HOST = "0.0.0.0"

# Ports for each service
TTS_PORT = 8000
MUSIC_PORT = 8001
ASR_PORT = 8002

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

# --- Device Configuration ---
# Default strategy: "auto" (Accelerate), "cuda", or "cpu"
DEVICE_MAP_DEFAULT = "auto"

# Helper to ensure HF_HOME is set in environment
os.environ["HF_HOME"] = HF_HOME
