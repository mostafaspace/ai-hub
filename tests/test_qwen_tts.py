import sys
import os

sys.path.append(r"d:\antigravity\Qwen3-TTS")

from qwen_tts import Qwen3TTSModel
import config

try:
    print("Loading model...")
    # Load model on CPU for testing the config issue
    model = Qwen3TTSModel.from_pretrained(config.TTS_MODEL_CUSTOM, device_map="cuda:0")
    print("Model loaded.")
    
    print("Generating voice...")
    wav, sr = model.generate_custom_voice(text="Hello world, this is a test.", language="Auto", speaker="Vivian")
    print("Generation successful.")
except Exception as e:
    import traceback
    traceback.print_exc()
