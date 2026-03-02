import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import asyncio
from Qwen3_TTS.qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
import logging
from transformers.generation.streamers import BaseStreamer

logging.basicConfig(level=logging.INFO)

class DebugStreamer(BaseStreamer):
    def __init__(self):
        super().__init__()
        self.chunk_count = 0

    def put(self, value):
        print(f"[DebugStreamer] Received chunk {self.chunk_count} with shape: {value.shape}")
        self.chunk_count += 1

    def end(self):
        print("[DebugStreamer] Ended!")

def main():
    print("Loading test model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # To save time loading, perhaps I should just run the test script test_tts_stream.py directly, 
    # since we already modified the model.
    # Actually wait, test_tts_stream.py connects via WEBSOCKET.
    # It will trigger the streamer!
    pass

if __name__ == "__main__":
    main()
