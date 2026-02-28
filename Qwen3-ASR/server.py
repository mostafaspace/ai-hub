
import os
import io
import time
import torch
import uvicorn
import logging
import urllib.parse
import librosa
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Union, Any
from pydantic import BaseModel
import threading
import gc

# --- Configuration ---
import sys
# Add parent directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

HOST = config.HOST
PORT = config.ASR_PORT  # Port 8000 is Qwen3-TTS, 8001 is ACE-Step
MODEL_PATH = config.ASR_MODEL_PATH

# logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Qwen3-ASR")

def _resolve_device_map() -> str:
    """
    Resolve device map strategy.
    
    If QWEN_ASR_DEVICE is set, use it.
    Otherwise, default to config.DEVICE_MAP_DEFAULT.
    """
    env_device = os.getenv("QWEN_ASR_DEVICE")
    if env_device:
        return env_device.strip()
    
    return config.DEVICE_MAP_DEFAULT

DEVICE_MAP = _resolve_device_map()
# DEVICE is used for tensors that need manual placement, but with device_map="auto"
# the model handles placement. We'll default to cuda:0 for inputs if available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Models ---
class ModelManager:
    def __init__(self, idle_timeout: float = 60.0, check_interval: float = 10.0):
        self.model = None
        self.processor = None
        self.last_active = time.time()
        self.idle_timeout = idle_timeout
        self.check_interval = check_interval
        self.lock = threading.RLock()
        
        # Start background monitor
        self.monitor_thread = threading.Thread(target=self._idle_monitor, daemon=True)
        self.monitor_thread.start()

    def _idle_monitor(self):
        """Background thread to unload models after idle timeout."""
        while True:
            time.sleep(self.check_interval)
            # Lock-free check (reading a float is atomic in CPython)
            if self.model and (time.time() - self.last_active > self.idle_timeout):
                with self.lock:
                    # Double-check under lock before unloading
                    if self.model and (time.time() - self.last_active > self.idle_timeout):
                        logger.info(f"Idle timeout reached ({self.idle_timeout}s). Unloading Qwen2-Audio model...")
                        self._unload_internal()

    def _unload_internal(self):
        """Unload without acquiring lock (caller must hold lock)."""
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("[ASR] Model unloaded, VRAM freed.")

    def unload(self):
        """Force unload model and clear cache."""
        with self.lock:
            self._unload_internal()

    def touch(self):
        """Reset the idle timer. Call after every generation completes."""
        self.last_active = time.time()

    def get_model(self):
        with self.lock:
            self.last_active = time.time()
            
            if self.model and self.processor:
                return self.model, self.processor
            
            logger.info(f"Loading Qwen2-Audio model from {MODEL_PATH} on {DEVICE_MAP}...")
            try:
                self.processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    MODEL_PATH,
                    device_map=DEVICE_MAP,
                    trust_remote_code=True,
                    dtype=torch.float16 if DEVICE == "cuda" else torch.float32
                )
                self.model.eval()
                logger.info("✅ Model loaded successfully.")
                return self.model, self.processor
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                self.unload() # Cleanup potential partial load
                raise e

manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lazy loading - nothing to do on startup
    yield
    # Cleanup on exit
    manager.unload()

app = FastAPI(title="Qwen3-ASR API", version="1.0", lifespan=lifespan)
try:
    from api_utils import GracefulJSONRoute
    app.router.route_class = GracefulJSONRoute
except ImportError as e:
    logger.warning(f"Could not load GracefulJSONRoute: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas ---

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {
        "status": "running", 
        "device": DEVICE, 
        "device_map": DEVICE_MAP,
        "model_loaded": manager.model is not None
    }

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),
    language: str = Form(None), 
    prompt: str = Form(None),
    response_format: str = Form("json"), 
    temperature: float = Form(0.0), 
):
    """
    OpenAI-compatible transcription endpoint.
    Actually runs "Audio Analysis" with a prompt asking to transcribe.
    """
    """
    OpenAI-compatible transcription endpoint.
    Actually runs "Audio Analysis" with a prompt asking to transcribe.
    """
    model, msg_processor = manager.get_model()

    try:
        # Read audio file
        content = await file.read()
        
        import tempfile
        
        # Determine extension from filename if possible
        ext = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        if not ext: ext = ".wav"
        
        logger.info(f"Transcription request: filename={file.filename}, ext={ext}, content_size={len(content)} bytes")
            
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            
        try:
            audio_array, sampling_rate = librosa.load(tmp_path, sr=None)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        duration = len(audio_array) / sampling_rate if sampling_rate > 0 else 0
        amplitude = float(np.max(np.abs(audio_array))) if len(audio_array) > 0 else 0
        logger.info(f"Audio loaded: sr={sampling_rate}, duration={duration:.2f}s, "
                     f"samples={len(audio_array)}, max_amplitude={amplitude:.4f}")
        
        # Reject empty/silent audio early
        if len(audio_array) == 0 or duration < 0.1:
            logger.warning("Audio too short or empty!")
            return {"text": "[Error: Audio file is empty or too short to transcribe]"}
        if amplitude < 0.001:
            logger.warning(f"Audio appears silent (max amplitude: {amplitude:.6f})")
            return {"text": "[Error: Audio file appears to be silent]"}
        
        target_sr = msg_processor.feature_extractor.sampling_rate
        
        # Resample if needed
        if sampling_rate != target_sr:
             audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=target_sr)
        
        # Simple direct prompt - Qwen2-Audio responds best to direct questions
        user_prompt = "What does the person say?" if not prompt else prompt
        
        conversation = [
             {"role": "user", "content": [
                 {"type": "audio", "audio_url": "placeholder_url_handled_manually"}, # We will swap this
                 {"type": "text", "text": user_prompt}
             ]}
        ]
        
        # The apply_chat_template will format the text.
        text_prompt = msg_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = msg_processor(text=text_prompt, audio=[audio_array], return_tensors="pt", padding=True, sampling_rate=target_sr)
        inputs = inputs.to(model.device)
             
        generate_ids = model.generate(**inputs, max_new_tokens=256)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        
        response_text = msg_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return {"text": response_text}

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        manager.touch()

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    Chat endpoint supporting audio.
    Messages format: Use Qwen2-Audio format: 
    content=[{"type": "audio", "audio_url": "..."}, {"type": "text", "text": "..."}]
    """
    model, msg_processor = manager.get_model()

    try:
        # We need to process the messages and download any audio_urls manually 
        # UNLESS we trust transformers to do it (it does valid URL downloads).
        # But for local files or base64 we might need logic.
        
        # For simplicity, we assume URLs are accessible or are handled by transformers if public.
        # If user passes base64 data URI, we might need to handle it.
        
        # NOTE: transformers `apply_chat_template` produces a string prompt.
        # The `processor` then expects `text` and `audios`.
        # We need to extract all audios from messages in order.
        
        audios = []
        
        # Deep copy messages to avoid mutating original request if we need to
        processed_messages = []
        
        import requests
        import tempfile
        from io import BytesIO
        
        # Helper to load audio
        def load_audio_from_url(url: str) -> tuple[np.ndarray, int]:
             content: Optional[bytes] = None
             ext: str = ".wav"
             
             if url.startswith("data:audio/"):
                  import base64
                  header, encoded = url.split(",", 1)
                  content = base64.b64decode(encoded)
                  # Try to guess extension from header
                  if "mp3" in header: ext = ".mp3"
                  elif "webm" in header: ext = ".webm"
             elif url.startswith("file://"):
                  # Use partition to avoid indexing/slicing lints
                  file_path = urllib.parse.unquote(url.partition("file://")[2])
                  # Windows paths like file:///C:/... result in /C:/...
                  # Use slicing instead of indexing to avoid potential lint picky-ness
                  if file_path.startswith("/") and len(file_path) > 2 and file_path[1:2] == ":":
                       file_path = file_path[1:]
                  with open(file_path, "rb") as f:
                       content = f.read()
                  ext = os.path.splitext(file_path)[1]
             elif os.path.exists(url):
                  with open(url, "rb") as f:
                       content = f.read()
                  ext = os.path.splitext(url)[1]
             else:
                  # Public URL
                  try:
                       resp = requests.get(url, timeout=10)
                       resp.raise_for_status()
                  except Exception as e:
                       logger.warning(f"Initial load failed for {url}: {e}. Retrying with verify=False...")
                       resp = requests.get(url, timeout=10, verify=False)
                       resp.raise_for_status()
                  content = resp.content
                  ext = os.path.splitext(urllib.parse.urlparse(url).path)[1]

             if not content:
                  raise ValueError(f"Could not load content from {url}")
             
             if not ext: ext = ".wav"
             
             # Reuse robust temp file loading logic
             with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                  tmp.write(content)
                  tmp_path = tmp.name
             try:
                  s, sr = librosa.load(tmp_path, sr=None)
             finally:
                  if os.path.exists(tmp_path):
                       os.remove(tmp_path)
             
             if s is None:
                  raise ValueError("librosa.load returned None")
             return (s, sr)
        
        target_sr = msg_processor.feature_extractor.sampling_rate

        for msg in req.messages:
            new_content = []
            if isinstance(msg.content, list):
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "audio":
                        audio_url = item.get("audio_url")
                        if audio_url:
                            # Load and allow processor to handle key matching?
                            # Actually, Qwen2Audio processor expects a list of raw audios matching the order of <|AUDIO|> tokens?
                            # apply_chat_template handles inserting tokens?
                            # It inserts <|audio_bos|><|AUDIO|><|audio_eos|>
                            
                            # We need to load the audio into array
                            try:
                                audio_array, sr = load_audio_from_url(audio_url)
                                # Resample
                                if sr != target_sr:
                                     if audio_array.ndim > 1: audio_array = audio_array.mean(axis=1)
                                     audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
                                
                                audios.append(audio_array)
                                new_content.append(item) # Keep item for template
                            except Exception as ex:
                                logger.error(f"Failed to load audio {audio_url}: {ex}")
                                # Skip or error?
                                new_content.append({"type": "text", "text": "[Audio Load Failed]"})
                    else:
                        new_content.append(item)
                processed_messages.append({"role": msg.role, "content": new_content})
            else:
                processed_messages.append({"role": msg.role, "content": msg.content})

        text_prompt = msg_processor.apply_chat_template(processed_messages, add_generation_prompt=True, tokenize=False)
        
        inputs = msg_processor(text=text_prompt, audio=audios if audios else None, return_tensors="pt", padding=True, sampling_rate=target_sr)
        inputs = inputs.to(model.device)

        generate_ids = model.generate(
            **inputs, 
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p
        )
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        response_text = msg_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return {
            "id": "chatcmpl-" + str(int(time.time())),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "Qwen2-Audio-7B-Instruct",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }]
        }

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        manager.touch()

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
