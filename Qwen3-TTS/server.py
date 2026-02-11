import os

# --- FFmpeg Configuration (for MP3/OGG) ---
# MUST be done before importing any packages that might trigger pydub (like qwen_tts or others)
try:
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
    
    import warnings
    # Suppress the pydub warning about missing ffmpeg, as we are setting it manually right after
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pydub import AudioSegment
    
    AudioSegment.converter = ffmpeg_path # Explicitly set converter path
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False

import io
import time
import torch
import uvicorn
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Union, Literal
from contextlib import asynccontextmanager

# Import Qwen3TTS
# Ensure 'qwen-tts' is installed
# (FFmpeg config moved to top)

# Import Qwen3TTS
# Ensure 'qwen-tts' is installed
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    print("Error: qwen-tts package not found. Please install it via 'pip install -e .' in the Qwen3-TTS repo.")
    exit(1)

# --- Configuration ---
HOST = "0.0.0.0"
PORT = 8000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths (Hugging Face IDs or local paths)
MODEL_CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

import threading

# Global model cache to manage VRAM
# Simple LRU-style: keep only one active model to be safe on VRAM
class ModelManager:
    def __init__(self, idle_timeout: float = 600.0, check_interval: float = 30.0): # 10 minutes default
        self.models = {}
        self.current_model_type = None
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
            time.sleep(self.check_interval) # Check every check_interval seconds
            with self.lock:
                if self.current_model_type and (time.time() - self.last_active > self.idle_timeout):
                    print(f"Idle timeout reached ({self.idle_timeout}s). Unloading {self.current_model_type} model...")
                    self.unload_all()

    def unload_all(self):
        """Force unload all models and clear cache."""
        with self.lock:
            self.models.clear()
            self.current_model_type = None
            torch.cuda.empty_cache()

    def get_model(self, model_type: Literal["custom", "design", "base"]):
        with self.lock:
            self.last_active = time.time()

            # If correct model is already loaded, just return it
            if self.current_model_type == model_type and model_type in self.models:
                return self.models[model_type]
            
            # Unload previous model if it's different to save VRAM
            if self.current_model_type and self.current_model_type != model_type:
                print(f"Auto-switching modes: Unloading {self.current_model_type} to load {model_type}...")
                self.unload_all()
            
            # Load new model if not in cache (idempotent check)
            if model_type not in self.models:
                path = ""
                if model_type == "custom": path = MODEL_CUSTOM
                elif model_type == "design": path = MODEL_DESIGN
                elif model_type == "base": path = MODEL_BASE
                
                print(f"Loading {model_type} model from {path}...")
                # FORCE single GPU to avoid multi-device tensor errors (cuda:0 vs cuda:1)
                device_arg = "cuda:0" if DEVICE == "cuda" else "cpu"
                
                self.models[model_type] = Qwen3TTSModel.from_pretrained(
                    path,
                    device_map=device_arg,
                    dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    # attn_implementation="flash_attention_2" # Enable if supported hardware
                )
                self.current_model_type = model_type
                
            return self.models[model_type]

manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are now lazy-loaded on request
    yield
    # Cleanup on exit
    manager.unload_all()

app = FastAPI(title="Qwen3 TTS Advanced API", lifespan=lifespan)

# --- Schemas ---

class TTSSpeechRequest(BaseModel):
    input: str
    voice: str = "Vivian" # Default Qwen speaker
    language: Optional[str] = "Auto"
    instruct: Optional[str] = None
    response_format: Literal["wav", "mp3", "pcm", "ogg"] = "mp3"

class VoiceDesignRequest(BaseModel):
    input: str
    instruct: str
    language: Optional[str] = "Auto"
    response_format: Literal["wav", "mp3", "pcm", "ogg"] = "mp3"

# --- Helper ---
def audio_to_stream(audio_data, sample_rate, fmt):
    # If MP3 requested and FFmpeg available, use pydub
    if fmt == "mp3" and HAS_FFMPEG:
        try:
            # Convert float32 numpy array to int16 PCM for pydub
            # Qwen output is usually float [-1, 1]
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create AudioSegment
            seg = AudioSegment(
                audio_int16.tobytes(), 
                frame_rate=sample_rate,
                sample_width=2, 
                channels=1
            )
            
            buffer = io.BytesIO()
            seg.export(buffer, format="mp3", bitrate="192k")
            buffer.seek(0)
            return StreamingResponse(buffer, media_type="audio/mpeg")
        except Exception as e:
            print(f"MP3 Conversion Failed: {e}. Falling back to WAV.")

    # OGG (Opus) Support for WhatsApp
    if fmt == "ogg" and HAS_FFMPEG:
        try:
            # Convert float32 numpy array to int16 PCM for pydub
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            seg = AudioSegment(
                audio_int16.tobytes(), 
                frame_rate=sample_rate,
                sample_width=2, 
                channels=1
            )
            
            buffer = io.BytesIO()
            # WhatsApp prefers basic OGG/Opus
            seg.export(buffer, format="ogg", codec="libopus", bitrate="24k") 
            buffer.seek(0)
            return StreamingResponse(buffer, media_type="audio/ogg")
        except Exception as e:
            print(f"OGG Conversion Failed: {e}. Falling back to WAV.")
    
    # Fallback / WAV handling
    buffer = io.BytesIO()
    
    # Soundfile formats for export
    sf_fmt = "WAV"
    if fmt == "pcm": sf_fmt = "RAW"
    
    try:
        sf.write(buffer, audio_data, sample_rate, format=sf_fmt, subtype='PCM_16' if fmt=='pcm' else None)
    except Exception:
        print("Warning: format write failed, falling back to WAV")
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        fmt = "wav"

    buffer.seek(0)
    media_type = f"audio/{fmt}"
    if fmt == "pcm": media_type = "audio/l16"
    return StreamingResponse(buffer, media_type=media_type)

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "running", "device": DEVICE}

@app.get("/v1/models")
def list_models():
    """List available models."""
    return {
        "data": [
            {"id": "qwen-tts-custom", "object": "model", "owned_by": "qwen", "permission": []},
            {"id": "qwen-tts-design", "object": "model", "owned_by": "qwen", "permission": []},
            {"id": "qwen-tts-base", "object": "model", "owned_by": "qwen", "permission": []},
        ],
        "object": "list"
    }

@app.get("/v1/audio/voices")
@app.get("/v1/audio/voices/list")
@app.get("/v1/audio/speakers")
def list_voices():
    """List available voices."""
    # Hardcoded list of known speakers for now.
    # TODO: Dynamically fetch from model if supported.
    voices = [
        {"voice_id": "Vivian", "name": "Vivian", "category": "premade"}, # Female
        {"voice_id": "Generic_Female", "name": "Generic_Female", "category": "premade"}, # Placeholder
        {"voice_id": "Generic_Male", "name": "Generic_Male", "category": "premade"}, # Placeholder
    ]
    return {"voices": voices}

@app.post("/v1/audio/speech")
def generate_speech(req: TTSSpeechRequest):
    """
    Standard TTS (CustomVoice model) with preset speakers.
    """
    try:
        model = manager.get_model("custom")
        
        # Determine language. If auto, pass "Auto" or None? 
        # API says "Auto" or omit.
        lang = req.language if req.language and req.language != "Auto" else "Auto"
        if lang == "Auto": lang = None # Library expects None or specific? wrapper usually handles it.
        # Check library signature. 
        # If library expects string "Auto" or specific language. 
        # Checking logic: if lang is None or "Auto", let model decide used 'Auto' in example
        
        print(f"Generating CustomVoice: '{req.input[:30]}...' Speaker: {req.voice}")
        
        wavs, sr = model.generate_custom_voice(
            text=req.input,
            language=req.language if req.language != "Auto" else "Auto",
            speaker=req.voice, # e.g. "Vivian", "Ryan"
            instruct=req.instruct
        )
        
        return audio_to_stream(wavs[0], sr, req.response_format)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/tts")
def generate_tts_alias(req: TTSSpeechRequest):
    """Alias for /v1/audio/speech to support different clients."""
    return generate_speech(req)


@app.post("/v1/audio/voice_design")
def generate_voice_design(req: VoiceDesignRequest):
    """
    Voice Design model - create new voice from text description.
    """
    try:
        model = manager.get_model("design")
        
        print(f"Generating VoiceDesign: '{req.input[:30]}...' Prompt: {req.instruct[:30]}...")
        
        wavs, sr = model.generate_voice_design(
            text=req.input,
            language=req.language if req.language != "Auto" else None,
            instruct=req.instruct
        )
        return audio_to_stream(wavs[0], sr, req.response_format)
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/voice_clone")
async def generate_voice_clone(
    text: str = Form(...),
    ref_text: str = Form(...),
    language: str = Form("Auto"),
    ref_audio: UploadFile = File(...),
    response_format: str = Form("mp3")
):
    """
    Voice Cloning (Base model) - clone from audio file.
    """
    try:
        model = manager.get_model("base")
        
        print(f"Cloning Voice. Text: '{text[:30]}...' Ref: {ref_audio.filename}")
        
        # Read uploaded audio
        audio_bytes = await ref_audio.read()
        audio_stream = io.BytesIO(audio_bytes)
        
        # Load audio for qwen (expects path or tuple)
        # We can pass data directly if supported, or save temp file.
        # Using temp file is safer for 'librosa.load' or 'sf.read' internals usually
        temp_filename = f"temp_ref_{int(time.time())}.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio_bytes)
            
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language if language != "Auto" else "Auto",
                ref_audio=temp_filename,
                ref_text=ref_text
            )
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        
        return audio_to_stream(wavs[0], sr, response_format)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
