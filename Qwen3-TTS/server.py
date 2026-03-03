import os

# --- FFmpeg Configuration (for MP3/OGG) ---
# MUST be done before importing any packages that might trigger pydub (like qwen_tts or others)
try:
    import imageio_ffmpeg
    import imageio_ffmpeg
    # ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Force use of local copy if it exists to avoid long path/permission issues
    local_ffmpeg = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ffmpeg.exe"))
    if os.path.exists(local_ffmpeg):
        ffmpeg_path = local_ffmpeg
    else:
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
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Union, Literal
from contextlib import asynccontextmanager
from transformers.generation.streamers import BaseStreamer

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
import sys
# Add parent directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

HOST = config.HOST
PORT = config.TTS_PORT

def _resolve_device_map() -> str:
    """
    Resolve device map strategy.
    
    If QWEN_TTS_DEVICE is set, use it.
    Otherwise, default to config.DEVICE_MAP_DEFAULT.
    """
    env_device = os.getenv("QWEN_TTS_DEVICE")
    if env_device:
        return env_device.strip()

    default_device = str(config.DEVICE_MAP_DEFAULT).strip()
    if default_device == "auto" and torch.cuda.is_available():
        # Auto-sharding frequently places parts of Qwen3-TTS on multiple GPUs,
        # which can trigger cross-device tensor errors in generation.
        return "cuda:0"
    return default_device


DEVICE_MAP = _resolve_device_map()
DEVICE = "cuda" if (DEVICE_MAP.startswith("cuda") or DEVICE_MAP == "auto") else "cpu"

def _resolve_tts_dtype() -> torch.dtype:
    env_dtype = os.getenv("QWEN_TTS_DTYPE", "").strip().lower()
    if env_dtype in {"float16", "fp16", "half"}:
        return torch.float16
    if env_dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if env_dtype in {"float32", "fp32"}:
        return torch.float32
    # Qwen models MUST use bfloat16 to avoid overflowing hidden states. float16 results in static gibberish.
    return torch.bfloat16 if DEVICE == "cuda" else torch.float32

TTS_DTYPE = _resolve_tts_dtype()

# Model paths (from config)
MODEL_CUSTOM = config.TTS_MODEL_CUSTOM
MODEL_DESIGN = config.TTS_MODEL_DESIGN
MODEL_BASE = config.TTS_MODEL_BASE

import threading
import gc
from transformers.generation.streamers import BaseStreamer

# --- Custom Real-Time Audio Streamer ---
class Qwen3AudioStreamer(BaseStreamer):
    """
    Subclasses HuggingFace BaseStreamer to intercept audio codec tokens,
    decode them, and pump raw PCM bytes into an asyncio.Queue for WebSockets.
    """
    def __init__(self, model, sample_rate: int = 24000):
        self.model = model  # The Qwen3TTSModel instance
        self.sample_rate = sample_rate
        self.audio_queue = asyncio.Queue()
        self.token_cache = []
        # Qwen3 processes audio in frames. We need enough codec tokens to make a decent sized audio chunk.
        self.chunk_size = 50 # Process every N tokens (experiment to find a balance between latency and stutter)
        self.loop = asyncio.get_running_loop()
        self.is_done = False
        self.stream_count = 0

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]
            
        self.token_cache.append(value.tolist())
        
        # If we have accumulated enough tokens for a chunk (modulo division so it streams every N tokens)
        if len(self.token_cache) % self.chunk_size == 0:
            target_device = self.model.model.device
            codes_tensor = torch.tensor(self.token_cache).to(target_device) # Shape [length, 16]
            wavs_all, _ = self.model.model.speech_tokenizer.decode([{"audio_codes": codes_tensor}])
            
            wav_chunk = wavs_all[0]
            # Derive samples per token to calculate how much the end of the audio wave grew
            samples_per_token = len(wav_chunk) // len(self.token_cache)
            new_samples_len = samples_per_token * self.chunk_size
            
            audio_data = wav_chunk[-new_samples_len:]
            audio_data = np.clip(audio_data, -1.0, 1.0)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            pcm_bytes = audio_int16.tobytes()
            
            self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, pcm_bytes)
            self.stream_count += 1
    def end(self):
        """Called when generation finishes."""
        self.is_done = True
        self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, None) # Poison pill

# Global model cache to manage VRAM
# Simple LRU-style: keep only one active model to be safe on VRAM
class ModelManager:
    def __init__(self, idle_timeout: float = 60.0, check_interval: float = 10.0): # 60 seconds default
        self.models = {}
        self.current_model_type = None
        self.last_active = time.time()
        self.idle_timeout = idle_timeout
        self.check_interval = check_interval
        self.is_generating = False
        self.lock = threading.RLock()
        
        # Start background monitor
        self.monitor_thread = threading.Thread(target=self._idle_monitor, daemon=True)
        self.monitor_thread.start()

    def _idle_monitor(self):
        """Background thread to unload models after idle timeout."""
        while True:
            time.sleep(self.check_interval)
            # Lock-free check (reading a float is atomic in CPython)
            if self.current_model_type and (time.time() - self.last_active > self.idle_timeout):
                with self.lock:
                    # Double-check under lock before unloading
                    if self.current_model_type and not self.is_generating and (time.time() - self.last_active > self.idle_timeout):
                        print(f"Idle timeout reached ({self.idle_timeout}s). Unloading {self.current_model_type} model...")
                        self._unload_internal()

    def _unload_internal(self):
        """Unload without acquiring lock (caller must hold lock)."""
        self.models.clear()
        self.current_model_type = None
        gc.collect()
        torch.cuda.empty_cache()
        print("[TTS] Models unloaded, VRAM freed.")

    def unload_all(self):
        """Force unload all models and clear cache."""
        with self.lock:
            self._unload_internal()

    def touch(self):
        """Reset the idle timer. Call after every generation completes."""
        self.last_active = time.time()

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
                # FORCE a single GPU to avoid cross-device tensor errors.
                device_arg = DEVICE_MAP if DEVICE == "cuda" else "cpu"
                
                model_instance = Qwen3TTSModel.from_pretrained(
                    path,
                    device_map=device_arg,
                    dtype=TTS_DTYPE,
                    # attn_implementation="flash_attention_2" # Enable if supported hardware
                )
                if hasattr(model_instance, 'config'):
                    # Some configs strictly raise AttributeError if missing instead of returning None
                    try:
                        _ = model_instance.config.pad_token_id
                    except AttributeError:
                        model_instance.config.pad_token_id = getattr(model_instance.config, 'eos_token_id', 0)

                self.models[model_type] = model_instance
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
try:
    from api_utils import GracefulJSONRoute
    app.router.route_class = GracefulJSONRoute
except ImportError as e:
    print(f"Warning: Could not load GracefulJSONRoute: {e}")

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
    # Normalize expected model output range before integer conversion.
    audio_data = np.clip(np.asarray(audio_data, dtype=np.float32), -1.0, 1.0)

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
            fmt = "wav"

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
            fmt = "wav"
    
    # Fallback / WAV handling
    buffer = io.BytesIO()
    
    # Soundfile formats for export
    sf_fmt = "WAV"
    if fmt == "pcm": sf_fmt = "RAW"
    
    try:
        sf.write(buffer, audio_data, sample_rate, format=sf_fmt, subtype='PCM_16')
    except Exception:
        print("Warning: format write failed, falling back to WAV")
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="WAV", subtype='PCM_16')
        fmt = "wav"

    buffer.seek(0)
    media_type = f"audio/{fmt}"
    if fmt == "pcm": media_type = "audio/l16"
    return StreamingResponse(buffer, media_type=media_type)

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "device": DEVICE,
        "device_map": DEVICE_MAP,
        "dtype": str(TTS_DTYPE).replace("torch.", ""),
        "model_loaded": manager.current_model_type is not None,
    }

@app.get("/v1/internal/status")
def internal_status():
    return {
        "status": "idle",
        "model_loaded": manager.current_model_type is not None,
        "current_model": manager.current_model_type,
        "device": DEVICE,
    }

@app.post("/v1/internal/unload")
@app.get("/v1/internal/unload")
def manual_unload():
    """Manually unload all models."""
    manager.unload_all()
    return {"status": "models unloaded"}

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
        
        manager.is_generating = True
        wavs, sr = model.generate_custom_voice(
            text=req.input,
            language=req.language if req.language != "Auto" else "Auto",
            speaker=req.voice, # e.g. "Vivian", "Ryan"
            instruct=req.instruct,
            non_streaming_mode=True,
            max_new_tokens=400
        )
        
        return audio_to_stream(wavs[0], sr, req.response_format)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        manager.is_generating = False
        manager.touch()

@app.post("/v1/audio/speech/stream")
async def generate_speech_stream(req: TTSSpeechRequest):
    """
    HTTP Chunked Streaming TTS.
    Returns a WAV file with audio chunks streamed as they're generated.
    Uses the Qwen3AudioStreamer to push PCM chunks via StreamingResponse.
    """
    import struct

    model = manager.get_model("custom")
    manager.is_generating = True

    streamer = Qwen3AudioStreamer(model, sample_rate=24000)

    def run_generation():
        try:
            model.generate_custom_voice(
                text=req.input,
                language=req.language if req.language != "Auto" else "Auto",
                speaker=req.voice,
                streamer=streamer,
                non_streaming_mode=False
            )
        except Exception as e:
            print(f"[Stream Generate Error] {e}")
            import traceback
            traceback.print_exc()
        finally:
            streamer.end()

    # Start generation in background thread
    gen_thread = threading.Thread(target=run_generation, daemon=True)
    gen_thread.start()

    async def audio_generator():
        """Yield WAV header + PCM chunks as they arrive."""
        sample_rate = 24000
        num_channels = 1
        bits_per_sample = 16

        # Write a WAV header with unknown data size (0xFFFFFFFF)
        # This allows streaming — the browser will still play it
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', 0xFFFFFFFF,  # Placeholder size
            b'WAVE',
            b'fmt ', 16,  # PCM format chunk size
            1,  # PCM format
            num_channels,
            sample_rate,
            sample_rate * num_channels * bits_per_sample // 8,  # byte rate
            num_channels * bits_per_sample // 8,  # block align
            bits_per_sample,
            b'data', 0xFFFFFFFF  # Placeholder data size
        )
        yield header

        total_bytes = 0
        while True:
            chunk = await streamer.audio_queue.get()
            if chunk is None:
                break
            yield chunk
            total_bytes += len(chunk)

        print(f"[Stream] Finished. Total PCM: {total_bytes / 1024:.1f} KB")
        manager.is_generating = False
        manager.touch()

    return StreamingResponse(
        audio_generator(),
        media_type="audio/wav",
        headers={"Transfer-Encoding": "chunked"}
    )

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
        
        manager.is_generating = True
        wavs, sr = model.generate_voice_design(
            text=req.input,
            language=req.language if req.language != "Auto" else None,
            instruct=req.instruct
        )
        return audio_to_stream(wavs[0], sr, req.response_format)
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        manager.is_generating = False
        manager.touch()

@app.websocket("/v1/audio/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Real-Time Streaming WebSocket Endpoint.
    1. Client connects.
    2. Client sends JSON config `{"input": "hello", "voice": "Vivian"}`
    3. Server starts generating in a background thread and pushes raw PCM int16 chunks incrementally down the socket.
    """
    await websocket.accept()
    print("[WebSocket] Client connected for TTS stream.")
    
    try:
        # Wait for config payload from client
        config_data = await websocket.receive_json()
        text = config_data.get("input", "")
        voice = config_data.get("voice", "Vivian")
        language = config_data.get("language", "Auto")
        
        if not text:
            await websocket.send_json({"error": "No input text provided."})
            await websocket.close()
            return

        print(f"[WebSocket] Streaming requested for voice: {voice}")
        
        # Prevent auto-unload deadlocks and load the model
        model = manager.get_model("custom")
        manager.is_generating = True
        
        # We assume sample rate is 24000 for Qwen3-TTS
        streamer = Qwen3AudioStreamer(model, sample_rate=24000)
        
        # Offload 15s+ blocking PyTorch inference generation to background thread
        def run_generation():
            try:
                # Modifying generate_custom_voice signature invocation to pass down streamer
                model.generate_custom_voice(
                    text=text,
                    language=language if language != "Auto" else "Auto",
                    speaker=voice,
                    audio_streamer=streamer, # The hook
                    non_streaming_mode=False # VERY IMPORTANT for streaming to trigger
                )
            except Exception as e:
                print(f"[WebSocket Generate Thread Error] {e}")
                import traceback
                traceback.print_exc()
            finally:
                streamer.end()
                
        thread = threading.Thread(target=run_generation)
        thread.start()
        
        # Consume the queue and pipe to websocket
        while True:
            chunk = await streamer.audio_queue.get()
            if chunk is None:
                # Poison pill received. Generation is complete.
                break
                
            # Send binary PCM chunk to client directly
            await websocket.send_bytes(chunk)
            
        print("[WebSocket] Stream complete. Connection gracefully closing.")
        
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected abruptly before generation could finish.")
    except Exception as e:
        print(f"[WebSocket Error] {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass
    finally:
        manager.is_generating = False
        manager.touch()

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
        
        # Initialize paths for cleanup safety
        temp_input_path = None
        temp_wav_path = None

        # Manual FFmpeg conversion to WAV to bypass pydub/librosa quirks
        # We know AudioSegment.converter points to a valid ffmpeg.exe
        
        # 1. Save upload to temp file with original extension
        import tempfile
        import subprocess
        original_ext = os.path.splitext(ref_audio.filename)[1] or ".tmp"
        
        # Create input temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext, prefix="tts_upload_") as tmp_in:
            tmp_in.write(audio_bytes)
            temp_input_path = tmp_in.name
            
        # Create output temp file (WAV)
        # We close it immediately so ffmpeg can write to it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", prefix="tts_converted_") as tmp_out:
            temp_wav_path = tmp_out.name
            
        try:
            # 2. Convert to WAV using ffmpeg
            print(f"Converting {temp_input_path} to {temp_wav_path} using {AudioSegment.converter}...")
            print(f"Input file size: {os.path.getsize(temp_input_path)} bytes")
            
            try:
                subprocess.run(
                    [AudioSegment.converter, "-y", "-i", temp_input_path, temp_wav_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError as e:
                err_msg = f"FFmpeg conversion failed: {e.stderr.decode('utf-8', errors='ignore')}"
                print(err_msg)
                raise HTTPException(status_code=500, detail=err_msg)

            
            # 3. Pass converted WAV to model
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language if language != "Auto" else "Auto",
                ref_audio=temp_wav_path,
                ref_text=ref_text
            )
            
        finally:
            # 4. Cleanup both files
            for path in [temp_input_path, temp_wav_path]:
                 try:
                    if os.path.exists(path):
                        os.remove(path)
                 except Exception as e:
                    print(f"Warning: Could not delete temp file {path}: {e}")

        
        return audio_to_stream(wavs[0], sr, response_format)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}")
        # Cleanup is handled in finally block of inner try/except
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        manager.touch()




if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
